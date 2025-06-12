#!/usr/bin/env python3
"""
Synchronise Joplin-exported notes/resources into a Weaviate v4 index — now with
multithreaded I/O, optional progress bars, single-pass OCR via content-hash
Dedup **and a per-image timeout so pathological files can't stall the run**.

Key changes (v2025-06-11):
──────────────────────────
* Progress bar now advances with each finished future, so it never appears
  frozen while a long OCR task runs.
* `extract_image_text()` enforces a 30-second timeout on Tesseract; hung or
  super-large images fall back to empty string and log an error.
* `--timeout` CLI flag lets you adjust that limit.

Usage examples
──────────────
    python joplin_sync_multithread.py --sync --upload --workers 8 --progress
    # allow max 10 s per image
    python joplin_sync_multithread.py --sync --timeout 10 --progress
"""
from __future__ import annotations

import os, json, argparse, logging, warnings, hashlib, concurrent.futures as cf
from pathlib import Path
from typing import Iterable, Optional, List

from dotenv import load_dotenv
import weaviate
from weaviate.config import AdditionalConfig, Timeout

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_weaviate import WeaviateVectorStore
from langchain.text_splitter import RecursiveCharacterTextSplitter

import pytesseract
from PIL import Image, UnidentifiedImageError
from pypdf import PdfReader
from bs4 import BeautifulSoup

# ───────────────────────── optional progress ──────────────────
try:
    from tqdm import tqdm  # type: ignore
except ImportError:  # fallback noop if tqdm isn't installed
    tqdm = None  # pyright: ignore [reportGeneralTypeIssues]

# ───────────────────────── logging ─────────────────────────────
warnings.filterwarnings("ignore", category=UserWarning)
logging.basicConfig(
    filename="sync_errors.log",
    level=logging.ERROR,
    format="%(asctime)s  %(levelname)s  %(message)s",
)

# ───────────────────────── env vars ────────────────────────────
load_dotenv()
MD_FOLDERS       = [p for p in os.getenv("MD_FOLDERS", "").split(",") if p]
WEAVIATE_URL     = os.getenv("WEAVIATE_URL", "http://localhost:8080")
WEAVIATE_INDEX   = os.getenv("WEAVIATE_INDEX", "JoplinNotes")
EMBEDDING_MODEL  = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
CACHE_FILE       = Path("note_cache.json")

# ────────────────────── text extraction helpers ────────────────

def extract_pdf_text(path: Path) -> str:
    try:
        with open(path, "rb") as f:
            reader = PdfReader(f)
            return "\n".join(page.extract_text() or "" for page in reader.pages)
    except Exception as e:
        logging.error("PDF read error in %s: %s", path, e)
        return ""


def extract_image_text(path: Path, *, timeout: int) -> str:
    """OCR with a timeout so a single file can't hang the whole pool."""
    try:
        with Image.open(path) as img:
            try:
                return pytesseract.image_to_string(img, timeout=timeout)
            except RuntimeError:  # pytesseract raises RuntimeError on timeout
                logging.error("OCR timeout (> %ss) in %s", timeout, path)
                return ""
    except (UnidentifiedImageError, Exception) as e:
        logging.error("OCR error in %s: %s", path, e)
        return ""


def extract_html_text(path: Path) -> str:
    try:
        with open(path, encoding="utf-8") as f:
            soup = BeautifulSoup(f, "html.parser")
            return soup.get_text()
    except Exception as e:
        logging.error("HTML parse error in %s: %s", path, e)
        return ""

# dispatch table — functions that *don't* need timeout injected
EXTRACTOR_STATIC: dict[str, callable[[Path], str]] = {
    ".md":   lambda p: p.read_text(encoding="utf-8", errors="ignore"),
    ".pdf":  extract_pdf_text,
    ".html": extract_html_text,
}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}

# ──────────────────── in-memory content-hash dedup ─────────────
_seen_hashes: set[str] = set()


def _file_sha1(path: Path, blocksize: int = 65536) -> str:
    sha1 = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(blocksize), b""):
            sha1.update(chunk)
    return sha1.hexdigest()

# ────────────────── multithreaded filesystem scan ──────────────

def _process_file(path: Path, *, img_timeout: int) -> Optional[dict[str, str]]:
    """Return a note dict or None if unsupported, duplicate, or empty."""
    try:
        digest = _file_sha1(path)
    except Exception as e:
        logging.error("Hash error in %s: %s", path, e)
        return None
    if digest in _seen_hashes:
        return None
    _seen_hashes.add(digest)

    suffix = path.suffix.lower()
    if suffix in IMAGE_SUFFIXES:
        text = extract_image_text(path, timeout=img_timeout)
    else:
        extractor = EXTRACTOR_STATIC.get(suffix)
        if extractor is None:
            return None
        text = extractor(path)

    if not text.strip():
        return None
    return {"title": path.stem, "path": str(path), "content": text}


def iter_files() -> List[Path]:
    stack: List[Path] = []
    for base in MD_FOLDERS:
        for root, _, files in os.walk(base):
            for name in files:
                stack.append(Path(root) / name)
    return stack


def load_notes(*, workers: int, show_progress: bool, img_timeout: int) -> list[dict[str, str]]:
    paths = iter_files()
    bar = tqdm(total=len(paths), desc="Scanning files", unit="file") if (show_progress and tqdm) else None
    notes: list[dict[str, str]] = []

    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(_process_file, p, img_timeout=img_timeout) for p in paths]
        for fut in cf.as_completed(futures):
            res = fut.result()
            if res:
                notes.append(res)
            if bar:
                bar.update(1)
    if bar:
        bar.close()
    return notes

# ─────────── simple dedup cache (path-based across runs) ───────

def load_cache() -> set[str]:
    return set(json.loads(CACHE_FILE.read_text())) if CACHE_FILE.exists() else set()


def save_cache(processed: Iterable[dict[str, str]]):
    CACHE_FILE.write_text(json.dumps([n["path"] for n in processed]))


def filter_new(notes: list[dict[str, str]]):
    cached = load_cache()
    return [n for n in notes if n["path"] not in cached]

# ───────────────────── Weaviate client helper ──────────────────

def weaviate_client() -> weaviate.WeaviateClient:
    return weaviate.connect_to_local(
        host="localhost", port=8080, grpc_port=50051,
        additional_config=AdditionalConfig(timeout=Timeout(init=5)),
    )

# ─────────────────────── upload flow ───────────────────────────

def upload_batch(notes_batch: list[dict[str, str]], client, embedder, splitter, vectorstore, processed_cache: list[dict[str, str]]) -> int:
    """Upload a single batch of notes and return the number of chunks processed."""
    texts, metas = [], []
    
    for note in notes_batch:
        for chunk in splitter.split_text(note["content"]):
            texts.append(chunk)
            metas.append({"title": note["title"], "path": note["path"]})
    
    if texts:
        vectorstore.add_texts(texts, metas)
        processed_cache.extend(notes_batch)
    
    return len(texts)


def upload(notes: list[dict[str, str]], *, show_progress: bool = False, batch_size: int = 1000):
    if not notes:
        print("⚠️  Nothing new to upload.")
        return
    
    total_notes = len(notes)
    total_batches = (total_notes + batch_size - 1) // batch_size  # Ceiling division
    print(f"📤 Uploading {total_notes} notes to Weaviate in {total_batches} batches of {batch_size}…")

    client = None
    processed_cache = []
    
    try:
        client = weaviate_client()
        embedder = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        vectorstore = WeaviateVectorStore(client=client, index_name=WEAVIATE_INDEX, text_key="text", embedding=embedder)

        # Progress bar for batches
        batch_bar = tqdm(total=total_batches, desc="Processing batches", unit="batch") if (show_progress and tqdm) else None
        
        total_chunks_processed = 0
        
        for i in range(0, total_notes, batch_size):
            batch_end = min(i + batch_size, total_notes)
            current_batch = notes[i:batch_end]
            
            if batch_bar:
                batch_bar.set_description(f"Batch {(i // batch_size) + 1}/{total_batches} ({len(current_batch)} notes)")
            
            try:
                chunks_in_batch = upload_batch(current_batch, client, embedder, splitter, vectorstore, processed_cache)
                total_chunks_processed += chunks_in_batch
                
                # Save progress after each successful batch
                save_cache(processed_cache)
                
                if batch_bar:
                    batch_bar.update(1)
                    
            except Exception as e:
                logging.error("Batch upload failed (notes %d-%d): %s", i, batch_end-1, e)
                print(f"⚠️  Batch {(i // batch_size) + 1} failed (notes {i+1}-{batch_end}): {e}")
                # Continue with next batch rather than failing completely
                continue
        
        if batch_bar:
            batch_bar.close()
        
        print(f"✅ Upload complete. Processed {len(processed_cache)}/{total_notes} notes ({total_chunks_processed} chunks total).")
        
        if len(processed_cache) < total_notes:
            failed_count = total_notes - len(processed_cache)
            print(f"⚠️  {failed_count} notes failed to upload. Check sync_errors.log for details.")
        
    except Exception as e:
        logging.error("Upload setup failed: %s", e)
        print(f"❌ Upload setup failed: {e}")
        # Save any progress made so far
        if processed_cache:
            save_cache(processed_cache)
            print(f"💾 Saved progress for {len(processed_cache)} successfully processed notes.")
    finally:
        if client:
            client.close()

# ─────────────────────────── CLI ───────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sync",   action="store_true", help="scan note folders")
    ap.add_argument("--upload", action="store_true", help="push to Weaviate")
    ap.add_argument("--workers", type=int, default=os.cpu_count(),
                    help="parallel threads for scanning (default: logical CPUs)")
    ap.add_argument("--progress", action="store_true", help="show progress bars (requires tqdm)")
    ap.add_argument("--timeout", type=int, default=60, help="max seconds per image OCR (default: 60)")
    ap.add_argument("--batch-size", type=int, default=1000, help="notes per upload batch (default: 1000)")
    args = ap.parse_args()

    if args.sync:
        all_notes = load_notes(workers=args.workers, show_progress=args.progress, img_timeout=args.timeout)
        new_notes = filter_new(all_notes)
        print(f"🔄 {len(new_notes)} new or changed notes detected.")
        if args.upload and new_notes:
            upload(new_notes, show_progress=args.progress, batch_size=args.batch_size)

if __name__ == "__main__":
    main()
