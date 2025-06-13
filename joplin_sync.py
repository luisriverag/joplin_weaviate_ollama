#!/usr/bin/env python3

from __future__ import annotations

import os
import re
import json
import argparse
import logging
import warnings
import hashlib
import concurrent.futures as cf
from pathlib import Path
from typing import Iterable, Optional, List, Dict, Any

from dotenv import load_dotenv
import weaviate
from weaviate.config import AdditionalConfig, Timeout
from weaviate.classes.config import Configure, Property, DataType

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
DEFAULT_INDEX    = os.getenv("WEAVIATE_INDEX", "Documents")
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

# ────────────── front‑matter & helper parsing ─────────────────
TAG_RE = re.compile(r"^tags:\s*(?P<tags>.+)", re.I | re.M)
META_BLOCK_RE = re.compile(r"<!--(.*?)-->", re.S)
COMMA_SPLIT_RE = re.compile(r"[;,]\s*|\n")  # split on comma, semicolon, or newline


def parse_front_matter(text: str) -> Dict[str, Any]:
    """Return dict with optional 'tags' from Joplin HTML comment header."""
    m_block = META_BLOCK_RE.search(text)
    if not m_block:
        return {}
    meta = m_block.group(1)

    tags_match = TAG_RE.search(meta)
    if not tags_match:
        return {}
    raw = tags_match.group("tags")
    tags = [t.strip() for t in COMMA_SPLIT_RE.split(raw) if t.strip()]
    return {"tags": tags}

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

def _relative_folder(path: Path) -> str:
    """Return folder part relative to the first matching MD_FOLDERS root."""
    for base in MD_FOLDERS:
        try:
            rel = path.relative_to(base)
            return str(rel.parent)
        except ValueError:
            continue
    return str(path.parent)


Note = Dict[str, Any]

def _process_file(path: Path, *, img_timeout: int) -> Optional[Note]:
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

    # Extract main text
    if suffix in IMAGE_SUFFIXES:
        text = extract_image_text(path, timeout=img_timeout)
    else:
        extractor = EXTRACTOR_STATIC.get(suffix)
        if extractor is None:
            return None
        text = extractor(path)

    if not text.strip():
        return None

    # Metadata enrichment
    meta: Note = {
        "title": path.stem,
        "path": str(path),
        "content": text,
        "folder": _relative_folder(path),
        "tags": [],
    }

    if suffix == ".md":
        fm = parse_front_matter(text)
        meta.update(fm)  # pulls 'tags' if any

    return meta


def iter_files() -> List[Path]:
    stack: List[Path] = []
    for base in MD_FOLDERS:
        for root, _, files in os.walk(base):
            for name in files:
                stack.append(Path(root) / name)
    return stack


def load_notes(*, workers: int, show_progress: bool, img_timeout: int, test_limit: Optional[int] = None) -> list[Note]:
    paths = iter_files()
    
    # Apply test limit to file scanning if specified
    if test_limit and test_limit > 0:
        paths = paths[:test_limit]
        print(f"🧪 Test mode: limiting scan to first {len(paths)} files")
    
    bar = tqdm(total=len(paths), desc="Scanning files", unit="file") if (show_progress and tqdm) else None
    notes: list[Note] = []

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


def save_cache(processed: Iterable[Note]):
    CACHE_FILE.write_text(json.dumps([n["path"] for n in processed]))


def filter_new(notes: list[Note]):
    cached = load_cache()
    return [n for n in notes if n["path"] not in cached]

# ───────────────────── Weaviate client helper ──────────────────

def weaviate_client() -> weaviate.WeaviateClient:
    return weaviate.connect_to_local(
        host="localhost", port=8080, grpc_port=50051,
        additional_config=AdditionalConfig(timeout=Timeout(init=5)),
    )

# ─────────────────────── schema helper ─────────────────────────

def ensure_schema(client: weaviate.WeaviateClient, index_name: str):
    """Create the collection if it doesn't exist with required properties."""
    try:
        # Check if collection exists
        if client.collections.exists(index_name):
            print(f"✅ Collection '{index_name}' already exists")
            return
        
        # Create new collection with properties
        client.collections.create(
            name=index_name,
            vectorizer_config=Configure.Vectorizer.none(),  # We'll handle embeddings ourselves
            properties=[
                Property(name="text", data_type=DataType.TEXT),
                Property(name="title", data_type=DataType.TEXT),
                Property(name="path", data_type=DataType.TEXT),
                Property(name="folder", data_type=DataType.TEXT),
                Property(name="tags", data_type=DataType.TEXT_ARRAY),
                Property(name="source", data_type=DataType.TEXT),
            ]
        )
        print(f"✅ Created collection '{index_name}' with schema")
        
    except Exception as e:
        logging.error("Schema creation error: %s", e)
        print(f"⚠️  Schema error: {e}")
        raise

# ─────────────────────── upload flow ───────────────────────────

def upload_batch(notes_batch: List[Note], splitter, vectorstore, processed_cache: List[Note]) -> int:
    texts, metas = [], []
    for note in notes_batch:
        doc_meta = {
            "title": note["title"],
            "path": note["path"],
            "folder": note.get("folder", ""),
            "tags": note.get("tags", []),
            "source": "joplin",
        }
        for chunk in splitter.split_text(note["content"]):
            texts.append(chunk)
            metas.append(doc_meta)

    if texts:
        vectorstore.add_texts(texts, metas)
        processed_cache.extend(notes_batch)
    return len(texts)


def upload(notes: List[Note], *, index_name: str, show_progress: bool = False, batch_size: int = 1000):
    if not notes:
        print("⚠️  Nothing new to upload.")
        return

    total_notes = len(notes)
    total_batches = (total_notes + batch_size - 1) // batch_size
    print(f"📤 Uploading {total_notes} notes → '{index_name}' in {total_batches} batch(es)…")

    client = None
    processed_cache: List[Note] = []

    try:
        client = weaviate_client()
        ensure_schema(client, index_name)
        embedder = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        vectorstore = WeaviateVectorStore(
            client=client,
            index_name=index_name,
            text_key="text",
            embedding=embedder,
        )

        batch_bar = tqdm(total=total_batches, desc="Processing", unit="batch") if (show_progress and tqdm) else None
        total_chunks = 0

        for i in range(0, total_notes, batch_size):
            batch = notes[i:i + batch_size]
            if batch_bar:
                batch_bar.set_description(f"Batch {(i // batch_size)+1}/{total_batches} ({len(batch)} notes)")
            try:
                total_chunks += upload_batch(batch, splitter, vectorstore, processed_cache)
                save_cache(processed_cache)
                if batch_bar:
                    batch_bar.update(1)
            except Exception as e:
                logging.error("Batch %d failed: %s", (i // batch_size)+1, e)
                print(f"⚠️  Batch {(i // batch_size)+1} failed: {e}")
                continue
        if batch_bar:
            batch_bar.close()
        print(f"✅ Upload complete. {total_chunks} chunks across {len(processed_cache)} notes.")
    finally:
        if client:
            client.close()

# ─────────────────────────── CLI ───────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sync",   action="store_true", help="scan note folders")
    ap.add_argument("--upload", action="store_true", help="push to Weaviate")
    ap.add_argument("--workers", type=int, default=os.cpu_count(),
                    help="threads for scanning (default: logical CPUs)")
    ap.add_argument("--progress", action="store_true", help="show progress bars (needs tqdm)")
    ap.add_argument("--timeout", type=int, default=60, help="max seconds per image OCR")
    ap.add_argument("--batch-size", type=int, default=1000, help="notes per upload batch")
    ap.add_argument("--index", default=DEFAULT_INDEX, help="Weaviate class/index to use")
    ap.add_argument("--test", type=int, default=None, help="test mode: limit to first N notes (e.g., --test 100)")
    args = ap.parse_args()

    if args.sync:
        all_notes = load_notes(workers=args.workers, show_progress=args.progress, 
                               img_timeout=args.timeout, test_limit=args.test)
        new_notes = filter_new(all_notes)
        print(f"🔄 {len(new_notes)} new/changed notes detected.")
        if args.upload and new_notes:
            upload(new_notes, index_name=args.index, show_progress=args.progress, 
                   batch_size=args.batch_size)

if __name__ == "__main__":
    main()
