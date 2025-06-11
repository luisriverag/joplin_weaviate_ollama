#!/usr/bin/env python3

from __future__ import annotations

import os, json, argparse, logging, warnings, functools, concurrent.futures as cf
from pathlib import Path
from typing import Iterable, Optional

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
RESOURCES_FOLDER = os.getenv("RESOURCES_FOLDER", "")  # unused but kept for compat
WEAVIATE_URL     = os.getenv("WEAVIATE_URL", "http://localhost:8080")
WEAVIATE_INDEX   = os.getenv("WEAVIATE_INDEX", "JoplinNotes")
EMBEDDING_MODEL  = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

CACHE_FILE = Path("note_cache.json")

# ── text extraction helpers ─────────────────────────────────────

def extract_pdf_text(path: Path) -> str:
    try:
        with open(path, "rb") as f:
            reader = PdfReader(f)
            return "\n".join(page.extract_text() or "" for page in reader.pages)
    except Exception as e:
        logging.error("PDF read error in %s: %s", path, e)
        return ""


def extract_image_text(path: Path) -> str:
    try:
        with Image.open(path) as img:
            return pytesseract.image_to_string(img)
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


# dispatch table, keeps decision logic out of the hot path
EXTRACTOR: dict[str, callable[[Path], str]] = {
    ".md":   lambda p: p.read_text(encoding="utf-8", errors="ignore"),
    ".pdf":  extract_pdf_text,
    ".html": extract_html_text,
}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}


def _process_file(path: Path) -> Optional[dict[str, str]]:
    """Return a note dict or None if unsupported/empty."""
    suffix = path.suffix.lower()
    extractor = EXTRACTOR.get(suffix) or (extract_image_text if suffix in IMAGE_SUFFIXES else None)
    if extractor is None:
        return None  # skip unknown types

    text: str = ""
    try:
        text = extractor(path)
    except Exception as e:
        logging.error("Read error in %s: %s", path, e)

    if not text.strip():
        return None
    return {"title": path.stem, "path": str(path), "content": text}


# ── multithreaded filesystem scan ──────────────────────────────

def iter_files() -> Iterable[Path]:
    for base in MD_FOLDERS:
        for root, _, files in os.walk(base):
            for name in files:
                yield Path(root) / name


def load_notes(workers: int | None = None) -> list[dict[str, str]]:
    """Scan MD_FOLDERS in parallel, returning extracted notes."""
    workers = workers or os.cpu_count() or 4
    notes: list[dict[str, str]] = []

    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        for res in ex.map(_process_file, iter_files(), chunksize=32):
            if res:
                notes.append(res)
    return notes


# ── simple dedup cache ─────────────────────────────────────────

def load_cache() -> set[str]:
    return set(json.loads(CACHE_FILE.read_text())) if CACHE_FILE.exists() else set()


def save_cache(processed: Iterable[dict[str, str]]):
    CACHE_FILE.write_text(json.dumps([n["path"] for n in processed]))


def filter_new(notes: list[dict[str, str]]):
    cached = load_cache()
    return [n for n in notes if n["path"] not in cached]


# ── Weaviate client helper ─────────────────────────────────────

def weaviate_client() -> weaviate.WeaviateClient:
    return weaviate.connect_to_local(
        host="localhost", port=8080, grpc_port=50051,
        additional_config=AdditionalConfig(timeout=Timeout(init=5)),
    )


# ── upload flow ────────────────────────────────────────────────

def upload(notes: list[dict[str, str]]):
    if not notes:
        print("⚠️  Nothing new to upload.")
        return
    print(f"📤 Uploading {len(notes)} notes to Weaviate…")

    client      = weaviate_client()
    embedder    = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    splitter    = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    vectorstore = WeaviateVectorStore(
        client=client, index_name=WEAVIATE_INDEX, text_key="text", embedding=embedder,
    )

    texts, metas = [], []
    for note in notes:
        for chunk in splitter.split_text(note["content"]):
            texts.append(chunk)
            metas.append({"title": note["title"], "path": note["path"]})

    # Batched inserts (WeaviateVectorStore handles internal batching) — could
    # also be threaded but network latency usually dominates once embedding is
    # computed, so keep it simple/stable.
    vectorstore.add_texts(texts, metas)

    client.close()
    save_cache(notes)
    print("✅ Upload complete.")


# ── CLI ────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sync",   action="store_true", help="scan note folders")
    ap.add_argument("--upload", action="store_true", help="push to Weaviate")
    ap.add_argument("--workers", type=int, default=None,
                    help="number of parallel workers (default: cpu cores)")
    args = ap.parse_args()

    if args.sync:
        all_notes = load_notes(workers=args.workers)
        new_notes = filter_new(all_notes)
        print(f"🔄 {len(new_notes)} new or changed notes detected.")
        if args.upload and new_notes:
            upload(new_notes)


if __name__ == "__main__":
    main()
