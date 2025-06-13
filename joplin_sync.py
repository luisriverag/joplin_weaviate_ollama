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
from datetime import datetime

from dotenv import load_dotenv
import weaviate
from weaviate.config import AdditionalConfig, Timeout
from weaviate.classes.config import Configure, Property, DataType
from weaviate.classes.query import Filter

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

# ──────────────────── enhanced content-hash + timestamp cache ─────────────
_seen_hashes: set[str] = set()

def _file_sha1(path: Path, blocksize: int = 65536) -> str:
    sha1 = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(blocksize), b""):
            sha1.update(chunk)
    return sha1.hexdigest()

def _get_file_info(path: Path) -> Dict[str, Any]:
    """Get file hash, modification time, and size for change detection."""
    try:
        stat = path.stat()
        return {
            "hash": _file_sha1(path),
            "mtime": stat.st_mtime,
            "size": stat.st_size,
            "path": str(path)
        }
    except Exception as e:
        logging.error("File info error for %s: %s", path, e)
        return {}

# ────────────────── enhanced cache system ──────────────────

class FileCache:
    """Enhanced cache with checksum + timestamp-based change detection."""
    
    def __init__(self, cache_file: Path = CACHE_FILE):
        self.cache_file = cache_file
        self._cache: Dict[str, Dict[str, Any]] = {}
        self.load()
    
    def load(self):
        """Load cache from disk."""
        if self.cache_file.exists():
            try:
                self._cache = json.loads(self.cache_file.read_text())
            except Exception as e:
                logging.error("Cache load error: %s", e)
                self._cache = {}
        else:
            self._cache = {}
    
    def save(self):
        """Save cache to disk."""
        try:
            self.cache_file.write_text(json.dumps(self._cache, indent=2))
        except Exception as e:
            logging.error("Cache save error: %s", e)
    
    def has_changed(self, path: Path) -> bool:
        """Check if file has changed since last cache."""
        path_str = str(path)
        if path_str not in self._cache:
            return True
        
        cached_info = self._cache[path_str]
        current_info = _get_file_info(path)
        
        if not current_info:  # Error getting file info
            return True
        
        # Check if any key attributes changed
        return (
            current_info.get("hash") != cached_info.get("hash") or
            current_info.get("mtime") != cached_info.get("mtime") or
            current_info.get("size") != cached_info.get("size")
        )
    
    def update(self, path: Path):
        """Update cache entry for a file."""
        info = _get_file_info(path)
        if info:
            self._cache[str(path)] = info
    
    def remove_missing(self, existing_paths: List[Path]):
        """Remove cache entries for files that no longer exist."""
        existing_str = {str(p) for p in existing_paths}
        to_remove = [path for path in self._cache.keys() if path not in existing_str]
        for path in to_remove:
            del self._cache[path]

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

def _process_file(path: Path, *, img_timeout: int, cache: FileCache) -> Optional[Note]:
    """Return a note dict or None if unsupported, duplicate, or empty."""
    
    # Check if file has changed
    if not cache.has_changed(path):
        return None
    
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
        "last_modified": datetime.fromtimestamp(path.stat().st_mtime).isoformat(),
        "file_size": path.stat().st_size,
    }

    if suffix == ".md":
        fm = parse_front_matter(text)
        meta.update(fm)  # pulls 'tags' if any

    # Update cache for this file
    cache.update(path)
    
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
    
    # Initialize cache
    cache = FileCache()
    
    # Clean up cache for missing files
    cache.remove_missing(paths)
    
    # Apply test limit to file scanning if specified
    if test_limit and test_limit > 0:
        paths = paths[:test_limit]
        print(f"🧪 Test mode: limiting scan to first {len(paths)} files")
    
    # Filter paths that need processing
    changed_paths = [p for p in paths if cache.has_changed(p)]
    
    print(f"📊 Found {len(paths)} total files, {len(changed_paths)} changed/new")
    
    if not changed_paths:
        cache.save()
        return []
    
    bar = tqdm(total=len(changed_paths), desc="Processing changed files", unit="file") if (show_progress and tqdm) else None
    notes: list[Note] = []

    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(_process_file, p, img_timeout=img_timeout, cache=cache) for p in changed_paths]
        for fut in cf.as_completed(futures):
            res = fut.result()
            if res:
                notes.append(res)
            if bar:
                bar.update(1)
    
    if bar:
        bar.close()
    
    # Save cache after processing
    cache.save()
    
    return notes

# ───────────────────── Weaviate client helper ──────────────────

def weaviate_client() -> weaviate.WeaviateClient:
    return weaviate.connect_to_local(
        host="localhost", port=8080, grpc_port=50051,
        additional_config=AdditionalConfig(timeout=Timeout(init=5)),
    )

# ─────────────────────── enhanced schema with BM25 ─────────────────────────

def ensure_schema(client: weaviate.WeaviateClient, index_name: str):
    """Create the collection if it doesn't exist with BM25 + vector search support."""
    try:
        # Check if collection exists
        if client.collections.exists(index_name):
            print(f"✅ Collection '{index_name}' already exists")
            return
        
        # Create new collection with BM25 + vector search
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
                Property(name="last_modified", data_type=DataType.DATE),
                Property(name="file_size", data_type=DataType.INT),
                Property(name="content_hash", data_type=DataType.TEXT),
            ],
            # Enable BM25 for hybrid search
            inverted_index_config=Configure.inverted_index(
                bm25=Configure.BM25(
                    b=0.75,  # Controls document length normalization
                    k1=1.2   # Term frequency saturation point
                )
            )
        )
        print(f"✅ Created collection '{index_name}' with hybrid search support")
        
    except Exception as e:
        logging.error("Schema creation error: %s", e)
        print(f"⚠️  Schema error: {e}")
        raise

# ─────────────────────── enhanced upload with deduplication ───────────────────────────

def remove_existing_documents(client: weaviate.WeaviateClient, index_name: str, file_paths: List[str]):
    """Remove existing documents for files that are being re-processed."""
    if not file_paths:
        return
    
    try:
        collection = client.collections.get(index_name)
        
        # Delete documents matching the file paths
        for path in file_paths:
            collection.data.delete_many(
                where=Filter.by_property("path").equal(path)
            )
        
        print(f"🗑️  Removed existing documents for {len(file_paths)} updated files")
        
    except Exception as e:
        logging.error("Document removal error: %s", e)
        print(f"⚠️  Document removal error: {e}")

def upload_batch(notes_batch: List[Note], splitter, vectorstore, processed_cache: List[Note]) -> int:
    """Upload a batch of notes with enhanced metadata."""
    texts, metas = [], []
    for note in notes_batch:
        content_hash = hashlib.sha256(note["content"].encode()).hexdigest()
        
        doc_meta = {
            "title": note["title"],
            "path": note["path"],
            "folder": note.get("folder", ""),
            "tags": note.get("tags", []),
            "source": "joplin",
            "last_modified": note.get("last_modified", ""),
            "file_size": note.get("file_size", 0),
            "content_hash": content_hash,
        }
        
        for chunk in splitter.split_text(note["content"]):
            texts.append(chunk)
            metas.append(doc_meta.copy())  # Each chunk gets its own metadata

    if texts:
        vectorstore.add_texts(texts, metas)
        processed_cache.extend(notes_batch)
    return len(texts)

def upload(notes: List[Note], *, index_name: str, show_progress: bool = False, batch_size: int = 1000):
    """Enhanced upload with deduplication and hybrid search support."""
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
        
        # Remove existing documents for files being updated
        file_paths = [note["path"] for note in notes]
        remove_existing_documents(client, index_name, file_paths)
        
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

# ─────────────────────── hybrid search functionality ───────────────────────────

class HybridSearch:
    """Hybrid search combining BM25 (keyword) and vector similarity."""
    
    def __init__(self, client: weaviate.WeaviateClient, index_name: str, embedder):
        self.client = client
        self.collection = client.collections.get(index_name)
        self.embedder = embedder
    
    def search(self, query: str, limit: int = 10, alpha: float = 0.7) -> List[Dict]:
        """
        Hybrid search with configurable weighting.
        
        Args:
            query: Search query
            limit: Maximum results to return
            alpha: Weight for vector search (0.0 = pure BM25, 1.0 = pure vector)
        
        Returns:
            List of search results with metadata
        """
        try:
            # Perform hybrid search
            response = self.collection.query.hybrid(
                query=query,
                alpha=alpha,  # Balance between BM25 and vector search
                limit=limit,
                return_metadata=["score"]
            )
            
            results = []
            for obj in response.objects:
                result = {
                    "content": obj.properties.get("text", ""),
                    "title": obj.properties.get("title", ""),
                    "path": obj.properties.get("path", ""),
                    "folder": obj.properties.get("folder", ""),
                    "tags": obj.properties.get("tags", []),
                    "score": obj.metadata.score if obj.metadata else 0,
                    "last_modified": obj.properties.get("last_modified", ""),
                }
                results.append(result)
            
            return results
            
        except Exception as e:
            logging.error("Hybrid search error: %s", e)
            return []

def search_notes(query: str, index_name: str = DEFAULT_INDEX, limit: int = 10, 
                search_type: str = "hybrid", alpha: float = 0.7):
    """
    Search notes with multiple search modes.
    
    Args:
        query: Search query
        index_name: Weaviate collection name
        limit: Maximum results
        search_type: "hybrid", "vector", or "bm25"
        alpha: Hybrid search balance (0.0=pure BM25, 1.0=pure vector)
    """
    client = None
    try:
        client = weaviate_client()
        
        if search_type == "hybrid":
            embedder = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
            searcher = HybridSearch(client, index_name, embedder)
            results = searcher.search(query, limit=limit, alpha=alpha)
        
        elif search_type == "vector":
            collection = client.collections.get(index_name)
            embedder = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
            query_vector = embedder.embed_query(query)
            
            response = collection.query.near_vector(
                near_vector=query_vector,
                limit=limit,
                return_metadata=["distance"]
            )
            
            results = []
            for obj in response.objects:
                result = {
                    "content": obj.properties.get("text", ""),
                    "title": obj.properties.get("title", ""),
                    "path": obj.properties.get("path", ""),
                    "folder": obj.properties.get("folder", ""),
                    "tags": obj.properties.get("tags", []),
                    "distance": obj.metadata.distance if obj.metadata else 1.0,
                }
                results.append(result)
        
        elif search_type == "bm25":
            collection = client.collections.get(index_name)
            response = collection.query.bm25(
                query=query,
                limit=limit,
                return_metadata=["score"]
            )
            
            results = []
            for obj in response.objects:
                result = {
                    "content": obj.properties.get("text", ""),
                    "title": obj.properties.get("title", ""),
                    "path": obj.properties.get("path", ""),
                    "folder": obj.properties.get("folder", ""),
                    "tags": obj.properties.get("tags", []),
                    "score": obj.metadata.score if obj.metadata else 0,
                }
                results.append(result)
        
        else:
            raise ValueError(f"Unknown search type: {search_type}")
        
        return results
        
    finally:
        if client:
            client.close()

# ─────────────────── cache analysis and maintenance ───────────────────────────

def cache_info(cache_file: Path = CACHE_FILE):
    """Display cache statistics and information."""
    if not cache_file.exists():
        print("📊 Cache file does not exist yet")
        return
    
    try:
        cache_data = json.loads(cache_file.read_text())
        total_files = len(cache_data)
        
        print(f"📊 Cache Information:")
        print(f"   • Total cached files: {total_files}")
        print(f"   • Cache file size: {cache_file.stat().st_size / 1024:.1f} KB")
        
        # Analyze file types
        extensions = {}
        for file_path in cache_data.keys():
            ext = Path(file_path).suffix.lower()
            extensions[ext] = extensions.get(ext, 0) + 1
        
        print(f"   • File types:")
        for ext, count in sorted(extensions.items(), key=lambda x: x[1], reverse=True):
            print(f"     - {ext or 'no extension'}: {count}")
            
        # Check for orphaned entries (files that no longer exist)
        orphaned = []
        for file_path in cache_data.keys():
            if not Path(file_path).exists():
                orphaned.append(file_path)
        
        if orphaned:
            print(f"   • Orphaned entries: {len(orphaned)}")
            if len(orphaned) <= 5:
                for path in orphaned:
                    print(f"     - {path}")
            else:
                print(f"     - {orphaned[0]}")
                print(f"     - ... and {len(orphaned) - 1} more")
        else:
            print(f"   • No orphaned entries found")
            
    except Exception as e:
        print(f"⚠️  Error reading cache: {e}")

def clean_cache(cache_file: Path = CACHE_FILE, dry_run: bool = True):
    """Clean orphaned entries from cache."""
    if not cache_file.exists():
        print("📊 Cache file does not exist")
        return
    
    try:
        cache_data = json.loads(cache_file.read_text())
        original_count = len(cache_data)
        
        # Find orphaned entries
        to_remove = []
        for file_path in cache_data.keys():
            if not Path(file_path).exists():
                to_remove.append(file_path)
        
        if not to_remove:
            print("✅ No orphaned entries to clean")
            return
        
        print(f"🧹 Found {len(to_remove)} orphaned entries")
        
        if dry_run:
            print("   (DRY RUN - use --clean-cache --force to actually remove)")
            for path in to_remove[:10]:  # Show first 10
                print(f"   - {path}")
            if len(to_remove) > 10:
                print(f"   - ... and {len(to_remove) - 10} more")
        else:
            # Actually remove orphaned entries
            for path in to_remove:
                del cache_data[path]
            
            # Save cleaned cache
            cache_file.write_text(json.dumps(cache_data, indent=2))
            print(f"✅ Removed {len(to_remove)} orphaned entries")
            print(f"   Cache size: {original_count} → {len(cache_data)} files")
            
    except Exception as e:
        print(f"⚠️  Error cleaning cache: {e}")

# ─────────────────────────── CLI ───────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sync",   action="store_true", help="scan note folders")
    ap.add_argument("--upload", action="store_true", help="push to Weaviate")
    ap.add_argument("--search", type=str, help="search query")
    ap.add_argument("--workers", type=int, default=os.cpu_count(),
                    help="threads for scanning (default: logical CPUs)")
    ap.add_argument("--progress", action="store_true", help="show progress bars (needs tqdm)")
    ap.add_argument("--timeout", type=int, default=60, help="max seconds per image OCR")
    ap.add_argument("--batch-size", type=int, default=1000, help="notes per upload batch")
    ap.add_argument("--index", default=DEFAULT_INDEX, help="Weaviate class/index to use")
    ap.add_argument("--test", type=int, default=None, help="test mode: limit to first N notes (e.g., --test 100)")
    ap.add_argument("--search-type", choices=["hybrid", "vector", "bm25"], default="hybrid",
                    help="search mode (default: hybrid)")
    ap.add_argument("--alpha", type=float, default=0.7,
                    help="hybrid search balance: 0.0=pure BM25, 1.0=pure vector (default: 0.7)")
    ap.add_argument("--limit", type=int, default=10, help="max search results (default: 10)")
    
    # Cache management
    ap.add_argument("--cache-info", action="store_true", help="show cache statistics")
    ap.add_argument("--clean-cache", action="store_true", help="clean orphaned cache entries")
    ap.add_argument("--force", action="store_true", help="actually perform cache cleaning (not dry run)")
    
    args = ap.parse_args()

    # Cache management commands
    if args.cache_info:
        cache_info()
        return
    
    if args.clean_cache:
        clean_cache(dry_run=not args.force)
        return

    if args.search:
        print(f"🔍 Searching for: '{args.search}' (mode: {args.search_type})")
        results = search_notes(
            args.search, 
            index_name=args.index, 
            limit=args.limit,
            search_type=args.search_type,
            alpha=args.alpha
        )
        
        if not results:
            print("📭 No results found.")
        else:
            print(f"📊 Found {len(results)} results:")
            for i, result in enumerate(results, 1):
                score_info = ""
                if "score" in result:
                    score_info = f" (score: {result['score']:.3f})"
                elif "distance" in result:
                    score_info = f" (distance: {result['distance']:.3f})"
                
                print(f"\n{i}. {result['title']}{score_info}")
                print(f"   📁 {result['folder']} | 🏷️ {', '.join(result['tags'])}")
                print(f"   📄 {result['path']}")
                
                # Show snippet of content
                content = result['content'][:200]
                if len(result['content']) > 200:
                    content += "..."
                print(f"   💬 {content}")
    
    elif args.sync:
        all_notes = load_notes(workers=args.workers, show_progress=args.progress, 
                               img_timeout=args.timeout, test_limit=args.test)
        print(f"🔄 {len(all_notes)} new/changed notes detected.")
        
        if args.upload and all_notes:
            upload(all_notes, index_name=args.index, show_progress=args.progress, 
                   batch_size=args.batch_size)
    
    else:
        ap.print_help()

if __name__ == "__main__":
    main()