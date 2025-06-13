# JOPLIN PRIVATE AI

**RAG your notes locally with 100% open‑source stack**

## What is it?

Pipeline that syncs Joplin‑exported Markdown/PDF/images into a local Weaviate vector DB, embeds with Sentence‑Transformers, and lets you chat with them using Ollama. Optional Telegram bot provided.

## Highlights

- ⚡ **Fast multithreaded scan & OCR** — uses all CPU cores
    
- 🕒 **Content‑hash caching** — uploads only new/changed files
    
- 🔍 **Per‑image OCR timeout** — hung files can't block a sync (`--timeout`)
    
- 📊 **Progress bars** — see what's happening with `--progress`
    
- 🧪 **Test mode** — scan only first N files for quick testing (`--test 100`)
    
- 🧠 **Generic RAG stack** — LangChain + Ollama + Weaviate
    
- 🗂️ **Built‑in document classifier** — prioritises personal docs when answering
    
- 🐞 **Debug tools** — inspect retrieved docs or save a JSON classification config
    
- 📱 **Optional Telegram bot** — restricted to a single user ID
    
- 🔧 **Fully configurable via** `.env`
    

## Quick start

```bash
# 1. Clone & install
git clone https://github.com/you/joplin-private-ai.git
cd joplin-private-ai
python -m pip install -r requirements.txt

# 2. System deps
# Ubuntu:
sudo apt install tesseract-ocr
# macOS (brew) / Windows → see Tesseract docs

# Ollama & Docker (Weaviate)
# https://ollama.com/download
docker compose up -d      # starts Weaviate on :8080

# 3. Configure
cp sample.env .env
nano .env                 # point MD_FOLDERS at your Joplin MD export
```

## Sync & upload

```bash
# Test run with first 100 files (recommended for first setup)
python joplin_sync.py --sync --upload --test 100 --progress

# Full run, 8 threads, progress bars
python joplin_sync.py --sync --upload --workers 8 --progress

# Custom OCR timeout and 500‑note batches
python joplin_sync.py --sync --upload --timeout 30 --batch-size 500
```

### Sync options

| Option | Description | Example |
| --- | --- | --- |
| `--sync` | Scan note folders for files | Required |
| `--upload` | Push to Weaviate | Usually combined with `--sync` |
| `--test N` | **Test mode**: Only scan first N files | `--test 50` |
| `--workers N` | Threading (default: CPU count) | `--workers 8` |
| `--progress` | Show progress bars | Recommended |
| `--timeout N` | OCR timeout per image (seconds) | `--timeout 30` |
| `--batch-size N` | Notes per upload batch | `--batch-size 500` |
| `--index NAME` | Weaviate collection name | `--index MyNotes` |

## Chat locally

```bash
python rag_query.py
🧠 > What is my motorbike's license plate?
```

### CLI tricks

- `debug:<query>` – show top retrieved docs with classifications
    
- `no-analysis:<query>` – skip ownership analysis
    
- `save-config my_config.json` – write current classifier template
    

## Telegram bot (optional)

```bash
python telegram_rag_bot.py
# then on Telegram
/start
> What is my motorbike's license plate?
```

Bot replies only to the `TELEGRAM_USER_ID` set in `.env`.

## File overview

| File | Purpose |
| --- | --- |
| `joplin_sync.py` | Multithreaded sync/embedding/upload tool |
| `rag_query.py` | Local interactive RAG CLI with classifier & debug |
| `telegram_rag_bot.py` | Single‑user Telegram interface |
| `sync_and_upload.sh` | Convenience wrapper script |
| `docker-compose.yml` | Weaviate stack |
| `sample.env` / `.env` | Runtime configuration |
| `note_cache.json` | Auto‑generated dedup cache |

## Environment variables

Important keys (see `.env` for full list):

```env
MD_FOLDERS=/path/to/joplin_export1,/path/to/second/folder
WEAVIATE_URL=http://localhost:8080
WEAVIATE_INDEX=Documents
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
OLLAMA_MODEL=llama3:8b
TELEGRAM_BOT_TOKEN=...
TELEGRAM_USER_ID=123456789
```

## Recommended workflow

1.  **First-time setup**: Use test mode to verify everything works
    
    ```bash
    python joplin_sync.py --sync --upload --test 100 --progress
    ```
    
2.  **Full sync**: Once confirmed working, run without test limit
    
    ```bash
    python joplin_sync.py --sync --upload --workers 8 --progress
    ```
    
3.  **Regular updates**: The cache system ensures only new/changed files are processed
    
    ```bash
    # Use your convenience script
    ./sync_and_upload.sh
    ```
    

## FAQ

**Why not point at my live Joplin profile?** A clean Markdown export avoids syncing in‑progress changes and keeps your original notebooks safe.

**Is everything local?** Yes – unless you enable the Telegram bot, which naturally sends your queries and model answers through Telegram's servers.

**Can I use another model/database?** Sure. Swap `EMBEDDING_MODEL`, `OLLAMA_MODEL`, or plug in a remote Weaviate URL.

**What's the test mode for?** Perfect for initial setup or testing changes. Instead of scanning all 50k+ files, `--test 100` only processes the first 100 files, giving you quick feedback on whether your configuration works.

**How do I know if my sync worked?** Check the output for:

- `✅ Created collection 'Documents' with schema` (first run)
- `🔄 X new/changed notes detected`
- `✅ Upload complete. Y chunks across Z notes`
