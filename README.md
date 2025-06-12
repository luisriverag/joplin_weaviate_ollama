# JOPLIN PRIVATE AI

**RAG your notes locally with 100% open‑source stack**

## What is it?

Pipeline that syncs Joplin‑exported Markdown/PDF/images into a local Weaviate vector DB, embeds with Sentence‑Transformers, and lets you chat with them using Ollama. Optional Telegram bot provided.

## Highlights

- ⚡ **Fast multithreaded scan & OCR** — uses all CPU cores
    
- 🕒 **Content‑hash caching** — uploads only new/changed files
    
- 🔍 **Per‑image OCR timeout** — hung files can’t block a sync (`--timeout`)
    
- 📊 **Progress bars** — see what’s happening with `--progress`
    
- 🧠 **Generic RAG stack** — LangChain + Ollama + Weaviate
    
- 🗂️ **Built‑in document classifier** — prioritises personal docs when answering
    
- 🐞 **Debug tools** — inspect retrieved docs or save a JSON classification config
    
- 📱 **Optional Telegram bot** — restricted to a single user ID
    
- 🔧 **Fully configurable via** `.env`
    

## Quick start

```
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

```
# full run, 8 threads, progress bars
python joplin_sync.py --sync --upload --workers 8 --progress
# custom OCR timeout and 500‑note batches
python joplin_sync.py --sync --upload --timeout 30 --batch-size 500
```

## Chat locally

```
python rag_query.py
🧠 > What is my motorbike's license plate?
```

### CLI tricks

- `debug:<query>` – show top retrieved docs with classifications
    
- `no-analysis:<query>` – skip ownership analysis
    
- `save-config my_config.json` – write current classifier template
    

## Telegram bot (optional)

```
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
| `sync_and_upload.sh` | Convenience wrapper (legacy) |
| `docker-compose.yml` | Weaviate stack |
| `sample.env` / `.env` | Runtime configuration |
| `note_cache.json` | Auto‑generated dedup cache |

## Environment variables

Important keys (see `.env` for full list):

```
MD_FOLDERS=/path/to/joplin_export1,/path/to/second/folder
WEAVIATE_URL=http://localhost:8080
WEAVIATE_INDEX=JoplinNotes
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
OLLAMA_MODEL=llama3:8b
TELEGRAM_BOT_TOKEN=...
TELEGRAM_USER_ID=123456789
```

## FAQ

**Why not point at my live Joplin profile?** A clean Markdown export avoids syncing in‑progress changes and keeps your original notebooks safe.

**Is everything local?** Yes – unless you enable the Telegram bot, which naturally sends your queries and model answers through Telegram’s servers.

**Can I use another model/database?** Sure. Swap `EMBEDDING_MODEL`, `OLLAMA_MODEL`, or plug in a remote Weaviate URL.

## NEXT STEPS

Looking into Element / Matrix.org to replace Telegram with an open source platform that allows bots and end to end encryption.
