# JOPLIN + WEAVIATE + OLLAMA = **Joplin PrivateAI**

**💡 100 % local, privacy‑first Knowledge Base** — Chat with your Joplin vault **offline** and **under your full control**.

* * *

## Project vision

Replace cloud Retrieval‑Augmented‑Generation services with a **100 % local stack**:

- **No vendor lock‑in** – all components are open‑source, containerised, and easily swappable.
    
- **Respect your data** – nothing leaves your machine unless *you* enable the optional Telegram bridge.
    
- **Optimised for commodity hardware** – runs smoothly on an 8 GB RAM laptop, scales down to a Raspberry Pi 4 (via *fast* CLI) and up to a GPU workstation.
    
- **Transparent plumbing** – plain Python, Docker, and YAML; every step is auditable.
    

* * *

## Architecture

```text
              ┌─────────────── ① ingestion ────────────────┐
Joplin Export │  Markdown / Attachments / Resources        │
              │           `joplin_sync.py`                 │
              └───────────────────┬────────────────────────┘
                                  │  batches of chunks
                                  ▼
                        ┌───────────────────────┐
                        │      Weaviate         │  ② retrieval
                        │  (BM25 + vectors)     │<──────────────┐
                        └───────────────────────┘               │
                                  ▲                             │
                                  │ GraphQL                     │
                                  │ ③ context                  │
                                  ▼                             │
                        ┌───────────────────────┐               │
                        │       Ollama          │  ④ answer     │
                        │ (local LLM/adapter)   │───────────────┘
                        └───────────────────────┘
                                  ▲
                                  │
    ┌───────────────┬─────────────┴───────────────┐
    │ Terminal CLI  │  Telegram (opt.)   │  REST API† (todo)
    └───────────────┴─────────────┬───────────────┘
                              Conversation
```

*(† a minimal Flask gateway is planned for v2025‑Q3.)*

### Data flow

1.  **Ingestion** – `joplin_sync.py` scans export folders, slices Markdown into ~500 token chunks, OCRs images/PDFs, computes embeddings (dimension 384), and uploads JSON batches.
    
2.  **Storage** – Chunks live in Weaviate with original note path, resource hash, timestamps, and tag list.
    
3.  **Retrieval** – CLIs perform hybrid search (BM25 + cosine) with a configurable α blend, rerank hits with recency & ownership heuristics, and craft the final prompt.
    
4.  **Generation** – A local Ollama model (e.g. *llama3:8b‑instruct*) produces the answer, including citations back to note source paths.
    

* * *

## Component deep‑dive

| Name | Highlights | Language / deps |
| --- | --- | --- |
| **`joplin_sync.py`** | Multithreaded crawler (I/O bound) • OCR via Tesseract • Incremental cache keyed by SHA‑256 • Automatic language detection for OCR • Hybrid schema bootstrapper • Search playground (`--search`) | Python 3.10, Tesseract ≥ 5.0 |
| **`rag_query.py`** | Intent classifier (EN/ES) • Sliding‑window memory (8 turns) • Ownership/procedural/temporal stages • Structured logs (`structlog`) • Rich colour console | Python 3.10, LangChain 0.2, Ollama |
| **`rag_query-fast.py`** | ≤ 2 s cold start • No memory, direct retrieval • YAML pattern matcher • Ideal for Raspberry Pi | Same |
| **`docker-compose.yml`** | Read‑only Weaviate vectorizer • Single‑node Raft with persistent shards • Tweaked limits (`QUERY_DEFAULTS_LIMIT=25`) | Docker ≥ 20.10 |
| `telegram_rag_bot.py` | One‑user hard‑locked • Markdown rendering • `/summary`, `/reset` commands | python‑telegram‑bot 21 |
| `sync_and_upload.sh` | CRON‑friendly wrapper for delta sync & upload | Bash |
| `scripts/migrations/` | Future schema migrations (placeholder) | Python |

* * *

## Requirements

| Category | Minimum | Recommended | Notes |
| --- | --- | --- | --- |
| **OS** | Linux x86‑64 / ARM 64, macOS, Windows 11 WSL2 | Ubuntu 22.04 LTS | On macOS, macFUSE may boost FS perf |
| **Python** | 3.9 | 3.10 | Tested with CPython |
| **CPU** | 2 cores | 4‑8 cores | Heavy OCR benefits |
| **RAM** | 4 GB | 8‑16 GB | Embeddings cached in RAM |
| **Storage** | 5 GB | 20 GB+ | Weaviate grows with vault size |
| **GPU** | optional | 8 GB VRAM | Accelerates Ollama (CUDA / Metal) |
| **Tesseract** | 5.0 | 5.3 + `tesseract-lang` | Add `eng`, `spa`, `deu`, … |

* * *

## Installation

### 1 · Clone & Python packages

```bash
git clone https://github.com/luisriverag/joplin_weaviate_ollama/.git
cd joplin_weaviate_ollama
python -m venv .venv
source .venv/bin/activate   # Windows → .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 2 · System packages

- **Debian/Ubuntu**
    
    ```bash
    sudo apt update
    sudo apt install tesseract-ocr libtesseract-dev poppler-utils
    sudo apt install tesseract-ocr-eng tesseract-ocr-spa  # language packs
    ```
    
- **Arch**
    
    ```bash
    sudo pacman -S tesseract tesseract-data-eng tesseract-data-spa
    ```
    
- **macOS (Homebrew)**
    
    ```bash
    brew install tesseract poppler
    brew install tesseract-lang   # add packs as needed
    ```
    
- **Windows**
    
    1.  Install Tesseract and add to `PATH`.
        
    2.  WSL2 users: run Weaviate inside WSL or in a Docker Desktop Linux container.
        

### 3 · Optional GPU acceleration

Install CUDA 12 + cuDNN 8 or Apple Metal plugins → restart Ollama. `ollama pull llama3:8b` will auto‑detect GPU and quantise accordingly.

* * *

## Configuration

Copy `.env` and tweak:

```bash
cp sample.env .env
nano .env
```

| Var | Default | Description |
| --- | --- | --- |
| `MD_FOLDERS` | –   | Comma‑separated paths to Joplin exports (`.md` + `_resources`) |
| `WEAVIATE_URL` | `http://localhost:8080` | Use `https://` behind a reverse proxy |
| `WEAVIATE_INDEX` | `Documents` | Each export profile can map to a unique index |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | 384‑dim embeddings |
| `OLLAMA_MODEL` | `llama3:8b` | Any Ollama label works (`mistral:7b-instruct`, …) |
| `TELEGRAM_BOT_TOKEN` | *(unset)* | Enable bot when set |
| `TELEGRAM_USER_ID` | *(unset)* | Numeric ID accepted by the bot |
| `TZ` | `UTC` | Affects temporal boosts; set to your timezone |
| `LOG_LEVEL` | `INFO` | `DEBUG` prints request bodies & queries |

* * *

## Running Weaviate & Ollama

### 1 · Boot vector DB

```bash
docker compose up -d
# wait for readiness
until docker compose logs --tail 5 weaviate | grep -q "Startup complete"; do sleep 2; done
```

### 2 · Start local LLM

```bash
ollama serve &
ollama pull llama3:8b   # first time only
```

*Tip:* Put `ollama serve` in a systemd user service for autostart.

* * *

## Synchronising notes

### Common recipes

| Goal | Command |
| --- | --- |
| Smoke test (100 notes) | `python joplin_sync.py --sync --upload --test 100 --progress` |
| Full sync (all CPUs) | `python joplin_sync.py --sync --upload --workers $(nproc) --progress` |
| Incremental delta (CRON) | `./sync_and_upload.sh` |
| Re‑OCR only (skip upload) | `python joplin_sync.py --sync --workers 4 --no-upload` |
| Ad‑hoc hybrid search | `python joplin_sync.py --search "license plate" --alpha 0.7 --top-k 5` |

### Flag reference (excerpt)

| Flag | Meaning |
| --- | --- |
| `--workers N` | Threads for OCR/embeddings (default = CPU count) |
| `--batch-size N` | Upload batch size (notes) |
| `--index NAME` | Override target Weaviate class |
| `--timeout N` | Per‑image OCR timeout (sec) |
| `--alpha 0‑1` | Weight of vector vs BM25 for hybrid search |
| `--cache-info` | Print cache statistics |
| `--clean-cache [--force]` | Remove orphaned cache entries |

* * *

## Chatting with the CLIs

### Enhanced CLI (`rag_query.py`)

```bash
python rag_query.py
🧠 > ¿Tengo la factura de la lavadora?
```

| Shortcut | Effect |
| --- | --- |
| `debug:<query>` | Show top docs, classification, scores |
| `no-analysis:<query>` | Skip ownership / procedural heuristics |
| `summary` | 3‑line recap of conversation memory |
| `reset` | Clear memory buffer |
| `help` | Quick in‑CLI cheat sheet |

### Lightweight CLI (`rag_query-fast.py`)

Ideal for constrained hardware; same usage minus memory & reranking:

```bash
python rag_query-fast.py --config patterns.yaml
```

*Hot‑reload config:* `:reload` inside CLI re‑reads the YAML without restart.

* * *

## Telegram bot / Element Matrix.org bot

```bash
python telegram_rag_bot.py &
```

1.  Set `TELEGRAM_BOT_TOKEN` and `TELEGRAM_USER_ID` in `.env`.
    
2.  Send `/start` to your bot.
    
3.  Use Markdown or plain text; answers cite note paths like `notebooks/Finanzas/2023-IRPF.md`.
    

*Security note:* The bot rejects messages from any ID except the whitelisted one.

Note: ragbot_elementmatrix.py is untested, pull requests welcome



* * *

## Maintenance & troubleshooting

| Symptom | Likely cause | Remedy |
| --- | --- | --- |
| `Connection refused` to Weaviate | Container not ready | `docker compose logs -f weaviate` |
| "Phantom shard" error | Dirty shutdown | Add `RAFT_ENABLE_ONE_NODE_RECOVERY: true`, restart once |
| OCR stalls on TIFF | Bad scan | Lower `--timeout`; file skipped & logged |
| Answers too generic | Using *fast* CLI | Switch to full CLI or enlarge `top-k` |
| Model OOM on GPU | VRAM too small | Pull 4‑bit quant (`llama3:8b-q4_0`) or use CPU |

Logs:

- `sync_errors.log` – ingestion issues
    
- `~/.ollama/logs` – LLM server
    
- `weaviate-data/logs` – DB warnings
    

* * *

## Performance tuning

| Lever | Default | Faster | Notes |
| --- | --- | --- | --- |
| **Embeddings model** | MiniLM‑L6‑v2 | all‑mpnet‑base‑v2 (768 dim) | Higher recall, slower |
| **Batch size** | 32 notes | 128 | RAM bound |
| **OCR lang packs** | eng + spa | exact language subset | Fewer dictionaries = faster |
| **`--alpha`** | 0.7 | 0.5 (vector‑heavy) | Lower → BM25 heavier |
| **LLM quant** | q4_0 | q2_K | Lower VRAM, slower |
| **Weaviate cache** | off | `search.cache` plug‑in | Enterprise feature |

* * *

## Security & privacy

- **Network isolation** – Weaviate listens on `localhost` by default. Use firewall rules or Docker network to restrict.
    
- **Encrypted volume** – Store `weaviate-data/` on an encrypted partition if note content is sensitive.
    
- **No telemetry** – `ENABLE_TELEMETRY=false` in compose file.
    
- **Telegram bridge** – Remember all traffic goes through Telegram’s servers; dont run the telegram ragbot if strict privacy is required.
    

* * *

## FAQ

- **Why two CLIs?** — `rag_query.py` aims for maximal context awareness; `rag_query-fast.py` boots in seconds, fitting IoT / Pi devices.
    
- **Does any data leave my machine?** — No, unless *you* enable the Telegram bot or point `WEAVIATE_URL` to a remote host.
    
- **How do I wipe and rebuild the index?** — `docker compose down -v` to drop shards, delete `note_cache.json`, then run a fresh sync.
    
- **Can I disable OCR?** — Yes: `--no-ocr` skips image/PDF text extraction.
    

* * *

## License

MIT
