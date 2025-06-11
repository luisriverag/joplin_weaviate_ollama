# JOPLIN PRIVATE AI
## RAG Your Notes Locally Using Open Source Models

*(Joplin Notes to Weaviate Ollama RAG pipeline with optional Telegram client)*

---


This program syncs your Joplin notes and resources into Weaviate,
embeds them with SentenceTransformers, and lets you query them via
LangChain + Ollama

It has 2 interfaces
- Console  (which you can SSH into, even if you need a VPN or a reverse tunnel on 
a relay server)
- Telegram bot (which kind of defeats the local only approach given Telegram
will receive a copy of all your conversations, but is useful for testing until 
we replace it with something better)

We recommend using it against an MD export of your notes, not the folder where
you store originals.

Features
--------
- ✅ Sync only changed files (Markdown, PDF, images with OCR)
- ✅ Embedding via HuggingFace sentence-transformers
- ✅ Vector storage in Weaviate
- ✅ RAG pipeline using LangChain + Ollama
- ✅ Telegram bot for private mobile access
- ✅ Configurable via `.env`

Setup
-----
1. Clone this repo and install Python packages:
   pip install -r requirements.txt

2. Install system dependencies:
   - Tesseract OCR:
     - Ubuntu:   sudo apt install tesseract-ocr
     - macOS:    brew install tesseract
     - Windows:  https://github.com/tesseract-ocr/tesseract/wiki
   - Ollama:    https://ollama.com/download

3. Start Weaviate locally:
   docker-compose up -d

4. Create `.env`:
   cp sample.env .env

5. Edit .env with your folder locations...

Usage
-----
1. Sync & Upload Notes:
   ./sync_and_upload.sh

2. Ask Questions Locally (Ollama RAG):
   python3 rag_query.py
   > What is my motorbike's license plate?

3. Telegram Bot:
   python3 telegram_rag_bot.py
   # On Telegram
   > /start
   > What is my motorbikes's license plate?

Bot is restricted to AUTHORIZED_USER_ID from `.env`.

Files
-----
- joplin_sync.py        — Sync + upload tool
- rag_query.py          — Local CLI RAG chat with Ollama
- telegram_rag_bot.py   — Telegram chat bot
- sync_and_upload.sh    — Helper script to run sync+upload
- docker-compose.yml    — Weaviate docker
- requirements.txt      — Python dependencies
- .env                  — Config for folders, models, and bot
