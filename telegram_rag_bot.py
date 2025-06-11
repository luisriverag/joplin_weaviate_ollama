#!/usr/bin/env python3

import os
from urllib.parse import urlparse
from dotenv import load_dotenv
import logging

import weaviate
from weaviate.classes.init import AdditionalConfig, Timeout

from langchain_weaviate import WeaviateVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA

from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes

# ── Config ─────────────────────────────────────────────────────
load_dotenv()

BOT_TOKEN      = os.getenv("TELEGRAM_BOT_TOKEN")
ALLOWED_USERID = int(os.getenv("TELEGRAM_USER_ID", "0"))

WEAVIATE_URL   = os.getenv("WEAVIATE_URL", "http://localhost:8080")
WEAVIATE_INDEX = os.getenv("WEAVIATE_INDEX", "Note")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
OLLAMA_MODEL    = os.getenv("OLLAMA_MODEL", "llama3:8b")


# ── Weaviate client factory ────────────────────────────────────
def make_weaviate_client(url: str) -> weaviate.WeaviateClient:
    parsed = urlparse(url)
    host = parsed.hostname or "localhost"
    port = parsed.port or (443 if parsed.scheme == "https" else 8080)
    secure = parsed.scheme == "https"

    if host in {"localhost", "127.0.0.1"}:
        return weaviate.connect_to_local(
            host=host,
            port=port,
            grpc_port=50051,
            additional_config=AdditionalConfig(timeout=Timeout(init=5)),
        )
    else:
        return weaviate.connect_to_custom(
            http_host=host,
            http_port=port,
            http_secure=secure,
            grpc_host=host,
            grpc_port=50051,
            grpc_secure=secure,
            additional_config=AdditionalConfig(timeout=Timeout(init=5)),
        )


# ── LangChain setup ────────────────────────────────────────────
client = make_weaviate_client(WEAVIATE_URL)
assert client.is_ready(), "❌ Weaviate connection failed"

embedder = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
vectorstore = WeaviateVectorStore(
    client=client,
    index_name=WEAVIATE_INDEX,
    text_key="text",
    embedding=embedder,
)
llm = OllamaLLM(model=OLLAMA_MODEL)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(),
)


# ── Bot handlers ───────────────────────────────────────────────
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id != ALLOWED_USERID:
        return
    await update.message.reply_text("🧠 Hello! Ask me anything from your notes.")


async def handle_query(update: Update, context: ContextTypes.DEFAULT_TYPE):
    print(f"📥 Received message from {update.effective_user.id}: {update.message.text}")
    
    if update.effective_user.id != ALLOWED_USERID:
        print("❌ Unauthorized user. Ignoring.")
        return

    query = update.message.text.strip()
    if not query:
        return

    try:
        answer = qa_chain.invoke(query)
        await update.message.reply_text(f"📝 {answer['result'].strip()}")
    except Exception as e:
        print("⚠️ Error during query:", e)
        await update.message.reply_text("⚠️ Sorry, I couldn't process that.")



# ── App init ───────────────────────────────────────────────────
def main():
    application = ApplicationBuilder().token(BOT_TOKEN).build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_query))

    print("🤖 Bot running...")
    application.run_polling()


if __name__ == "__main__":
    main()
