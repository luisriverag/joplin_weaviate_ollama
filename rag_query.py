#!/usr/bin/env python3
from __future__ import annotations

import os
from urllib.parse import urlparse

from dotenv import load_dotenv
import weaviate
from weaviate.classes.init import AdditionalConfig, Timeout

from langchain_weaviate import WeaviateVectorStore
from langchain_ollama import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA


# ── env vars ─────────────────────────────────────────────────────────────
load_dotenv()

WEAVIATE_URL   = os.getenv("WEAVIATE_URL",   "http://localhost:8080")
WEAVIATE_INDEX = os.getenv("WEAVIATE_INDEX", "Note")
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)
OLLAMA_MODEL   = os.getenv("OLLAMA_MODEL",   "llama3:8b")


# ── Weaviate v4 client ───────────────────────────────────────────────────
def make_weaviate_client(url: str) -> weaviate.WeaviateClient:
    """
    Parse WEAVIATE_URL and connect with the v4 helpers.

    Works for both localhost (`connect_to_local`) and remote hosts
    (`connect_to_custom`).
    """
    parsed = urlparse(url)

    host = parsed.hostname or "localhost"
    port = parsed.port or (443 if parsed.scheme == "https" else 8080)
    secure = parsed.scheme == "https"

    if host in {"localhost", "127.0.0.1"}:
        # simple helper – returns a v4 WeaviateClient instance
        return weaviate.connect_to_local(
            host=host,
            port=port,
            grpc_port=50051,
            additional_config=AdditionalConfig(
                timeout=Timeout(init=5)          # 5-second init check
            ),
        )                                       # :contentReference[oaicite:1]{index=1}
    else:
        # full-spec helper for remote instances
        return weaviate.connect_to_custom(
            http_host=host,
            http_port=port,
            http_secure=secure,
            grpc_host=host,
            grpc_port=50051,
            grpc_secure=secure,
            additional_config=AdditionalConfig(timeout=Timeout(init=5)),
        )


client = make_weaviate_client(WEAVIATE_URL)
assert client.is_ready(), "Weaviate instance is not ready"

# ── LangChain components ─────────────────────────────────────────────────
embedder = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

vectorstore = WeaviateVectorStore(
    client=client,                # v4 `WeaviateClient` instance
    index_name=WEAVIATE_INDEX,
    text_key="text",
    embedding=embedder,
)

llm = OllamaLLM(model=OLLAMA_MODEL)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(),
)

# ── interactive loop ─────────────────────────────────────────────────────
def main() -> None:
    print("✨ Connected. Ask a question or type 'exit'.")
    while True:
        try:
            query = input("🧠 > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if query.lower() in {"exit", "quit", "q"}:
            break
        if not query:
            continue

        try:
            answer = qa_chain.invoke(query)    # {"query": ..., "result": ...}
            print("📝", answer["result"].strip())
        except Exception as exc:
            print("⚠️  Error:", exc)


if __name__ == "__main__":
    main()
