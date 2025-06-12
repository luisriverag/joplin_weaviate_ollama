#!/usr/bin/env python3
import os
import logging
from dotenv import load_dotenv

from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

from rag_query import GenericRAG  # ⬅️ imports the improved RAG system

# ── Configuration ──────────────────────────────────────────────
load_dotenv()

BOT_TOKEN: str | None = os.getenv("TELEGRAM_BOT_TOKEN")
ALLOWED_USERID: int = int(os.getenv("TELEGRAM_USER_ID", "0"))
CLASSIFIER_CONFIG: str | None = os.getenv("CLASSIFIER_CONFIG")  # JSON file with custom document patterns

if not BOT_TOKEN:
    raise RuntimeError("TELEGRAM_BOT_TOKEN is not set in the environment")

# Logging setup
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("TelegramRAGBot")

# ── RAG System ────────────────────────────────────────────────
rag = GenericRAG(config_path=CLASSIFIER_CONFIG)

# ── Bot handlers ──────────────────────────────────────────────
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/start command handler"""
    if update.effective_user.id != ALLOWED_USERID:
        return  # Ignore strangers

    await update.message.reply_text(
        "🧠 Hi! I'm your personal knowledge‑base assistant.\n"
        "Send me any question.\n\n"
        "• Prefix with *no-analysis:* to skip document analysis.\n"
        "• Prefix with *debug:* to see a detailed breakdown of the top retrieved files.\n",
        parse_mode="Markdown",
    )


async def handle_query(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Main message handler"""
    user_id = update.effective_user.id
    message  = update.message.text or ""
    query = message.strip()

    logger.info("📥 Message from %s: %s", user_id, query)

    if user_id != ALLOWED_USERID:
        logger.warning("❌ Unauthorized user %s – ignoring", user_id)
        return

    if not query:
        return  # Empty message

    try:
        # Command-style prefixes -------------------------------------------------
        lowered = query.lower()

        if lowered.startswith("no-analysis:"):
            actual_query = query[12:].strip()
            answer = rag.query_with_analysis(actual_query, show_analysis=False)
            await update.message.reply_text(f"📝 {answer}")
            return

        if lowered.startswith("debug:"):
            actual_query = query[6:].strip()
            analysis = rag.analyze_retrieved_docs(actual_query)

            lines: list[str] = [
                f"🔍 Debug Analysis for '{actual_query}':",
                f"• Total documents retrieved: {analysis['total_docs']}",
            ]

            for i, cls in enumerate(analysis["classifications"][:5], 1):
                lines.append(f"{i}. {cls['title'] or 'Untitled'}")
                lines.append(f"   Type: {cls['classification']['type']}")
                lines.append(f"   Confidence: {cls['classification']['confidence']:.2f}")
                indicators = ", ".join(cls['classification']['indicators']) or "general match"
                lines.append(f"   Indicators: {indicators}")

            await update.message.reply_text("\n".join(lines))
            return

        # Default behaviour ------------------------------------------------------
        answer = rag.query_with_analysis(query)
        await update.message.reply_text(f"📝 {answer}")

    except Exception as exc:  # pylint: disable=broad-except
        logger.exception("⚠️ Error while processing query: %s", exc)
        await update.message.reply_text("⚠️ Sorry, something went wrong while processing your question.")


# ── Bot application setup ─────────────────────────────────────

def main() -> None:
    """Starts the Telegram bot and enters polling loop"""
    application = ApplicationBuilder().token(BOT_TOKEN).build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_query))

    logger.info("🤖 Bot is up and running – press Ctrl+C to stop")
    application.run_polling()


if __name__ == "__main__":
    main()
