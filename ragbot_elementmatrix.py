#!/usr/bin/env python3
import asyncio
import logging
import os
from dotenv import load_dotenv

from nio import (
    AsyncClient,
    MatrixRoom,
    RoomMessageText,
    InviteEvent,
)

from rag_query import GenericRAG

# ── Configuration ─────────────────────────────────────────────

load_dotenv()

HOMESERVER = os.getenv("MATRIX_HOMESERVER")
ACCESS_TOKEN = os.getenv("MATRIX_ACCESS_TOKEN")
BOT_USER_ID = os.getenv("MATRIX_USER_ID")
ALLOWED_USER_ID = os.getenv("ALLOWED_USER_ID")

CLASSIFIER_CONFIG = os.getenv("CLASSIFIER_CONFIG")

if not all((HOMESERVER, ACCESS_TOKEN, BOT_USER_ID, ALLOWED_USER_ID)):
    raise RuntimeError(
        "Missing env vars. Please set MATRIX_HOMESERVER, MATRIX_ACCESS_TOKEN, "
        "MATRIX_USER_ID and ALLOWED_USER_ID in your environment."
    )

# Logging
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("ElementRAGBot")

# ── RAG system ────────────────────────────────────────────────

rag = GenericRAG(config_path=CLASSIFIER_CONFIG)

# ── Bot logic ─────────────────────────────────────────────────


class RAGBot:
    """Matrix RAG bot wrapper around AsyncClient."""

    def __init__(self, homeserver: str, access_token: str, user_id: str):
        self.client = AsyncClient(homeserver, user_id)
        self.client.access_token = access_token
        self.client.user_id = user_id

        # Register callbacks
        self.client.add_event_callback(self.on_invite, InviteEvent)
        self.client.add_event_callback(self.on_message, RoomMessageText)

    # ---------------- event handlers ---------------- #

    async def on_invite(self, room: MatrixRoom, event: InviteEvent) -> None:
        """Auto‑join rooms when invited."""
        logger.info("Invited to room %s – joining", room.room_id)
        await self.client.join(room.room_id)

    async def on_message(self, room: MatrixRoom, event: RoomMessageText) -> None:
        """Handle incoming text messages."""
        # Ignore own messages
        if event.sender == self.client.user_id:
            return

        # Only accept DMs from allowed user
        if event.sender != ALLOWED_USER_ID or room.member_count > 2:
            logger.warning("Ignoring message from unauthorised user %s", event.sender)
            return

        # Ensure message body exists
        if not event.body:
            return

        query = event.body.strip()
        logger.info("📥 Received: %s", query)

        try:
            lowered = query.lower()

            if lowered == "!start":
                await self.reply(
                    room,
                    "🧠 Hi! I'm your personal knowledge‑base assistant.\n"
                    "Send me any question.\n\n"
                    "• Prefix with *no-analysis:* to skip document analysis.\n"
                    "• Prefix with *debug:* to see a detailed breakdown of the top retrieved files.\n",
                    markdown=True,
                )
                return

            if lowered.startswith("no-analysis:"):
                actual_query = query[12:].strip()
                answer = rag.query_with_analysis(actual_query, show_analysis=False)
                await self.reply(room, f"📝 {answer}")
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

                await self.reply(room, "\n".join(lines))
                return

            # Default behaviour --------------------------------------------------
            answer = rag.query_with_analysis(query)
            await self.reply(room, f"📝 {answer}")

        except Exception as exc:  # pylint: disable=broad-except
            logger.exception("⚠️ Error while processing query: %s", exc)
            await self.reply(room, "⚠️ Sorry, something went wrong while processing your question.")

    # ---------------- helpers ---------------- #

    async def reply(self, room: MatrixRoom, message: str, markdown: bool = False) -> None:
        """Send a message to the room."""
        await self.client.room_send(
            room_id=room.room_id,
            message_type="m.room.message",
            content={
                "msgtype": "m.text",
                "format": "org.matrix.custom.html" if markdown else None,
                "formatted_body": message if markdown else None,
                "body": message,
            },
        )

    # ---------------- main loop ---------------- #

    async def run(self) -> None:
        """Run sync loop forever."""
        logger.info("🤖 Element RAG bot logged in as %s", self.client.user_id)
        try:
            await self.client.sync_forever(timeout=30_000)  # millisecond timeout
        except Exception:  # pylint: disable=broad-except
            logger.exception("Fatal error – exiting")
        finally:
            await self.client.close()


# ── Entrypoint ────────────────────────────────────────────────

def main() -> None:
    bot = RAGBot(HOMESERVER, ACCESS_TOKEN, BOT_USER_ID)
    asyncio.run(bot.run())


if __name__ == "__main__":
    main()
