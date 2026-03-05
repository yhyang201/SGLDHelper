"""Slack Bolt async app with Socket Mode."""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

import structlog
from slack_bolt.adapter.socket_mode.async_handler import AsyncSocketModeHandler
from slack_bolt.async_app import AsyncApp

from sgldhelper.config import Settings

if TYPE_CHECKING:
    import aiosqlite

log = structlog.get_logger()


class SlackApp:
    """Wrapper around slack_bolt AsyncApp with Socket Mode support."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self.app = AsyncApp(token=settings.slack_bot_token)
        self._handler: AsyncSocketModeHandler | None = None
        self.bot_user_id: str | None = None

        # Debug middleware: log every incoming event
        @self.app.middleware
        async def log_all_events(body, next):
            event = body.get("event", {})
            event_type = event.get("type", body.get("type", "unknown"))
            subtype = event.get("subtype", "")
            channel = event.get("channel", "")
            user = event.get("user", "")
            text_preview = (event.get("text", "") or "")[:80]
            log.debug(
                "slack.event_received",
                event_type=event_type,
                subtype=subtype,
                channel=channel,
                user=user,
                text_preview=text_preview,
            )
            await next()

    async def start(self) -> None:
        """Start the Socket Mode handler (non-blocking)."""
        self._handler = AsyncSocketModeHandler(
            self.app, self._settings.slack_app_token
        )
        await self._handler.connect_async()
        # Fetch bot's own user ID so we can filter out self-messages
        try:
            auth = await self.app.client.auth_test()
            self.bot_user_id = auth.get("user_id")
            log.info("slack.connected", bot_user_id=self.bot_user_id)
        except Exception:
            log.info("slack.connected")

    async def stop(self) -> None:
        """Disconnect the Socket Mode handler."""
        if self._handler:
            await self._handler.close_async()
            log.info("slack.disconnected")

    async def post_message(
        self,
        channel: str,
        *,
        text: str,
        blocks: list[dict[str, Any]] | None = None,
        thread_ts: str | None = None,
    ) -> dict[str, Any]:
        """Post a message to a Slack channel."""
        result = await self.app.client.chat_postMessage(
            channel=channel,
            text=text,
            blocks=blocks,
            thread_ts=thread_ts,
        )
        return result.data  # type: ignore[return-value]

    async def post_message_with_context(
        self,
        channel: str,
        *,
        text: str,
        blocks: list[dict[str, Any]] | None = None,
        thread_ts: str | None = None,
        db_conn: "aiosqlite.Connection | None" = None,
    ) -> dict[str, Any]:
        """Post a message and save it to conversation history.

        When the bot posts proactively (health checks, CI notifications),
        the message needs to be stored so that user replies in the thread
        have context about what the bot said.
        """
        result = await self.post_message(
            channel, text=text, blocks=blocks, thread_ts=thread_ts,
        )
        if db_conn is not None:
            ts = result.get("ts", "")
            # Use the thread_ts if replying, otherwise the new message ts
            ctx_thread_ts = thread_ts or ts
            from sgldhelper.db import queries
            await queries.save_conversation_message(
                db_conn, ctx_thread_ts, channel,
                role="assistant", content=text,
            )
        return result

    async def update_message(
        self,
        channel: str,
        ts: str,
        *,
        text: str,
        blocks: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Update an existing Slack message."""
        result = await self.app.client.chat_update(
            channel=channel,
            ts=ts,
            text=text,
            blocks=blocks,
        )
        return result.data  # type: ignore[return-value]
