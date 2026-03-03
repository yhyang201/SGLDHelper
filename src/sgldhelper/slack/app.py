"""Slack Bolt async app with Socket Mode."""

from __future__ import annotations

from typing import Any

import structlog
from slack_bolt.adapter.socket_mode.async_handler import AsyncSocketModeHandler
from slack_bolt.async_app import AsyncApp

from sgldhelper.config import Settings

log = structlog.get_logger()


class SlackApp:
    """Wrapper around slack_bolt AsyncApp with Socket Mode support."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self.app = AsyncApp(token=settings.slack_bot_token)
        self._handler: AsyncSocketModeHandler | None = None

    async def start(self) -> None:
        """Start the Socket Mode handler (non-blocking)."""
        self._handler = AsyncSocketModeHandler(
            self.app, self._settings.slack_app_token
        )
        await self._handler.connect_async()
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
