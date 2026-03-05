"""Passive message classification using K2.5 instant mode."""

from __future__ import annotations

import json
import re
from typing import Any

import structlog

from sgldhelper.ai.client import KimiClient
from sgldhelper.config import Settings
from sgldhelper.db import queries
from sgldhelper.db.engine import Database

log = structlog.get_logger()


class MessageClassifier:
    """Classify channel messages as progress updates, blockers, questions, or general.

    Uses K2.5 instant mode for cheap/fast classification, then matches
    detected updates to feature items in the DB.
    """

    def __init__(
        self,
        client: KimiClient,
        db: Database,
        settings: Settings,
    ) -> None:
        self._client = client
        self._db = db
        self._settings = settings

    async def classify_message(
        self,
        text: str,
        user_id: str,
        channel_id: str,
        message_ts: str,
    ) -> dict[str, Any] | None:
        """Classify a channel message.

        Returns classification result dict if actionable (progress_update or blocker),
        None for general/question messages.
        """
        # Skip very short messages or bot messages
        if len(text.strip()) < 10:
            return None

        # Quick heuristic pre-filter: skip obvious non-updates
        if not self._might_be_relevant(text):
            return None

        result = await self._client.classify(text)
        category = result.get("category", "general")

        tokens_in, tokens_out = 0, 0  # Usage tracked inside client.classify via chat()
        await queries.log_llm_usage(
            self._db.conn, "classification", self._settings.kimi_model,
            tokens_in, tokens_out,
        )

        log.info(
            "classifier.result",
            category=category,
            summary=result.get("summary", "")[:80],
            channel=channel_id,
        )

        if category not in ("progress_update", "blocker"):
            return None

        # Save to DB
        update_id = await queries.save_detected_update(
            self._db.conn,
            channel_id=channel_id,
            message_ts=message_ts,
            user_id=user_id,
            classification=category,
            extracted_data=json.dumps(result),
        )

        return {
            "update_id": update_id,
            "category": category,
            "summary": result.get("summary", ""),
            "mentioned_pr": result.get("mentioned_pr"),
            "mentioned_feature": result.get("mentioned_feature"),
        }

    def _might_be_relevant(self, text: str) -> bool:
        """Quick heuristic check before calling LLM."""
        text_lower = text.lower()
        keywords = [
            "done", "finished", "completed", "merged", "landed",
            "blocked", "blocker", "stuck", "help",
            "progress", "update", "working on", "started",
            "pr", "pull request", "#",
            "完成", "进度", "阻塞", "卡住", "搞定",
        ]
        return any(kw in text_lower for kw in keywords)
