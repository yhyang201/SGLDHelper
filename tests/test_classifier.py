"""Tests for AI message classifier."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from sgldhelper.ai.classifier import MessageClassifier
from sgldhelper.ai.client import KimiClient


@pytest.fixture
def mock_kimi():
    client = AsyncMock(spec=KimiClient)
    return client


@pytest.fixture
async def classifier(mock_kimi, db, settings):
    return MessageClassifier(mock_kimi, db, settings)


class TestHeuristicFilter:
    def test_relevant_messages_pass_filter(self, classifier):
        assert classifier._might_be_relevant("I finished the ControlNet PR") is True
        assert classifier._might_be_relevant("PR #1234 is blocked on review") is True
        assert classifier._might_be_relevant("Working on SDXL integration") is True
        assert classifier._might_be_relevant("搞定了 PR 的修改") is True
        assert classifier._might_be_relevant("进度更新：已完成 50%") is True

    def test_irrelevant_messages_filtered(self, classifier):
        assert classifier._might_be_relevant("Hello everyone") is False
        assert classifier._might_be_relevant("lunch?") is False
        assert classifier._might_be_relevant("") is False


class TestClassifyMessage:
    @pytest.mark.asyncio
    async def test_progress_update_detected(self, classifier, mock_kimi, db):
        mock_kimi.classify = AsyncMock(return_value={
            "category": "progress_update",
            "summary": "Finished ControlNet integration",
            "mentioned_pr": 1234,
            "mentioned_feature": None,
        })

        result = await classifier.classify_message(
            text="I finished the ControlNet integration, PR #1234 is ready for review",
            user_id="U123",
            channel_id="C_TEST_PR",
            message_ts="1234567890.123456",
        )

        assert result is not None
        assert result["category"] == "progress_update"
        assert result["summary"] == "Finished ControlNet integration"
        assert result["mentioned_pr"] == 1234
        mock_kimi.classify.assert_called_once()

    @pytest.mark.asyncio
    async def test_blocker_detected(self, classifier, mock_kimi, db):
        mock_kimi.classify = AsyncMock(return_value={
            "category": "blocker",
            "summary": "GPU OOM on large batch sizes",
            "mentioned_pr": None,
            "mentioned_feature": "SDXL support",
        })

        result = await classifier.classify_message(
            text="I'm blocked on SDXL support - getting GPU OOM on large batch sizes",
            user_id="U456",
            channel_id="C_TEST_CI",
            message_ts="1234567890.789012",
        )

        assert result is not None
        assert result["category"] == "blocker"
        assert result["mentioned_feature"] == "SDXL support"

    @pytest.mark.asyncio
    async def test_general_message_returns_none(self, classifier, mock_kimi):
        mock_kimi.classify = AsyncMock(return_value={
            "category": "general",
            "summary": "Just a question",
            "mentioned_pr": None,
            "mentioned_feature": None,
        })

        result = await classifier.classify_message(
            text="Has anyone seen the new update from PyTorch?",
            user_id="U789",
            channel_id="C_TEST_PR",
            message_ts="1234567890.000000",
        )

        # general messages should be filtered as None (but only if heuristic passes)
        # This message won't pass the heuristic filter
        assert result is None

    @pytest.mark.asyncio
    async def test_short_message_skipped(self, classifier, mock_kimi):
        result = await classifier.classify_message(
            text="ok",
            user_id="U123",
            channel_id="C_TEST_PR",
            message_ts="1234567890.000000",
        )
        assert result is None
        mock_kimi.classify.assert_not_called()



class TestDetectedUpdatesDB:
    @pytest.mark.asyncio
    async def test_update_saved_to_db(self, classifier, mock_kimi, db):
        from sgldhelper.db import queries

        mock_kimi.classify = AsyncMock(return_value={
            "category": "progress_update",
            "summary": "Test update",
            "mentioned_pr": None,
            "mentioned_feature": None,
        })

        result = await classifier.classify_message(
            text="I completed the progress update for this feature",
            user_id="U123",
            channel_id="C_TEST_FEAT",
            message_ts="1234567890.333333",
        )

        assert result is not None
        pending = await queries.get_pending_updates(db.conn, "progress_update")
        assert len(pending) == 1
        assert pending[0]["user_id"] == "U123"
        assert pending[0]["confirmed"] == 0

    @pytest.mark.asyncio
    async def test_confirm_update(self, classifier, mock_kimi, db):
        from sgldhelper.db import queries

        mock_kimi.classify = AsyncMock(return_value={
            "category": "blocker",
            "summary": "Test blocker",
            "mentioned_pr": None,
            "mentioned_feature": None,
        })

        result = await classifier.classify_message(
            text="I'm blocked on the PR review process, need help",
            user_id="U123",
            channel_id="C_TEST_PR",
            message_ts="1234567890.444444",
        )

        assert result is not None
        update_id = result["update_id"]

        await queries.confirm_detected_update(db.conn, update_id)
        blockers = await queries.get_confirmed_blockers(db.conn, 24)
        assert len(blockers) == 1
