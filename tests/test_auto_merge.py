"""Tests for auto-merge manager: conditions, countdown, cancellation."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest

from sgldhelper.ci.auto_merge import AutoMergeManager
from sgldhelper.db import queries


@pytest.fixture
def mock_gh():
    gh = AsyncMock()
    gh.get_pull = AsyncMock(return_value={
        "number": 19876,
        "state": "open",
        "mergeable": True,
        "head": {"sha": "abc123"},
        "user": {"login": "alice"},
        "title": "Test PR",
        "labels": [],
        "updated_at": "2025-03-01T00:00:00Z",
    })
    gh.merge_pull = AsyncMock(return_value={"merged": True})
    return gh


@pytest.fixture
def auto_merge(mock_gh, db, settings):
    # Use very short delay for testing
    settings.auto_merge_delay_seconds = 1
    return AutoMergeManager(mock_gh, db, settings)


class TestConditionCheck:
    @pytest.mark.asyncio
    async def test_skip_when_disabled(self, auto_merge, settings):
        settings.auto_merge_enabled = False
        result = await auto_merge.check_and_start(19876, ["U1"], "approved")
        assert result is False

    @pytest.mark.asyncio
    async def test_skip_no_approval(self, auto_merge):
        result = await auto_merge.check_and_start(19876, ["U1"], "none")
        assert result is False

    @pytest.mark.asyncio
    async def test_skip_changes_requested(self, auto_merge):
        result = await auto_merge.check_and_start(19876, ["U1"], "changes_requested")
        assert result is False

    @pytest.mark.asyncio
    async def test_skip_not_mergeable(self, auto_merge, mock_gh):
        mock_gh.get_pull = AsyncMock(return_value={
            "number": 19876, "state": "open", "mergeable": False,
            "head": {"sha": "abc123"}, "user": {"login": "alice"},
            "title": "Test", "labels": [], "updated_at": "2025-03-01",
        })
        result = await auto_merge.check_and_start(19876, ["U1"], "approved")
        assert result is False

    @pytest.mark.asyncio
    async def test_skip_not_in_db(self, auto_merge):
        """PR must be in our DB (a known diffusion PR)."""
        result = await auto_merge.check_and_start(19876, ["U1"], "approved")
        assert result is False

    @pytest.mark.asyncio
    async def test_start_when_eligible(self, auto_merge, db):
        """Should start countdown when all conditions met."""
        await queries.upsert_pr(db.conn, {
            "pr_number": 19876, "title": "Test PR", "author": "alice",
            "state": "open", "head_sha": "abc123", "updated_at": "2025-03-01",
            "changed_files": 5,
        })
        result = await auto_merge.check_and_start(19876, ["U1"], "approved")
        assert result is True
        assert auto_merge.is_pending(19876)
        # Clean up
        await auto_merge.cancel(19876)

    @pytest.mark.asyncio
    async def test_no_duplicate_pending(self, auto_merge, db):
        """Should not start a second countdown for the same PR."""
        await queries.upsert_pr(db.conn, {
            "pr_number": 19876, "title": "Test PR", "author": "alice",
            "state": "open", "head_sha": "abc123", "updated_at": "2025-03-01",
            "changed_files": 5,
        })
        await auto_merge.check_and_start(19876, ["U1"], "approved")
        result = await auto_merge.check_and_start(19876, ["U1"], "approved")
        assert result is False
        await auto_merge.cancel(19876)


class TestCancellation:
    @pytest.mark.asyncio
    async def test_cancel_pending(self, auto_merge, db):
        await queries.upsert_pr(db.conn, {
            "pr_number": 19876, "title": "Test PR", "author": "alice",
            "state": "open", "head_sha": "abc123", "updated_at": "2025-03-01",
            "changed_files": 5,
        })
        await auto_merge.check_and_start(19876, ["U1"], "approved")
        assert await auto_merge.cancel(19876) is True
        assert not auto_merge.is_pending(19876)

    @pytest.mark.asyncio
    async def test_cancel_nonexistent(self, auto_merge):
        assert await auto_merge.cancel(99999) is False


class TestCancelKeywords:
    def test_detects_cancel_keyword(self, auto_merge, settings):
        """Should detect cancel keywords in text."""
        # No pending merges, so returns None
        result = auto_merge.check_cancel_keywords("取消merge")
        assert result is None

    @pytest.mark.asyncio
    async def test_cancel_keyword_with_pending(self, auto_merge, db, settings):
        """Should return PR number when cancel keyword matches a pending merge."""
        await queries.upsert_pr(db.conn, {
            "pr_number": 19876, "title": "Test PR", "author": "alice",
            "state": "open", "head_sha": "abc123", "updated_at": "2025-03-01",
            "changed_files": 5,
        })
        await auto_merge.check_and_start(19876, ["U1"], "approved")
        result = auto_merge.check_cancel_keywords("取消merge")
        assert result == 19876
        await auto_merge.cancel(19876)

    @pytest.mark.asyncio
    async def test_cancel_keyword_with_pr_number(self, auto_merge, db, settings):
        """Should match specific PR number in text."""
        await queries.upsert_pr(db.conn, {
            "pr_number": 19876, "title": "Test PR", "author": "alice",
            "state": "open", "head_sha": "abc123", "updated_at": "2025-03-01",
            "changed_files": 5,
        })
        await auto_merge.check_and_start(19876, ["U1"], "approved")
        result = auto_merge.check_cancel_keywords("cancel merge #19876")
        assert result == 19876
        await auto_merge.cancel(19876)

    def test_no_cancel_keyword(self, auto_merge):
        result = auto_merge.check_cancel_keywords("looks good, let's merge")
        assert result is None


class TestMergeExecution:
    @pytest.mark.asyncio
    async def test_merge_after_countdown(self, auto_merge, mock_gh, db):
        """Should merge after countdown completes."""
        await queries.upsert_pr(db.conn, {
            "pr_number": 19876, "title": "Test PR", "author": "alice",
            "state": "open", "head_sha": "abc123", "updated_at": "2025-03-01",
            "changed_files": 5,
        })
        on_complete = AsyncMock()
        on_countdown = AsyncMock()
        auto_merge.set_callbacks(on_countdown=on_countdown, on_complete=on_complete)

        await auto_merge.check_and_start(19876, ["U1"], "approved")
        # Wait for the short countdown to finish
        await asyncio.sleep(2)

        mock_gh.merge_pull.assert_called_once_with(19876, merge_method="squash")
        on_complete.assert_called_once()
        assert not auto_merge.is_pending(19876)
