"""Tests for stall detection and review nudge logic."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock

import pytest

from sgldhelper.ai.stall_detector import StallDetector
from sgldhelper.db import queries


@pytest.fixture
def mock_gh():
    gh = AsyncMock()
    gh.get_pull_reviews = AsyncMock(return_value=[])
    return gh


@pytest.fixture
async def detector(db, mock_gh, settings):
    return StallDetector(db, mock_gh, settings)


def _days_ago(n: int) -> str:
    return (datetime.now(timezone.utc) - timedelta(days=n)).strftime("%Y-%m-%dT%H:%M:%SZ")


class TestStalledFeatures:
    @pytest.mark.asyncio
    async def test_detects_stalled_feature(self, detector, db, settings):
        # Insert a stale PR
        await queries.upsert_pr(db.conn, {
            "pr_number": 100, "title": "Stale PR", "author": "alice",
            "state": "open", "head_sha": "aaa111",
            "updated_at": _days_ago(5), "changed_files": 3,
        })
        # Insert an open feature item linked to that PR
        await queries.upsert_feature_item(db.conn, {
            "item_id": "stale_item_1",
            "parent_issue": 14199,
            "title": "Add feature X",
            "state": "open",
            "linked_pr": 100,
        })

        alerts = await detector.check_stalled_features()
        assert len(alerts) == 1
        assert alerts[0].alert_type == "feature_stall"
        assert alerts[0].pr_number == 100
        assert "hasn't been updated" in alerts[0].details

    @pytest.mark.asyncio
    async def test_ignores_recently_updated(self, detector, db, settings):
        await queries.upsert_pr(db.conn, {
            "pr_number": 200, "title": "Fresh PR", "author": "bob",
            "state": "open", "head_sha": "bbb222",
            "updated_at": _days_ago(0), "changed_files": 2,
        })
        await queries.upsert_feature_item(db.conn, {
            "item_id": "fresh_item_1",
            "parent_issue": 14199,
            "title": "Add feature Y",
            "state": "open",
            "linked_pr": 200,
        })

        alerts = await detector.check_stalled_features()
        assert len(alerts) == 0

    @pytest.mark.asyncio
    async def test_ignores_completed_features(self, detector, db, settings):
        await queries.upsert_pr(db.conn, {
            "pr_number": 300, "title": "Old PR", "author": "carol",
            "state": "open", "head_sha": "ccc333",
            "updated_at": _days_ago(10), "changed_files": 1,
        })
        await queries.upsert_feature_item(db.conn, {
            "item_id": "done_item_1",
            "parent_issue": 14199,
            "title": "Already done",
            "state": "completed",
            "linked_pr": 300,
        })

        alerts = await detector.check_stalled_features()
        assert len(alerts) == 0

    @pytest.mark.asyncio
    async def test_dedup_stall_alerts(self, detector, db, settings):
        await queries.upsert_pr(db.conn, {
            "pr_number": 400, "title": "Dup PR", "author": "dave",
            "state": "open", "head_sha": "ddd444",
            "updated_at": _days_ago(5), "changed_files": 4,
        })
        await queries.upsert_feature_item(db.conn, {
            "item_id": "dup_item_1",
            "parent_issue": 14199,
            "title": "Dup feature",
            "state": "open",
            "linked_pr": 400,
        })

        # First check should produce an alert
        alerts1 = await detector.check_stalled_features()
        assert len(alerts1) == 1

        # Second check should be deduplicated
        alerts2 = await detector.check_stalled_features()
        assert len(alerts2) == 0


class TestReviewNudge:
    @pytest.mark.asyncio
    async def test_nudges_pr_without_approval(self, detector, db, mock_gh, settings):
        await queries.upsert_pr(db.conn, {
            "pr_number": 500, "title": "Needs review", "author": "eve",
            "state": "open", "head_sha": "eee555",
            "updated_at": _days_ago(4), "changed_files": 5,
        })
        mock_gh.get_pull_reviews = AsyncMock(return_value=[
            {"user": {"login": "frank"}, "state": "COMMENTED"},
        ])

        alerts = await detector.check_reviews_needed()
        assert len(alerts) == 1
        assert alerts[0].alert_type == "review_nudge"
        assert alerts[0].pr_number == 500

    @pytest.mark.asyncio
    async def test_skips_approved_pr(self, detector, db, mock_gh, settings):
        await queries.upsert_pr(db.conn, {
            "pr_number": 600, "title": "Already approved", "author": "grace",
            "state": "open", "head_sha": "fff666",
            "updated_at": _days_ago(4), "changed_files": 2,
        })
        mock_gh.get_pull_reviews = AsyncMock(return_value=[
            {"user": {"login": "hank"}, "state": "APPROVED"},
        ])

        alerts = await detector.check_reviews_needed()
        assert len(alerts) == 0

    @pytest.mark.asyncio
    async def test_skips_recently_updated_pr(self, detector, db, mock_gh, settings):
        await queries.upsert_pr(db.conn, {
            "pr_number": 700, "title": "Fresh PR", "author": "ivy",
            "state": "open", "head_sha": "ggg777",
            "updated_at": _days_ago(0), "changed_files": 1,
        })

        alerts = await detector.check_reviews_needed()
        assert len(alerts) == 0

    @pytest.mark.asyncio
    async def test_dedup_review_nudge(self, detector, db, mock_gh, settings):
        await queries.upsert_pr(db.conn, {
            "pr_number": 800, "title": "Dup review", "author": "jack",
            "state": "open", "head_sha": "hhh888",
            "updated_at": _days_ago(5), "changed_files": 3,
        })
        mock_gh.get_pull_reviews = AsyncMock(return_value=[])

        alerts1 = await detector.check_reviews_needed()
        assert len(alerts1) == 1

        alerts2 = await detector.check_reviews_needed()
        assert len(alerts2) == 0


class TestCheckAll:
    @pytest.mark.asyncio
    async def test_check_all_combines_results(self, detector, db, mock_gh, settings):
        # A stalled feature
        await queries.upsert_pr(db.conn, {
            "pr_number": 900, "title": "Stale", "author": "kate",
            "state": "open", "head_sha": "iii999",
            "updated_at": _days_ago(5), "changed_files": 2,
        })
        await queries.upsert_feature_item(db.conn, {
            "item_id": "all_item_1",
            "parent_issue": 14199,
            "title": "Stale feature",
            "state": "open",
            "linked_pr": 900,
        })

        # A PR needing review (different from above)
        await queries.upsert_pr(db.conn, {
            "pr_number": 901, "title": "No review", "author": "leo",
            "state": "open", "head_sha": "jjj000",
            "updated_at": _days_ago(4), "changed_files": 1,
        })
        mock_gh.get_pull_reviews = AsyncMock(return_value=[])

        alerts = await detector.check_all()
        types = {a.alert_type for a in alerts}
        assert "feature_stall" in types
        assert "review_nudge" in types
