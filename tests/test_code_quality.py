"""Tests for the daily code quality reporter."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from sgldhelper.ci.code_quality import CodeQualityReporter
from sgldhelper.db import queries


@pytest.fixture
def mock_kimi():
    kimi = AsyncMock()
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = (
        ":mag: *Daily Code Quality Report*\n"
        "PR #123: Good quality, score 8/10\n"
        "*Daily Overall Score*: 8/10"
    )
    kimi.chat.return_value = response
    kimi.extract_usage.return_value = (500, 200)
    return kimi


@pytest.fixture
def mock_gh():
    gh = AsyncMock()
    gh.get_pull_diff.return_value = "diff --git a/foo.py\n+def bar(): pass"
    return gh


@pytest.fixture
def reporter(mock_kimi, mock_gh, db, settings):
    return CodeQualityReporter(mock_kimi, mock_gh, db, settings)


class TestCodeQualityReporter:
    @pytest.mark.asyncio
    async def test_poll_no_merged_prs(self, reporter, db):
        """Should skip when no PRs merged today."""
        callback = AsyncMock()
        reporter.set_callback(callback)
        await reporter.poll()
        callback.assert_not_called()

    @pytest.mark.asyncio
    async def test_poll_with_merged_prs(self, reporter, db, mock_kimi):
        """Should generate report when merged PRs exist."""
        # Insert a merged PR with today's date
        await db.conn.execute(
            """INSERT INTO pull_requests
               (pr_number, title, author, state, head_sha, changed_files, created_at, updated_at)
               VALUES (123, 'Add feature X', 'testuser', 'merged', 'abc123', 3,
                       datetime('now'), datetime('now'))""",
        )
        await db.conn.commit()

        callback = AsyncMock()
        reporter.set_callback(callback)
        await reporter.poll()

        callback.assert_called_once()
        report_text, pr_count = callback.call_args[0]
        assert pr_count == 1
        assert "Daily Code Quality Report" in report_text

    @pytest.mark.asyncio
    async def test_poll_runs_once_per_day(self, reporter, db):
        """Should not generate a second report on the same day."""
        await db.conn.execute(
            """INSERT INTO pull_requests
               (pr_number, title, author, state, head_sha, changed_files, created_at, updated_at)
               VALUES (456, 'Fix bug', 'dev', 'merged', 'def456', 1,
                       datetime('now'), datetime('now'))""",
        )
        await db.conn.commit()

        callback = AsyncMock()
        reporter.set_callback(callback)

        await reporter.poll()
        await reporter.poll()  # second call same day

        assert callback.call_count == 1

    @pytest.mark.asyncio
    async def test_diff_truncation(self, reporter, mock_gh, db):
        """Should truncate large diffs."""
        mock_gh.get_pull_diff.return_value = "x" * 20000

        await db.conn.execute(
            """INSERT INTO pull_requests
               (pr_number, title, author, state, head_sha, changed_files, created_at, updated_at)
               VALUES (789, 'Big change', 'dev', 'merged', 'ghi789', 10,
                       datetime('now'), datetime('now'))""",
        )
        await db.conn.commit()

        callback = AsyncMock()
        reporter.set_callback(callback)
        await reporter.poll()

        callback.assert_called_once()

    @pytest.mark.asyncio
    async def test_diff_fetch_failure_handled(self, reporter, mock_gh, db):
        """Should continue even if diff fetch fails."""
        mock_gh.get_pull_diff.side_effect = Exception("API error")

        await db.conn.execute(
            """INSERT INTO pull_requests
               (pr_number, title, author, state, head_sha, changed_files, created_at, updated_at)
               VALUES (101, 'Some PR', 'dev', 'merged', 'xyz101', 2,
                       datetime('now'), datetime('now'))""",
        )
        await db.conn.commit()

        callback = AsyncMock()
        reporter.set_callback(callback)
        await reporter.poll()

        # Should still generate report with "(diff unavailable)"
        callback.assert_called_once()


class TestBuildCodeQualityReport:
    def test_message_format(self):
        from sgldhelper.slack.messages import build_code_quality_report

        msg = build_code_quality_report("Test report content", 3)
        assert msg["text"] == "Daily Code Quality Report (3 PRs)"
        assert msg["blocks"][0]["type"] == "section"
        assert "Test report content" in msg["blocks"][0]["text"]["text"]
        assert "3 diffusion PR(s)" in msg["blocks"][0]["text"]["text"]
