"""Tests for the daily code quality reporter."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from sgldhelper.ci.code_quality import CodeQualityReporter, _parse_report


_REPORT_WITH_SCORES = (
    "PR #123: Good quality, score 8/10\n"
    "*Daily Overall Score*: 8/10\n"
    '<!--SCORES:{"overall":8,"prs":[{"pr":123,"score":8}]}-->'
)

_REPORT_WITH_ALERT = (
    "PR #200: Terrible code\n"
    "*Daily Overall Score*: 2/10\n"
    '<!--SCORES:{"overall":2,"prs":[{"pr":200,"score":2,"reason":"massive duplication"}]}-->'
)

_REPORT_NO_SCORES = "PR #123: Good quality, score 8/10\n*Daily Overall Score*: 8/10"


@pytest.fixture
def mock_kimi():
    kimi = AsyncMock()
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = _REPORT_WITH_SCORES
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


class TestParseReport:
    def test_parse_with_scores(self):
        result = _parse_report(_REPORT_WITH_SCORES, threshold=3)
        assert result.overall_score == 8
        assert len(result.pr_scores) == 1
        assert result.pr_scores[0]["pr"] == 123
        assert result.alert_prs == []
        assert "<!--SCORES:" not in result.display_text

    def test_parse_with_alert(self):
        result = _parse_report(_REPORT_WITH_ALERT, threshold=3)
        assert result.overall_score == 2
        assert len(result.alert_prs) == 1
        assert result.alert_prs[0]["pr"] == 200
        assert result.alert_prs[0]["reason"] == "massive duplication"

    def test_parse_no_scores_line(self):
        result = _parse_report(_REPORT_NO_SCORES, threshold=3)
        assert result.overall_score is None
        assert result.alert_prs == []
        assert result.display_text == _REPORT_NO_SCORES

    def test_parse_malformed_json(self):
        raw = "Some report\n<!--SCORES:{bad json}-->"
        result = _parse_report(raw, threshold=3)
        assert result.display_text == "Some report"
        assert result.overall_score is None

    def test_threshold_boundary(self):
        raw = '<!--SCORES:{"overall":3,"prs":[{"pr":1,"score":3},{"pr":2,"score":4}]}-->'
        result = _parse_report(raw, threshold=3)
        assert len(result.alert_prs) == 1
        assert result.alert_prs[0]["pr"] == 1


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
        report_text, pr_count, alert_prs = callback.call_args[0]
        assert pr_count == 1
        assert "<!--SCORES:" not in report_text
        assert alert_prs == []

    @pytest.mark.asyncio
    async def test_poll_with_alert(self, reporter, db, mock_kimi):
        """Should pass alert PRs to callback when score is below threshold."""
        response = MagicMock()
        response.choices = [MagicMock()]
        response.choices[0].message.content = _REPORT_WITH_ALERT
        mock_kimi.chat.return_value = response

        await db.conn.execute(
            """INSERT INTO pull_requests
               (pr_number, title, author, state, head_sha, changed_files, created_at, updated_at)
               VALUES (200, 'Bad PR', 'dev', 'merged', 'bad200', 5,
                       datetime('now'), datetime('now'))""",
        )
        await db.conn.commit()

        callback = AsyncMock()
        reporter.set_callback(callback)
        await reporter.poll()

        callback.assert_called_once()
        _, _, alert_prs = callback.call_args[0]
        assert len(alert_prs) == 1
        assert alert_prs[0]["pr"] == 200

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
        await reporter.poll()

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

        callback.assert_called_once()


class TestBuildCodeQualityReport:
    def test_message_format_no_alert(self):
        from sgldhelper.slack.messages import build_code_quality_report

        msg = build_code_quality_report("Test report content", 3, [], [])
        assert msg["text"] == "Daily Code Quality Report (3 PRs)"
        assert len(msg["blocks"]) == 1
        assert "Test report content" in msg["blocks"][0]["text"]["text"]

    def test_message_format_with_alert(self):
        from sgldhelper.slack.messages import build_code_quality_report

        alert_prs = [{"pr": 200, "score": 2, "reason": "massive duplication"}]
        msg = build_code_quality_report(
            "Report", 1, alert_prs, ["U_MICK"], "sgl-project/sglang",
        )
        # Should have: report section + divider + alert section
        assert len(msg["blocks"]) == 3
        alert_text = msg["blocks"][2]["text"]["text"]
        assert ":rotating_light:" in alert_text
        assert "<@U_MICK>" in alert_text
        assert "PR #200" in alert_text
        assert "massive duplication" in alert_text

    def test_no_alert_block_when_no_user_ids(self):
        from sgldhelper.slack.messages import build_code_quality_report

        alert_prs = [{"pr": 200, "score": 1}]
        msg = build_code_quality_report("Report", 1, alert_prs, [])
        assert len(msg["blocks"]) == 1  # no alert block
