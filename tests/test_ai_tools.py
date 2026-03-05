"""Tests for AI tool registry and execution."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from sgldhelper.ai.tools import ToolRegistry
from sgldhelper.config import Settings
from sgldhelper.db.engine import Database


@pytest.fixture
def mock_gh():
    gh = AsyncMock()
    gh.get_pull = AsyncMock(return_value={
        "title": "Add diffusion model",
        "user": {"login": "alice"},
        "state": "open",
        "head": {"sha": "abc12345"},
        "labels": [{"name": "diffusion"}],
        "mergeable": True,
        "review_comments": 2,
        "updated_at": "2025-03-01T00:00:00Z",
        "html_url": "https://github.com/sgl-project/sglang/pull/1234",
    })
    gh.get_pull_reviews = AsyncMock(return_value=[
        {"user": {"login": "bob"}, "state": "APPROVED", "submitted_at": "2025-03-01T12:00:00Z"},
    ])
    return gh


@pytest.fixture
async def registry(db, mock_gh, settings):
    return ToolRegistry(db, mock_gh, settings)


class TestToolSchemas:
    def test_schemas_are_valid(self, registry):
        schemas = registry.get_schemas()
        assert len(schemas) == 16

        for schema in schemas:
            assert schema["type"] == "function"
            assert "function" in schema
            assert "name" in schema["function"]
            assert "description" in schema["function"]
            assert "parameters" in schema["function"]

    def test_tool_names(self, registry):
        schemas = registry.get_schemas()
        names = {s["function"]["name"] for s in schemas}
        expected = {
            "get_open_prs", "get_pr_details",
            "get_pr_reviews", "search_prs", "search_github_prs",
            "get_recent_activity",
            "get_my_preferences", "update_tracked_prs",
            "save_user_note", "review_pr_code",
            "get_ci_status", "trigger_ci", "cancel_auto_merge",
            "merge_pr", "get_merge_ready_prs", "run_health_check",
        }
        assert names == expected


class TestToolConfirmation:
    def test_read_tools_no_confirmation(self, registry):
        assert registry.needs_confirmation("get_open_prs") is False
        assert registry.needs_confirmation("get_pr_details") is False
        assert registry.needs_confirmation("get_ci_status") is False

    def test_write_tools_need_confirmation(self, registry):
        assert registry.needs_confirmation("trigger_ci") is False
        assert registry.needs_confirmation("cancel_auto_merge") is True
        assert registry.needs_confirmation("merge_pr") is True

    def test_unknown_tool_no_confirmation(self, registry):
        assert registry.needs_confirmation("nonexistent") is False


class TestToolExecution:
    @pytest.mark.asyncio
    async def test_get_open_prs(self, registry, db):
        from sgldhelper.db import queries
        await queries.upsert_pr(db.conn, {
            "pr_number": 1234, "title": "Test PR", "author": "alice",
            "state": "open", "head_sha": "abc123", "updated_at": "2025-03-01",
            "changed_files": 5,
        })
        result_str = await registry.execute("get_open_prs", "{}")
        result = json.loads(result_str)
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["pr_number"] == 1234

    @pytest.mark.asyncio
    async def test_get_pr_details(self, registry, mock_gh):
        result_str = await registry.execute("get_pr_details", '{"pr_number": 1234}')
        result = json.loads(result_str)
        assert result["title"] == "Add diffusion model"
        assert result["author"] == "alice"
        mock_gh.get_pull.assert_called_once_with(1234)

    @pytest.mark.asyncio
    async def test_get_pr_reviews(self, registry, mock_gh):
        result_str = await registry.execute("get_pr_reviews", '{"pr_number": 1234}')
        result = json.loads(result_str)
        assert len(result) == 1
        assert result[0]["user"] == "bob"
        assert result[0]["state"] == "APPROVED"

    @pytest.mark.asyncio
    async def test_unknown_tool(self, registry):
        result_str = await registry.execute("nonexistent_tool", "{}")
        result = json.loads(result_str)
        assert "error" in result

    @pytest.mark.asyncio
    async def test_invalid_json_arguments(self, registry):
        result_str = await registry.execute("get_pr_details", "not-json")
        result = json.loads(result_str)
        assert "error" in result

    @pytest.mark.asyncio
    async def test_search_prs(self, registry, db):
        from sgldhelper.db import queries
        await queries.upsert_pr(db.conn, {
            "pr_number": 5678, "title": "Add ControlNet support", "author": "bob",
            "state": "open", "head_sha": "def456", "updated_at": "2025-03-01",
            "changed_files": 3,
        })
        result_str = await registry.execute("search_prs", '{"query": "ControlNet"}')
        result = json.loads(result_str)
        assert len(result) == 1
        assert result[0]["pr_number"] == 5678

    @pytest.mark.asyncio
    async def test_execute_with_dict_arguments(self, registry, mock_gh):
        result_str = await registry.execute("get_pr_details", {"pr_number": 1234})
        result = json.loads(result_str)
        assert result["title"] == "Add diffusion model"


class TestCodeReview:
    @pytest.mark.asyncio
    async def test_review_pr_code_returns_diff(self, registry, mock_gh):
        mock_gh.get_pull_diff = AsyncMock(return_value="diff --git a/file.py b/file.py\n+hello")
        result_str = await registry.execute("review_pr_code", '{"pr_number": 42}')
        result = json.loads(result_str)
        assert result["pr_number"] == 42
        assert "diff --git" in result["diff"]
        assert result["truncated"] is False
        mock_gh.get_pull_diff.assert_called_once_with(42)

    @pytest.mark.asyncio
    async def test_review_pr_code_empty_diff(self, registry, mock_gh):
        mock_gh.get_pull_diff = AsyncMock(return_value="")
        result_str = await registry.execute("review_pr_code", '{"pr_number": 99}')
        result = json.loads(result_str)
        assert "error" in result

    @pytest.mark.asyncio
    async def test_review_pr_code_truncates_large_diff(self, registry, mock_gh):
        from sgldhelper.ai.tools import _DIFF_MAX_CHARS
        large_diff = "x" * (_DIFF_MAX_CHARS + 5000)
        mock_gh.get_pull_diff = AsyncMock(return_value=large_diff)
        result_str = await registry.execute("review_pr_code", '{"pr_number": 1}')
        result = json.loads(result_str)
        assert result["truncated"] is True
        assert len(result["diff"]) == _DIFF_MAX_CHARS

    @pytest.mark.asyncio
    async def test_review_pr_code_not_truncated_at_limit(self, registry, mock_gh):
        from sgldhelper.ai.tools import _DIFF_MAX_CHARS
        exact_diff = "y" * _DIFF_MAX_CHARS
        mock_gh.get_pull_diff = AsyncMock(return_value=exact_diff)
        result_str = await registry.execute("review_pr_code", '{"pr_number": 2}')
        result = json.loads(result_str)
        assert result["truncated"] is False
        assert len(result["diff"]) == _DIFF_MAX_CHARS

    @pytest.mark.asyncio
    async def test_review_pr_code_no_confirmation(self, registry):
        assert registry.needs_confirmation("review_pr_code") is False


class TestCITools:
    @pytest.mark.asyncio
    async def test_get_ci_status_no_monitor(self, registry):
        """get_ci_status returns error when CI monitor not set."""
        result_str = await registry.execute("get_ci_status", '{"pr_number": 1234}')
        result = json.loads(result_str)
        assert "error" in result

    @pytest.mark.asyncio
    async def test_trigger_ci_no_monitor(self, registry):
        result_str = await registry.execute("trigger_ci", '{"pr_number": 1234}')
        result = json.loads(result_str)
        assert "error" in result

    @pytest.mark.asyncio
    async def test_cancel_auto_merge_no_manager(self, registry):
        result_str = await registry.execute("cancel_auto_merge", '{"pr_number": 1234}')
        result = json.loads(result_str)
        assert "error" in result

    @pytest.mark.asyncio
    async def test_search_github_prs(self, registry, mock_gh):
        mock_gh.search_issues = AsyncMock(return_value=[
            {
                "number": 5555,
                "title": "Fix diffusion interpolation bug",
                "user": {"login": "bob"},
                "state": "closed",
                "labels": [{"name": "diffusion"}],
                "updated_at": "2025-03-01T00:00:00Z",
                "html_url": "https://github.com/sgl-project/sglang/pull/5555",
            },
        ])
        result_str = await registry.execute(
            "search_github_prs",
            '{"keywords": ["diffusion", "interpolation", "fix"]}',
        )
        result = json.loads(result_str)
        assert len(result) == 1
        assert result[0]["pr_number"] == 5555
        assert result[0]["title"] == "Fix diffusion interpolation bug"
        mock_gh.search_issues.assert_called_once_with(
            ["diffusion", "interpolation", "fix"],
            is_pr=True, state=None, max_results=10,
        )

    @pytest.mark.asyncio
    async def test_search_github_prs_with_state(self, registry, mock_gh):
        mock_gh.search_issues = AsyncMock(return_value=[])
        result_str = await registry.execute(
            "search_github_prs",
            '{"keywords": ["test"], "state": "open", "max_results": 5}',
        )
        result = json.loads(result_str)
        assert result == []
        mock_gh.search_issues.assert_called_once_with(
            ["test"], is_pr=True, state="open", max_results=5,
        )

    @pytest.mark.asyncio
    async def test_merge_pr(self, registry, mock_gh):
        mock_gh.merge_pull = AsyncMock(return_value={"message": "Pull Request successfully merged"})
        result_str = await registry.execute("merge_pr", '{"pr_number": 1234}')
        result = json.loads(result_str)
        assert result["merged"] is True
        assert result["merge_method"] == "squash"
        mock_gh.merge_pull.assert_called_once_with(1234, merge_method="squash")

    @pytest.mark.asyncio
    async def test_merge_pr_custom_method(self, registry, mock_gh):
        mock_gh.merge_pull = AsyncMock(return_value={"message": "merged"})
        result_str = await registry.execute("merge_pr", '{"pr_number": 1234, "merge_method": "rebase"}')
        result = json.loads(result_str)
        assert result["merge_method"] == "rebase"
        mock_gh.merge_pull.assert_called_once_with(1234, merge_method="rebase")

    @pytest.mark.asyncio
    async def test_merge_pr_not_mergeable(self, registry, mock_gh):
        mock_gh.merge_pull = AsyncMock(side_effect=Exception("405 Method Not Allowed"))
        result_str = await registry.execute("merge_pr", '{"pr_number": 1234}')
        result = json.loads(result_str)
        assert "error" in result
        assert "not mergeable" in result["error"]

    @pytest.mark.asyncio
    async def test_get_merge_ready_prs_empty(self, registry, db):
        """Returns empty when no open PRs in DB."""
        result_str = await registry.execute("get_merge_ready_prs", "{}")
        result = json.loads(result_str)
        assert result == []

    @pytest.mark.asyncio
    async def test_get_merge_ready_prs_finds_ready(self, registry, db, mock_gh, settings):
        """Returns PRs that are open + mergeable + approved + CI passed."""
        from sgldhelper.ci.monitor import CIMonitor, CIOverallStatus, CIStatus
        from sgldhelper.db import queries

        # Insert an open PR into DB
        await queries.upsert_pr(db.conn, {
            "pr_number": 9999, "title": "Ready PR", "author": "alice",
            "state": "open", "head_sha": "aaa111", "updated_at": "2025-03-01",
            "changed_files": 3,
        })

        # Mock GitHub responses
        mock_gh.get_pull = AsyncMock(return_value={
            "number": 9999, "title": "Ready PR", "state": "open",
            "mergeable": True, "head": {"sha": "aaa111"},
            "user": {"login": "alice"}, "labels": [{"name": "run-ci"}],
            "html_url": "https://github.com/sgl-project/sglang/pull/9999",
            "updated_at": "2025-03-01",
        })
        mock_gh.get_pull_reviews = AsyncMock(return_value=[
            {"user": {"login": "bob"}, "state": "APPROVED"},
        ])

        ci_monitor = AsyncMock()
        ci_monitor.check_pr_ci = AsyncMock(return_value=CIStatus(
            pr_number=9999, head_sha="aaa111",
            overall=CIOverallStatus.PASSED, has_run_ci_label=True,
            all_runs_completed=True,
        ))
        registry.set_ci_components(ci_monitor, None)

        result_str = await registry.execute("get_merge_ready_prs", "{}")
        result = json.loads(result_str)
        assert len(result) == 1
        assert result[0]["pr_number"] == 9999

    @pytest.mark.asyncio
    async def test_get_merge_ready_prs_skips_unapproved(self, registry, db, mock_gh, settings):
        """Skips PRs without approval."""
        from sgldhelper.db import queries

        await queries.upsert_pr(db.conn, {
            "pr_number": 8888, "title": "No approval", "author": "bob",
            "state": "open", "head_sha": "bbb222", "updated_at": "2025-03-01",
            "changed_files": 1,
        })
        mock_gh.get_pull = AsyncMock(return_value={
            "number": 8888, "title": "No approval", "state": "open",
            "mergeable": True, "head": {"sha": "bbb222"},
            "user": {"login": "bob"}, "labels": [],
            "updated_at": "2025-03-01",
        })
        mock_gh.get_pull_reviews = AsyncMock(return_value=[])

        result_str = await registry.execute("get_merge_ready_prs", "{}")
        result = json.loads(result_str)
        assert result == []

    @pytest.mark.asyncio
    async def test_get_ci_status_with_monitor(self, registry, mock_gh, settings):
        from sgldhelper.ci.monitor import CIMonitor, CIOverallStatus, CIStatus
        ci_monitor = AsyncMock()
        ci_monitor.check_pr_ci = AsyncMock(return_value=CIStatus(
            pr_number=1234,
            head_sha="abc12345",
            overall=CIOverallStatus.PASSED,
            has_run_ci_label=True,
            nvidia_jobs=[],
            amd_jobs=[],
            failed_jobs=[],
            all_runs_completed=True,
        ))
        registry.set_ci_components(ci_monitor, None)
        result_str = await registry.execute("get_ci_status", '{"pr_number": 1234}')
        result = json.loads(result_str)
        assert result["overall"] == "passed"
        assert result["has_run_ci_label"] is True
