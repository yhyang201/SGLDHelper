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
def mock_rerunner():
    rerunner = AsyncMock()
    rerunner.manual_rerun = AsyncMock(return_value=[
        MagicMock(run_id=100, triggered=True, reason="Manual rerun triggered"),
    ])
    return rerunner


@pytest.fixture
def mock_issue_tracker():
    tracker = AsyncMock()
    tracker.get_progress = AsyncMock(return_value=MagicMock(
        issue_number=14199,
        title="Diffusion Roadmap",
        total=10,
        completed=3,
        percent=30.0,
        items=[
            {"title": "Add SDXL support", "state": "completed"},
            {"title": "Add ControlNet", "state": "open"},
        ],
    ))
    return tracker


@pytest.fixture
async def registry(db, mock_gh, mock_rerunner, mock_issue_tracker, settings):
    return ToolRegistry(db, mock_gh, mock_rerunner, mock_issue_tracker, settings)


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
            "get_open_prs", "get_pr_details", "get_ci_status",
            "get_feature_progress", "get_feature_items", "get_pr_reviews",
            "search_prs", "get_recent_activity", "get_stalled_features",
            "rerun_ci", "get_my_preferences", "update_tracked_prs",
            "save_user_note", "link_pr_to_feature", "get_unlinked_features",
            "review_pr_code",
        }
        assert names == expected


class TestToolConfirmation:
    def test_rerun_ci_requires_confirmation(self, registry):
        assert registry.needs_confirmation("rerun_ci") is True

    def test_read_tools_no_confirmation(self, registry):
        assert registry.needs_confirmation("get_open_prs") is False
        assert registry.needs_confirmation("get_pr_details") is False
        assert registry.needs_confirmation("get_ci_status") is False

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
    async def test_rerun_ci(self, registry, mock_rerunner):
        result_str = await registry.execute("rerun_ci", '{"pr_number": 1234}')
        result = json.loads(result_str)
        assert result["pr_number"] == 1234
        assert result["results"][0]["triggered"] is True
        mock_rerunner.manual_rerun.assert_called_once_with(1234)

    @pytest.mark.asyncio
    async def test_get_feature_progress(self, registry, mock_issue_tracker):
        result_str = await registry.execute("get_feature_progress", '{"issue_number": 14199}')
        result = json.loads(result_str)
        assert result["total"] == 10
        assert result["completed"] == 3
        assert result["percent"] == 30.0

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


class TestFeatureBinding:
    @pytest.mark.asyncio
    async def test_get_unlinked_features(self, registry, db):
        from sgldhelper.db import queries
        # One open+unlinked, one open+linked, one completed+unlinked
        await queries.upsert_feature_item(db.conn, {
            "item_id": "14199/1", "parent_issue": 14199, "title": "SDXL support",
            "item_type": "checkbox", "state": "open", "linked_pr": None,
        })
        await queries.upsert_feature_item(db.conn, {
            "item_id": "14199/2", "parent_issue": 14199, "title": "ControlNet",
            "item_type": "checkbox", "state": "open", "linked_pr": 5678,
        })
        await queries.upsert_feature_item(db.conn, {
            "item_id": "14199/3", "parent_issue": 14199, "title": "Done item",
            "item_type": "checkbox", "state": "completed", "linked_pr": None,
        })
        result_str = await registry.execute("get_unlinked_features", "{}")
        result = json.loads(result_str)
        assert len(result) == 1
        assert result[0]["item_id"] == "14199/1"

    @pytest.mark.asyncio
    async def test_link_pr_to_feature_success(self, registry, db):
        from sgldhelper.db import queries
        await queries.upsert_pr(db.conn, {
            "pr_number": 9999, "title": "Impl SDXL", "author": "alice",
            "state": "open", "head_sha": "aaa111", "updated_at": "2025-03-01",
            "changed_files": 3,
        })
        await queries.upsert_feature_item(db.conn, {
            "item_id": "14199/1", "parent_issue": 14199, "title": "SDXL support",
            "item_type": "checkbox", "state": "open",
        })
        result_str = await registry.execute(
            "link_pr_to_feature", '{"pr_number": 9999, "item_id": "14199/1"}'
        )
        result = json.loads(result_str)
        assert result["success"] is True
        assert result["linked_pr"] == 9999

    @pytest.mark.asyncio
    async def test_link_pr_to_feature_invalid_pr(self, registry, db):
        from sgldhelper.db import queries
        await queries.upsert_feature_item(db.conn, {
            "item_id": "14199/1", "parent_issue": 14199, "title": "SDXL support",
            "item_type": "checkbox", "state": "open",
        })
        result_str = await registry.execute(
            "link_pr_to_feature", '{"pr_number": 77777, "item_id": "14199/1"}'
        )
        result = json.loads(result_str)
        assert "error" in result
        assert "not found" in result["error"]

    @pytest.mark.asyncio
    async def test_link_pr_to_feature_invalid_item(self, registry, db):
        from sgldhelper.db import queries
        await queries.upsert_pr(db.conn, {
            "pr_number": 9999, "title": "Impl SDXL", "author": "alice",
            "state": "open", "head_sha": "aaa111", "updated_at": "2025-03-01",
            "changed_files": 3,
        })
        result_str = await registry.execute(
            "link_pr_to_feature", '{"pr_number": 9999, "item_id": "nonexistent/99"}'
        )
        result = json.loads(result_str)
        assert "error" in result
        assert "not found" in result["error"]


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
