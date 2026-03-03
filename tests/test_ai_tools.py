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
        assert len(schemas) == 10

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
            "rerun_ci",
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
