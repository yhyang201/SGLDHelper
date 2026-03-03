"""Tool registry: maps OpenAI function-calling schemas to existing functions."""

from __future__ import annotations

import json
from typing import Any, Callable, Awaitable

import structlog

from sgldhelper.db import queries
from sgldhelper.db.engine import Database
from sgldhelper.github.ci_rerunner import CIRerunner
from sgldhelper.github.client import GitHubClient
from sgldhelper.github.issue_tracker import IssueTracker
from sgldhelper.config import Settings

log = structlog.get_logger()

_DIFF_MAX_CHARS = 30_000


class ToolRegistry:
    """Registry of tools exposed to Kimi K2.5 via function calling.

    Each tool has:
    - schema: OpenAI function-calling JSON schema
    - handler: async callable that executes the tool
    - requires_confirmation: if True, the bot asks the user before executing
    """

    def __init__(
        self,
        db: Database,
        gh: GitHubClient,
        rerunner: CIRerunner,
        issue_tracker: IssueTracker,
        settings: Settings,
    ) -> None:
        self._db = db
        self._gh = gh
        self._rerunner = rerunner
        self._issue_tracker = issue_tracker
        self._settings = settings
        self._tools: dict[str, _ToolDef] = {}
        self._register_all()

    def _register_all(self) -> None:
        """Register all available tools."""

        self._register(
            name="get_open_prs",
            description="List all currently open diffusion PRs with author, SHA, and title",
            parameters={"type": "object", "properties": {}, "required": []},
            handler=self._get_open_prs,
        )

        self._register(
            name="get_pr_details",
            description="Get detailed information about a specific PR including title, author, state, and labels",
            parameters={
                "type": "object",
                "required": ["pr_number"],
                "properties": {
                    "pr_number": {"type": "integer", "description": "PR number"},
                },
            },
            handler=self._get_pr_details,
        )

        self._register(
            name="get_ci_status",
            description="Get CI run results for a diffusion PR, including job names, status, failure categories",
            parameters={
                "type": "object",
                "required": ["pr_number"],
                "properties": {
                    "pr_number": {"type": "integer", "description": "PR number"},
                },
            },
            handler=self._get_ci_status,
        )

        self._register(
            name="get_feature_progress",
            description="Get feature roadmap progress for a specific issue, including completion percentage and item list",
            parameters={
                "type": "object",
                "required": ["issue_number"],
                "properties": {
                    "issue_number": {"type": "integer", "description": "GitHub issue number"},
                },
            },
            handler=self._get_feature_progress,
        )

        self._register(
            name="get_feature_items",
            description="Get all feature checklist items for a roadmap issue",
            parameters={
                "type": "object",
                "required": ["parent_issue"],
                "properties": {
                    "parent_issue": {"type": "integer", "description": "Parent issue number"},
                },
            },
            handler=self._get_feature_items,
        )

        self._register(
            name="get_pr_reviews",
            description="Get review status for a PR from GitHub (approved, changes requested, etc.)",
            parameters={
                "type": "object",
                "required": ["pr_number"],
                "properties": {
                    "pr_number": {"type": "integer", "description": "PR number"},
                },
            },
            handler=self._get_pr_reviews,
        )

        self._register(
            name="search_prs",
            description="Search PRs by title or author keyword",
            parameters={
                "type": "object",
                "required": ["query"],
                "properties": {
                    "query": {"type": "string", "description": "Search query (matches title or author)"},
                },
            },
            handler=self._search_prs,
        )

        self._register(
            name="get_recent_activity",
            description="Get recent PR, CI, and feature activity from the last N hours",
            parameters={
                "type": "object",
                "properties": {
                    "since_hours": {"type": "integer", "description": "Hours to look back (default 24)", "default": 24},
                },
                "required": [],
            },
            handler=self._get_recent_activity,
        )

        self._register(
            name="get_stalled_features",
            description="Find open feature items whose linked PRs haven't been updated recently",
            parameters={
                "type": "object",
                "properties": {},
                "required": [],
            },
            handler=self._get_stalled_features,
        )

        self._register(
            name="rerun_ci",
            description="Rerun failed CI jobs for a PR. This is a WRITE operation that triggers CI pipelines.",
            parameters={
                "type": "object",
                "required": ["pr_number"],
                "properties": {
                    "pr_number": {"type": "integer", "description": "PR number"},
                },
            },
            handler=self._rerun_ci,
            requires_confirmation=True,
        )

        self._register(
            name="get_my_preferences",
            description="Get the current user's saved preferences, tracked PRs, focus areas, and notes",
            parameters={"type": "object", "properties": {}, "required": []},
            handler=self._get_my_preferences,
        )

        self._register(
            name="update_tracked_prs",
            description="Add or remove PRs from the current user's tracking list",
            parameters={
                "type": "object",
                "required": ["action", "pr_numbers"],
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["add", "remove"],
                        "description": "Whether to add or remove PRs",
                    },
                    "pr_numbers": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "List of PR numbers to add or remove",
                    },
                },
            },
            handler=self._update_tracked_prs,
        )

        self._register(
            name="save_user_note",
            description="Save a note about the current user (e.g. their role, responsibilities, interests)",
            parameters={
                "type": "object",
                "required": ["note"],
                "properties": {
                    "note": {"type": "string", "description": "Note to save about the user"},
                },
            },
            handler=self._save_user_note,
        )

        self._register(
            name="link_pr_to_feature",
            description="Link a PR to a feature roadmap item so stall detection covers it",
            parameters={
                "type": "object",
                "required": ["pr_number", "item_id"],
                "properties": {
                    "pr_number": {"type": "integer", "description": "PR number to link"},
                    "item_id": {"type": "string", "description": "Feature item ID to bind the PR to"},
                },
            },
            handler=self._link_pr_to_feature,
        )

        self._register(
            name="get_unlinked_features",
            description="List open feature items that have no linked PR (candidates for binding)",
            parameters={"type": "object", "properties": {}, "required": []},
            handler=self._get_unlinked_features,
        )

        self._register(
            name="review_pr_code",
            description="Fetch the full diff of a PR for inline code review",
            parameters={
                "type": "object",
                "required": ["pr_number"],
                "properties": {
                    "pr_number": {"type": "integer", "description": "PR number"},
                },
            },
            handler=self._review_pr_code,
        )

    def _register(
        self,
        name: str,
        description: str,
        parameters: dict[str, Any],
        handler: Callable[..., Awaitable[Any]],
        requires_confirmation: bool = False,
    ) -> None:
        self._tools[name] = _ToolDef(
            schema={
                "type": "function",
                "function": {
                    "name": name,
                    "description": description,
                    "parameters": parameters,
                },
            },
            handler=handler,
            requires_confirmation=requires_confirmation,
        )

    # --- Public API ---

    def get_schemas(self) -> list[dict[str, Any]]:
        """Return all tool schemas in OpenAI function-calling format."""
        return [t.schema for t in self._tools.values()]

    def needs_confirmation(self, tool_name: str) -> bool:
        tool = self._tools.get(tool_name)
        return tool.requires_confirmation if tool else False

    async def execute(self, tool_name: str, arguments: str | dict[str, Any]) -> str:
        """Execute a tool by name with JSON arguments. Returns JSON string result."""
        tool = self._tools.get(tool_name)
        if not tool:
            return json.dumps({"error": f"Unknown tool: {tool_name}"})

        if isinstance(arguments, str):
            try:
                args = json.loads(arguments)
            except json.JSONDecodeError:
                return json.dumps({"error": f"Invalid JSON arguments: {arguments}"})
        else:
            args = arguments

        try:
            result = await tool.handler(**args)
            return json.dumps(result, default=str)
        except Exception as e:
            log.error("tool.execution_error", tool=tool_name, error=str(e))
            return json.dumps({"error": f"Tool execution failed: {e}"})

    # --- Tool handlers ---

    async def _get_open_prs(self) -> list[dict[str, Any]]:
        db_prs = await queries.get_open_prs(self._db.conn)
        if db_prs:
            return db_prs
        # DB is empty (pollers may not have run yet) — query GitHub directly
        try:
            gh_prs = await self._gh.get_open_pulls()
            return [
                {
                    "pr_number": pr["number"],
                    "title": pr["title"],
                    "author": pr.get("user", {}).get("login", "unknown"),
                    "state": pr["state"],
                    "head_sha": pr.get("head", {}).get("sha", "")[:8],
                    "updated_at": pr.get("updated_at", ""),
                    "html_url": pr.get("html_url", ""),
                }
                for pr in gh_prs
            ]
        except Exception as e:
            log.warning("tool.github_fallback_failed", error=str(e))
            return []

    async def _get_pr_details(self, pr_number: int) -> dict[str, Any]:
        db_pr = await queries.get_pr(self._db.conn, pr_number)
        try:
            gh_pr = await self._gh.get_pull(pr_number)
            return {
                "pr_number": pr_number,
                "title": gh_pr.get("title", db_pr["title"] if db_pr else "Unknown"),
                "author": gh_pr.get("user", {}).get("login", "unknown"),
                "state": gh_pr.get("state", "unknown"),
                "head_sha": gh_pr.get("head", {}).get("sha", "")[:8],
                "labels": [l["name"] for l in gh_pr.get("labels", [])],
                "mergeable": gh_pr.get("mergeable"),
                "review_comments": gh_pr.get("review_comments", 0),
                "updated_at": gh_pr.get("updated_at", ""),
                "html_url": gh_pr.get("html_url", ""),
            }
        except Exception:
            if db_pr:
                return dict(db_pr)
            return {"error": f"PR #{pr_number} not found"}

    async def _get_ci_status(self, pr_number: int) -> list[dict[str, Any]]:
        db_runs = await queries.get_ci_runs_for_pr(self._db.conn, pr_number)
        if db_runs:
            return [
                {
                    "run_id": r["run_id"],
                    "job_name": r["job_name"],
                    "status": r["status"],
                    "conclusion": r["conclusion"],
                    "failure_category": r["failure_category"],
                    "failure_summary": (r["failure_summary"] or "")[:300],
                    "auto_rerun_count": r["auto_rerun_count"],
                }
                for r in db_runs[:15]
            ]
        # Fallback: query GitHub directly for CI runs
        try:
            pr = await self._gh.get_pull(pr_number)
            head_sha = pr.get("head", {}).get("sha", "")
            if not head_sha:
                return []
            workflow_runs = await self._gh.get_workflow_runs_for_ref(head_sha)
            results = []
            for wr in workflow_runs[:10]:
                results.append({
                    "run_id": wr["id"],
                    "job_name": wr.get("name", "unknown"),
                    "status": wr.get("status", "unknown"),
                    "conclusion": wr.get("conclusion"),
                    "html_url": wr.get("html_url", ""),
                })
            return results
        except Exception as e:
            log.warning("tool.ci_github_fallback_failed", error=str(e))
            return []

    async def _get_feature_progress(self, issue_number: int) -> dict[str, Any]:
        progress = await self._issue_tracker.get_progress(issue_number)
        return {
            "issue_number": progress.issue_number,
            "title": progress.title,
            "total": progress.total,
            "completed": progress.completed,
            "percent": round(progress.percent, 1),
            "items": [
                {"title": i["title"] if isinstance(i, dict) else i.title,
                 "state": i["state"] if isinstance(i, dict) else i.state}
                for i in progress.items[:20]
            ],
        }

    async def _get_feature_items(self, parent_issue: int) -> list[dict[str, Any]]:
        return await queries.get_feature_items(self._db.conn, parent_issue)

    async def _get_pr_reviews(self, pr_number: int) -> list[dict[str, Any]]:
        reviews = await self._gh.get_pull_reviews(pr_number)
        return [
            {
                "user": r.get("user", {}).get("login", "unknown"),
                "state": r.get("state", ""),
                "submitted_at": r.get("submitted_at", ""),
            }
            for r in reviews
        ]

    async def _search_prs(self, query: str) -> list[dict[str, Any]]:
        return await queries.search_prs(self._db.conn, query)

    async def _get_recent_activity(self, since_hours: int = 24) -> dict[str, Any]:
        return await queries.get_recent_activity(self._db.conn, since_hours)

    async def _get_stalled_features(self) -> list[dict[str, Any]]:
        return await queries.get_stalled_features(
            self._db.conn, self._settings.stall_days_threshold
        )

    async def _rerun_ci(self, pr_number: int) -> dict[str, Any]:
        results = await self._rerunner.manual_rerun(pr_number)
        return {
            "pr_number": pr_number,
            "results": [
                {"run_id": r.run_id, "triggered": r.triggered, "reason": r.reason}
                for r in results
            ],
        }

    async def _get_my_preferences(self) -> dict[str, Any]:
        from sgldhelper.ai.conversation import _current_user
        user_id = _current_user.get()
        mem = await queries.get_user_memory(self._db.conn, user_id)
        if not mem:
            return {"user_id": user_id, "tracked_prs": [], "focus_areas": [], "preferences": {}, "notes": ""}
        return {
            "user_id": user_id,
            "tracked_prs": json.loads(mem["tracked_prs"]),
            "focus_areas": json.loads(mem["focus_areas"]),
            "preferences": json.loads(mem["preferences"]),
            "notes": mem["notes"],
        }

    async def _update_tracked_prs(self, action: str, pr_numbers: list[int]) -> dict[str, Any]:
        from sgldhelper.ai.conversation import _current_user
        user_id = _current_user.get()
        updated: list[int] = []
        for pr in pr_numbers:
            if action == "add":
                updated = await queries.add_tracked_pr(self._db.conn, user_id, pr)
            else:
                updated = await queries.remove_tracked_pr(self._db.conn, user_id, pr)
        return {"user_id": user_id, "action": action, "tracked_prs": updated}

    async def _save_user_note(self, note: str) -> dict[str, Any]:
        from sgldhelper.ai.conversation import _current_user
        user_id = _current_user.get()
        await queries.save_user_note(self._db.conn, user_id, note)
        return {"user_id": user_id, "note": note, "saved": True}

    async def _link_pr_to_feature(self, pr_number: int, item_id: str) -> dict[str, Any]:
        pr = await queries.get_pr(self._db.conn, pr_number)
        if not pr:
            return {"error": f"PR #{pr_number} not found in database"}
        updated = await queries.update_feature_linked_pr(self._db.conn, item_id, pr_number)
        if not updated:
            return {"error": f"Feature item '{item_id}' not found"}
        return {"success": True, "item_id": item_id, "linked_pr": pr_number}

    async def _get_unlinked_features(self) -> list[dict[str, Any]]:
        return await queries.get_unlinked_features(self._db.conn)

    async def _review_pr_code(self, pr_number: int) -> dict[str, Any]:
        diff = await self._gh.get_pull_diff(pr_number)
        if not diff:
            return {"error": f"PR #{pr_number} returned an empty diff"}
        truncated = len(diff) > _DIFF_MAX_CHARS
        if truncated:
            diff = diff[:_DIFF_MAX_CHARS]
        return {"pr_number": pr_number, "diff": diff, "truncated": truncated}


class _ToolDef:
    __slots__ = ("schema", "handler", "requires_confirmation")

    def __init__(
        self,
        schema: dict[str, Any],
        handler: Callable[..., Awaitable[Any]],
        requires_confirmation: bool,
    ) -> None:
        self.schema = schema
        self.handler = handler
        self.requires_confirmation = requires_confirmation
