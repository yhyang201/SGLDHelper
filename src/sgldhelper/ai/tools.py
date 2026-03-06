"""Tool registry: maps OpenAI function-calling schemas to existing functions."""

from __future__ import annotations

import json
from typing import Any, Callable, Awaitable

import structlog

from sgldhelper.db import queries
from sgldhelper.db.engine import Database
from sgldhelper.github.client import GitHubClient
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
        settings: Settings,
    ) -> None:
        self._db = db
        self._gh = gh
        self._settings = settings
        self._ci_monitor: Any = None
        self._auto_merge: Any = None
        self._tools: dict[str, _ToolDef] = {}
        self._register_all()

    def set_ci_components(self, ci_monitor: Any, auto_merge: Any, health_checker: Any = None) -> None:
        """Inject CI monitor, auto-merge manager, and health checker after construction."""
        self._ci_monitor = ci_monitor
        self._auto_merge = auto_merge
        self._health_checker = health_checker

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
            description="Search PRs in the local database by title or author keyword. Use search_github_prs for broader search across the whole repo.",
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
            name="search_github_prs",
            description=(
                "Search PRs across the entire GitHub repo using the GitHub Search API. "
                "Accepts multiple keywords for fuzzy matching. "
                "Use this when the user vaguely describes a PR they remember."
            ),
            parameters={
                "type": "object",
                "required": ["keywords"],
                "properties": {
                    "keywords": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of keywords to search for (e.g. ['diffusion', 'interpolation', 'fix'])",
                    },
                    "state": {
                        "type": "string",
                        "enum": ["open", "closed"],
                        "description": "Filter by PR state (optional, omit for all)",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Max number of results to return (default 10)",
                        "default": 10,
                    },
                },
            },
            handler=self._search_github_prs,
        )

        self._register(
            name="get_recent_activity",
            description="Get recent PR activity from the last N hours",
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

        self._register(
            name="get_ci_status",
            description="Get the current CI status (Nvidia + AMD workflows) for a PR",
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
            name="trigger_ci",
            description="Trigger CI for a PR by adding the run-ci label via comment",
            parameters={
                "type": "object",
                "required": ["pr_number"],
                "properties": {
                    "pr_number": {"type": "integer", "description": "PR number"},
                },
            },
            handler=self._trigger_ci,
        )

        self._register(
            name="cancel_auto_merge",
            description="Cancel a pending auto-merge for a PR",
            parameters={
                "type": "object",
                "required": ["pr_number"],
                "properties": {
                    "pr_number": {"type": "integer", "description": "PR number"},
                },
            },
            handler=self._cancel_auto_merge,
            requires_confirmation=True,
        )

        self._register(
            name="merge_pr",
            description=(
                "Merge a pull request on GitHub via squash merge. "
                "Use when a user explicitly asks to merge a PR. "
                "Before calling, verify CI is passed and the PR has approval."
            ),
            parameters={
                "type": "object",
                "required": ["pr_number"],
                "properties": {
                    "pr_number": {"type": "integer", "description": "PR number"},
                    "merge_method": {
                        "type": "string",
                        "enum": ["squash", "merge", "rebase"],
                        "description": "Merge method (default: squash)",
                    },
                },
            },
            handler=self._merge_pr,
            requires_confirmation=True,
        )

        self._register(
            name="get_merge_ready_prs",
            description=(
                "Batch-check all open diffusion PRs (or tracked PRs) and return "
                "only those that are ready to merge (CI passed + approved + mergeable). "
                "Use this when the user asks 'which PRs can be merged' or 'merge所有可以merge的PR'."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "tracked_only": {
                        "type": "boolean",
                        "description": "If true, only check the current user's tracked PRs. Default false (all open diffusion PRs).",
                        "default": False,
                    },
                },
                "required": [],
            },
            handler=self._get_merge_ready_prs,
        )

        self._register(
            name="run_health_check",
            description=(
                "Manually trigger the Diffusion PR health check report. "
                "Posts the same report as the periodic 2-hour check to the CI channel. "
                "Use when the user asks to run/trigger a health check or 跑健康检查."
            ),
            parameters={"type": "object", "properties": {}, "required": []},
            handler=self._run_health_check,
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

    async def _search_github_prs(
        self,
        keywords: list[str],
        state: str | None = None,
        max_results: int = 10,
    ) -> list[dict[str, Any]]:
        try:
            items = await self._gh.search_issues(
                keywords, is_pr=True, state=state, max_results=max_results,
            )
            return [
                {
                    "pr_number": item["number"],
                    "title": item["title"],
                    "author": item.get("user", {}).get("login", "unknown"),
                    "state": item["state"],
                    "labels": [l["name"] for l in item.get("labels", [])],
                    "updated_at": item.get("updated_at", ""),
                    "html_url": item.get("html_url", ""),
                }
                for item in items
            ]
        except Exception as e:
            return [{"error": f"GitHub search failed: {e}"}]

    async def _get_recent_activity(self, since_hours: int = 24) -> dict[str, Any]:
        return await queries.get_recent_activity(self._db.conn, since_hours)

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

    async def _review_pr_code(self, pr_number: int) -> dict[str, Any]:
        diff = await self._gh.get_pull_diff(pr_number)
        if not diff:
            return {"error": f"PR #{pr_number} returned an empty diff"}
        truncated = len(diff) > _DIFF_MAX_CHARS
        if truncated:
            diff = diff[:_DIFF_MAX_CHARS]
        return {"pr_number": pr_number, "diff": diff, "truncated": truncated}

    async def _get_ci_status(self, pr_number: int) -> dict[str, Any]:
        if not self._ci_monitor:
            return {"error": "CI monitor not initialized"}
        try:
            pr_data = await self._gh.get_pull(pr_number)
            head_sha = pr_data["head"]["sha"]
            ci_status = await self._ci_monitor.check_pr_ci(
                pr_number, head_sha, pr_data=pr_data,
            )
            return {
                "pr_number": pr_number,
                "head_sha": head_sha[:8],
                "overall": ci_status.overall.value,
                "has_run_ci_label": ci_status.has_run_ci_label,
                "all_runs_completed": ci_status.all_runs_completed,
                "nvidia_jobs": [
                    {"name": j.job_name, "status": j.status, "conclusion": j.conclusion}
                    for j in ci_status.nvidia_jobs
                ],
                "amd_jobs": [
                    {"name": j.job_name, "status": j.status, "conclusion": j.conclusion}
                    for j in ci_status.amd_jobs
                ],
                "failed_jobs": [
                    {"name": j.job_name, "workflow": j.workflow_name}
                    for j in ci_status.failed_jobs
                ],
            }
        except Exception as e:
            return {"error": f"Failed to get CI status: {e}"}

    async def _trigger_ci(self, pr_number: int) -> dict[str, Any]:
        if not self._ci_monitor:
            return {"error": "CI monitor not initialized"}
        try:
            pr_data = await self._gh.get_pull(pr_number)
            labels = [l["name"].lower() for l in pr_data.get("labels", [])]
            has_label = "run-ci" in labels
            result = await self._ci_monitor.trigger_ci(pr_number, has_label)
            triggered = bool(result["rerun_ids"]) or not result["skipped_runs"]
            return {
                "pr_number": pr_number,
                "triggered": triggered,
                "method": result["method"],
                "rerun_ids": result["rerun_ids"],
                "skipped_runs": result["skipped_runs"],
            }
        except Exception as e:
            return {"error": f"Failed to trigger CI: {e}"}

    async def _cancel_auto_merge(self, pr_number: int) -> dict[str, Any]:
        if not self._auto_merge:
            return {"error": "Auto-merge manager not initialized"}
        cancelled = await self._auto_merge.cancel(pr_number)
        if cancelled:
            return {"pr_number": pr_number, "cancelled": True}
        return {"pr_number": pr_number, "cancelled": False, "reason": "No pending merge found"}

    async def _get_merge_ready_prs(self, tracked_only: bool = False) -> list[dict[str, Any]]:
        """Batch-check PRs and return those ready to merge."""
        # Determine which PRs to check
        if tracked_only:
            from sgldhelper.ai.conversation import _current_user
            user_id = _current_user.get()
            mem = await queries.get_user_memory(self._db.conn, user_id)
            if not mem:
                return []
            pr_numbers = json.loads(mem["tracked_prs"])
        else:
            db_prs = await queries.get_open_prs(self._db.conn)
            pr_numbers = [p["pr_number"] for p in db_prs]

        if not pr_numbers:
            return []

        ready: list[dict[str, Any]] = []
        for pr_num in pr_numbers:
            try:
                pr_data = await self._gh.get_pull(pr_num)
            except Exception:
                continue

            if pr_data["state"] != "open":
                continue

            # Check mergeable
            if not pr_data.get("mergeable"):
                continue

            # Check reviews
            reviews = await self._gh.get_pull_reviews(pr_num)
            has_approval = any(r.get("state") == "APPROVED" for r in reviews)
            if not has_approval:
                continue

            # Check CI
            head_sha = pr_data["head"]["sha"]
            if self._ci_monitor:
                ci_status = await self._ci_monitor.check_pr_ci(
                    pr_num, head_sha, pr_data=pr_data,
                )
                ci_passed = ci_status.overall.value == "passed"
            else:
                ci_passed = False

            if not ci_passed:
                continue

            ready.append({
                "pr_number": pr_num,
                "title": pr_data["title"],
                "author": pr_data["user"]["login"],
                "head_sha": head_sha[:8],
                "html_url": pr_data.get("html_url", ""),
            })

        return ready

    async def _merge_pr(
        self, pr_number: int, merge_method: str = "squash"
    ) -> dict[str, Any]:
        try:
            result = await self._gh.merge_pull(pr_number, merge_method=merge_method)
            # Cancel any pending auto-merge for this PR
            if self._auto_merge:
                await self._auto_merge.cancel(pr_number)
            return {
                "pr_number": pr_number,
                "merged": True,
                "merge_method": merge_method,
                "message": result.get("message", "Successfully merged"),
            }
        except Exception as e:
            error_msg = str(e)
            if "405" in error_msg:
                return {"error": f"PR #{pr_number} is not mergeable (conflicts, checks failing, or not approved)"}
            if "404" in error_msg:
                return {"error": f"PR #{pr_number} not found"}
            return {"error": f"Merge failed: {error_msg}"}

    async def _run_health_check(self) -> dict[str, Any]:
        if not self._health_checker:
            return {"error": "Health checker not initialized"}
        try:
            await self._health_checker.poll()
            return {"status": "ok", "message": "Health check report posted to CI channel"}
        except Exception as e:
            return {"error": f"Health check failed: {e}"}


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
