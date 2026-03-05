"""Periodic summary generation for tracked PRs (every 12h)."""

from __future__ import annotations

import json
from typing import Any, Callable, Awaitable

import structlog

from sgldhelper.ai.client import KimiClient
from sgldhelper.config import Settings
from sgldhelper.db import queries
from sgldhelper.db.engine import Database
from sgldhelper.github.client import GitHubClient

log = structlog.get_logger()

_TRACKED_PR_SUMMARY_PROMPT = (
    "You are a PR status summary generator. Compare the previous snapshot with current data "
    "and generate a concise update for each tracked PR that has changed.\n\n"
    "IMPORTANT — you are outputting to Slack mrkdwn, NOT standard Markdown:\n"
    "- Bold: *text* (single asterisk, NOT double **)\n"
    "- Bullet list: use • or - at line start\n"
    "- Links: <https://url|display text>\n"
    "- DO NOT use Markdown tables or headings\n\n"
    "Focus on:\n"
    "- New commits since last check\n"
    "- CI status changes\n"
    "- Review state changes\n"
    "Keep each PR summary to 2-3 lines.\n"
    "Respond in the same language as the data.\n"
)


class TrackedPRSummaryGenerator:
    """Generate periodic summaries for tracked PRs by comparing snapshots."""

    def __init__(
        self,
        kimi: KimiClient,
        gh: GitHubClient,
        db: Database,
        settings: Settings,
    ) -> None:
        self._kimi = kimi
        self._gh = gh
        self._db = db
        self._settings = settings
        self._on_summary: Callable[..., Awaitable[Any]] | None = None

    def set_callback(self, on_summary: Callable[..., Awaitable[Any]]) -> None:
        self._on_summary = on_summary

    async def poll(self) -> None:
        """Check all tracked PRs and generate summaries for those with changes."""
        tracked = await queries.get_all_tracked_prs(self._db.conn)
        if not tracked:
            return

        for pr_number, user_ids in tracked.items():
            try:
                await self._check_and_summarize(pr_number, user_ids)
            except Exception:
                log.exception("tracked_pr_summary.error", pr=pr_number)

    async def _check_and_summarize(
        self, pr_number: int, user_ids: list[str]
    ) -> None:
        # Get current state
        try:
            pr_data = await self._gh.get_pull(pr_number)
        except Exception:
            return

        if pr_data["state"] != "open":
            return

        head_sha = pr_data["head"]["sha"]

        # Get current CI snapshot
        current_snapshot = await queries.get_ci_snapshot(self._db.conn, pr_number, head_sha)

        # Get last summary
        last_summary = await queries.get_last_tracked_pr_summary(self._db.conn, pr_number)

        # Get previous snapshot for comparison
        prev_snapshot = await queries.get_latest_ci_snapshot(self._db.conn, pr_number)

        # Build current state description
        try:
            commits = await self._gh.get_pull_commits(pr_number)
            commit_count = len(commits)
        except Exception:
            commit_count = 0

        reviews = await self._gh.get_pull_reviews(pr_number)
        review_state = "none"
        for r in reviews:
            if r.get("state") == "APPROVED":
                review_state = "approved"
                break
            elif r.get("state") == "CHANGES_REQUESTED":
                review_state = "changes_requested"

        ci_status = current_snapshot["overall_status"] if current_snapshot else "unknown"

        # Check if anything changed
        if prev_snapshot and last_summary:
            prev_ci = prev_snapshot.get("overall_status", "unknown")
            prev_review = prev_snapshot.get("review_state", "none")
            prev_commits = prev_snapshot.get("commit_count", 0)

            if (prev_ci == ci_status and prev_review == review_state
                    and prev_commits == commit_count):
                log.debug("tracked_pr_summary.no_changes", pr=pr_number)
                return

        # Build context for LLM
        context = (
            f"PR #{pr_number}: {pr_data['title']}\n"
            f"Author: {pr_data['user']['login']}\n"
            f"Current state: {pr_data['state']}\n"
            f"Head SHA: {head_sha[:8]}\n"
            f"Commits: {commit_count}\n"
            f"CI status: {ci_status}\n"
            f"Review state: {review_state}\n"
        )

        if prev_snapshot:
            context += (
                f"\nPrevious snapshot:\n"
                f"  CI: {prev_snapshot.get('overall_status', 'unknown')}\n"
                f"  Review: {prev_snapshot.get('review_state', 'none')}\n"
                f"  Commits: {prev_snapshot.get('commit_count', 0)}\n"
                f"  Failed jobs: {prev_snapshot.get('failed_jobs', '[]')}\n"
            )

        messages = [
            {"role": "system", "content": _TRACKED_PR_SUMMARY_PROMPT},
            {"role": "user", "content": f"Generate a status update for this tracked PR:\n\n{context}"},
        ]

        response = await self._kimi.chat(messages, thinking=False)
        tokens_in, tokens_out = self._kimi.extract_usage(response)

        await queries.log_llm_usage(
            self._db.conn, "tracked_pr_summary", self._settings.kimi_model,
            tokens_in, tokens_out,
        )

        summary_text = response.choices[0].message.content
        if not summary_text:
            return

        # Save summary
        await queries.save_tracked_pr_summary(self._db.conn, pr_number, summary_text, user_ids)

        # Notify
        if self._on_summary:
            await self._on_summary(pr_number, user_ids, summary_text)

        log.info("tracked_pr_summary.generated", pr=pr_number)
