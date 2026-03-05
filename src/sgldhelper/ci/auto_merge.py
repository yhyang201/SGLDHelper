"""Auto-merge manager for tracked PRs that pass CI."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Callable, Awaitable

import structlog

from sgldhelper.config import Settings
from sgldhelper.db import queries
from sgldhelper.db.engine import Database
from sgldhelper.github.client import GitHubClient

log = structlog.get_logger()


@dataclass
class PendingMerge:
    pr_number: int
    user_ids: list[str]
    task: asyncio.Task[None] | None = None
    cancel_event: asyncio.Event = field(default_factory=asyncio.Event)


class AutoMergeManager:
    """Manage auto-merge with countdown and cancellation for tracked PRs."""

    def __init__(
        self,
        gh: GitHubClient,
        db: Database,
        settings: Settings,
    ) -> None:
        self._gh = gh
        self._db = db
        self._settings = settings
        self._pending: dict[int, PendingMerge] = {}
        # Callbacks for notifications
        self._on_countdown: Callable[..., Awaitable[Any]] | None = None
        self._on_complete: Callable[..., Awaitable[Any]] | None = None
        self._on_cancelled: Callable[..., Awaitable[Any]] | None = None

    def set_callbacks(
        self,
        *,
        on_countdown: Callable[..., Awaitable[Any]] | None = None,
        on_complete: Callable[..., Awaitable[Any]] | None = None,
        on_cancelled: Callable[..., Awaitable[Any]] | None = None,
    ) -> None:
        self._on_countdown = on_countdown
        self._on_complete = on_complete
        self._on_cancelled = on_cancelled

    async def check_and_start(
        self,
        pr_number: int,
        user_ids: list[str],
        review_state: str,
    ) -> bool:
        """Check if a PR is eligible for auto-merge and start countdown if so.

        Returns True if a merge countdown was started.
        """
        if not self._settings.auto_merge_enabled:
            return False

        if pr_number in self._pending:
            return False  # already pending

        # Check conditions
        if review_state != "approved":
            log.debug("auto_merge.skip_no_approval", pr=pr_number)
            return False

        try:
            pr_data = await self._gh.get_pull(pr_number)
        except Exception:
            log.warning("auto_merge.pr_fetch_failed", pr=pr_number)
            return False

        if not pr_data.get("mergeable"):
            log.debug("auto_merge.skip_not_mergeable", pr=pr_number)
            return False

        # Check if it's a diffusion PR (stored in our DB)
        db_pr = await queries.get_pr(self._db.conn, pr_number)
        if not db_pr:
            log.debug("auto_merge.skip_not_in_db", pr=pr_number)
            return False

        # Start countdown
        pending = PendingMerge(pr_number=pr_number, user_ids=user_ids)
        self._pending[pr_number] = pending
        pending.task = asyncio.create_task(self._merge_countdown(pending))

        log.info("auto_merge.countdown_started", pr=pr_number,
                 delay=self._settings.auto_merge_delay_seconds)
        return True

    async def cancel(self, pr_number: int) -> bool:
        """Cancel a pending auto-merge. Returns True if one was cancelled."""
        pending = self._pending.pop(pr_number, None)
        if not pending:
            return False

        pending.cancel_event.set()
        if pending.task and not pending.task.done():
            pending.task.cancel()

        if self._on_cancelled:
            await self._on_cancelled(pr_number, pending.user_ids)

        log.info("auto_merge.cancelled", pr=pr_number)
        return True

    def is_pending(self, pr_number: int) -> bool:
        return pr_number in self._pending

    def get_pending_prs(self) -> list[int]:
        return list(self._pending.keys())

    async def _merge_countdown(self, pending: PendingMerge) -> None:
        """Wait for delay, then merge if conditions still hold."""
        pr_number = pending.pr_number
        delay = self._settings.auto_merge_delay_seconds

        # Notify countdown start
        if self._on_countdown:
            await self._on_countdown(pr_number, pending.user_ids, delay)

        # Wait for delay or cancellation
        try:
            await asyncio.wait_for(
                pending.cancel_event.wait(),
                timeout=delay,
            )
            # cancel_event was set — merge cancelled
            return
        except asyncio.TimeoutError:
            pass  # timeout = delay elapsed, proceed to merge
        except asyncio.CancelledError:
            return

        # Re-check conditions before merging
        try:
            pr_data = await self._gh.get_pull(pr_number)
            if pr_data["state"] != "open":
                log.info("auto_merge.skip_not_open", pr=pr_number)
                self._pending.pop(pr_number, None)
                return

            if not pr_data.get("mergeable"):
                log.info("auto_merge.skip_not_mergeable_at_merge", pr=pr_number)
                self._pending.pop(pr_number, None)
                return

            # Perform the merge
            await self._gh.merge_pull(pr_number, merge_method="squash")
            log.info("auto_merge.merged", pr=pr_number)

            if self._on_complete:
                await self._on_complete(pr_number, pending.user_ids)

        except Exception:
            log.exception("auto_merge.merge_failed", pr=pr_number)
        finally:
            self._pending.pop(pr_number, None)

    def check_cancel_keywords(self, text: str) -> int | None:
        """Check if text contains cancel keywords. Returns PR number if matched.

        For now, checks all pending PRs. If text contains a cancel keyword,
        returns the first pending PR number.
        """
        text_lower = text.lower()
        for keyword in self._settings.auto_merge_cancel_keywords:
            if keyword.lower() in text_lower:
                # Try to find PR number in the text
                import re
                match = re.search(r"#?(\d{4,6})", text)
                if match:
                    pr_num = int(match.group(1))
                    if pr_num in self._pending:
                        return pr_num
                # If no PR number mentioned, cancel the first (or only) pending
                if self._pending:
                    return next(iter(self._pending))
                return None
        return None
