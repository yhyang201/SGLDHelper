"""Stall detection and review nudge logic (pure SQL + GitHub API, no LLM)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import structlog

from sgldhelper.config import Settings
from sgldhelper.db import queries
from sgldhelper.db.engine import Database
from sgldhelper.github.client import GitHubClient

log = structlog.get_logger()


@dataclass
class StallAlert:
    alert_type: str  # "feature_stall" or "review_nudge"
    ref_id: str  # item_id or pr_number
    title: str
    pr_number: int | None
    author: str
    days_stalled: int
    details: str


class StallDetector:
    """Detect stalled features and PRs needing review.

    Runs on a timer (every 12h by default). Pure DB + GitHub queries, no LLM.
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

    async def check_all(self) -> list[StallAlert]:
        """Run all stall checks. Returns new alerts (deduped)."""
        alerts: list[StallAlert] = []
        alerts.extend(await self.check_stalled_features())
        alerts.extend(await self.check_reviews_needed())
        return alerts

    async def check_stalled_features(self) -> list[StallAlert]:
        """Find open feature items with stalled linked PRs."""
        stalled = await queries.get_stalled_features(
            self._db.conn, self._settings.stall_days_threshold
        )
        alerts: list[StallAlert] = []

        for item in stalled:
            days = self._settings.stall_days_threshold
            is_new = await queries.record_stall_alert(
                self._db.conn,
                alert_type="feature_stall",
                ref_id=item["item_id"],
                days_stalled=days,
            )
            if not is_new:
                continue

            alert = StallAlert(
                alert_type="feature_stall",
                ref_id=item["item_id"],
                title=item["title"],
                pr_number=item.get("linked_pr"),
                author=item.get("pr_author", "unknown"),
                days_stalled=days,
                details=(
                    f"Feature item *{item['title']}* has linked PR "
                    f"#{item.get('linked_pr')} (by `{item.get('pr_author', '?')}`) "
                    f"that hasn't been updated in {days}+ days."
                ),
            )
            alerts.append(alert)
            log.info(
                "stall.feature_detected",
                item_id=item["item_id"],
                pr=item.get("linked_pr"),
                days=days,
            )

        return alerts

    async def check_reviews_needed(self) -> list[StallAlert]:
        """Find open PRs that haven't received review approval."""
        prs = await queries.get_prs_needing_review(
            self._db.conn, self._settings.review_nudge_days
        )
        alerts: list[StallAlert] = []

        for pr in prs:
            pr_number = pr["pr_number"]

            # Check if PR actually has an approved review via GitHub API
            try:
                reviews = await self._gh.get_pull_reviews(pr_number)
                has_approval = any(
                    r.get("state") == "APPROVED" for r in reviews
                )
                if has_approval:
                    continue
            except Exception as e:
                log.warning("stall.review_check_failed", pr=pr_number, error=str(e))
                continue

            days = self._settings.review_nudge_days
            is_new = await queries.record_stall_alert(
                self._db.conn,
                alert_type="review_nudge",
                ref_id=str(pr_number),
                days_stalled=days,
            )
            if not is_new:
                continue

            alert = StallAlert(
                alert_type="review_nudge",
                ref_id=str(pr_number),
                title=pr["title"],
                pr_number=pr_number,
                author=pr["author"],
                days_stalled=days,
                details=(
                    f"PR #{pr_number} *{pr['title']}* by `{pr['author']}` "
                    f"has been open for {days}+ days without review approval."
                ),
            )
            alerts.append(alert)
            log.info("stall.review_nudge", pr=pr_number, days=days)

        return alerts
