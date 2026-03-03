"""Feature roadmap tracking via GitHub issue checkboxes."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any

import structlog

from sgldhelper.config import Settings
from sgldhelper.db.engine import Database
from sgldhelper.db import queries
from sgldhelper.github.client import GitHubClient

log = structlog.get_logger()

# Matches GitHub checkbox items: - [x] or - [ ] followed by text
CHECKBOX_RE = re.compile(
    r"^[-*]\s+\[([ xX])\]\s+(.+?)(?:\s+#(\d+))?$", re.MULTILINE
)


@dataclass
class FeatureProgress:
    issue_number: int
    title: str
    total: int
    completed: int
    items: list[dict[str, Any]]

    @property
    def percent(self) -> float:
        return (self.completed / self.total * 100) if self.total > 0 else 0


class IssueTracker:
    """Parse roadmap issues and track feature checkbox progress."""

    def __init__(
        self, client: GitHubClient, db: Database, settings: Settings
    ) -> None:
        self._client = client
        self._db = db
        self._settings = settings

    async def poll(self) -> list[FeatureProgress]:
        """Poll all configured roadmap issues and return progress summaries."""
        results: list[FeatureProgress] = []

        for issue_num in self._settings.roadmap_issue_numbers:
            try:
                progress = await self._track_issue(issue_num)
                results.append(progress)
            except Exception as e:
                log.error("feature.track_failed", issue=issue_num, error=str(e))

        return results

    async def _track_issue(self, issue_number: int) -> FeatureProgress:
        """Parse a single issue's body for checkbox items and track them."""
        issue = await self._client.get_issue(issue_number)
        body = issue.get("body", "") or ""
        title = issue.get("title", f"Issue #{issue_number}")

        items = self._parse_checkboxes(body, issue_number)

        for item in items:
            old = await queries.upsert_feature_item(self._db.conn, item)
            if old and old["state"] != item["state"]:
                log.info(
                    "feature.state_changed",
                    item=item["title"][:50],
                    old_state=old["state"],
                    new_state=item["state"],
                )

        completed = sum(1 for i in items if i["state"] == "completed")
        return FeatureProgress(
            issue_number=issue_number,
            title=title,
            total=len(items),
            completed=completed,
            items=items,
        )

    def _parse_checkboxes(
        self, body: str, issue_number: int
    ) -> list[dict[str, Any]]:
        """Extract checkbox items from issue body markdown."""
        items: list[dict[str, Any]] = []
        for match in CHECKBOX_RE.finditer(body):
            checked = match.group(1).lower() == "x"
            text = match.group(2).strip()
            linked_pr_str = match.group(3)
            linked_pr = int(linked_pr_str) if linked_pr_str else None

            # Generate stable item ID from content
            item_id = hashlib.md5(
                f"{issue_number}:{text}".encode()
            ).hexdigest()[:12]

            items.append({
                "item_id": item_id,
                "parent_issue": issue_number,
                "title": text,
                "item_type": "checkbox",
                "state": "completed" if checked else "open",
                "linked_pr": linked_pr,
                "completed_at": None,  # Could be enriched from PR merge date
            })

        return items

    async def get_progress(self, issue_number: int) -> FeatureProgress:
        """Get cached progress for an issue from DB."""
        issue = await self._client.get_issue(issue_number)
        items = await queries.get_feature_items(self._db.conn, issue_number)
        completed = sum(1 for i in items if i["state"] == "completed")
        return FeatureProgress(
            issue_number=issue_number,
            title=issue.get("title", f"Issue #{issue_number}"),
            total=len(items),
            completed=completed,
            items=items,
        )
