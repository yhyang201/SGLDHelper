"""PR status tracking and change detection for diffusion PRs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from typing import Any

import structlog

from sgldhelper.config import Settings
from sgldhelper.db.engine import Database
from sgldhelper.db import queries
from sgldhelper.github.client import GitHubClient

log = structlog.get_logger()


class PREvent(str, Enum):
    OPENED = "pr_opened"
    UPDATED = "pr_updated"
    REVIEWED = "pr_reviewed"
    MERGED = "pr_merged"
    CLOSED = "pr_closed"


@dataclass
class PRChange:
    event: PREvent
    pr: dict[str, Any]
    old_state: dict[str, Any] | None = None


class PRTracker:
    """Detect diffusion-related PR changes by polling the GitHub API."""

    def __init__(
        self, client: GitHubClient, db: Database, settings: Settings
    ) -> None:
        self._client = client
        self._db = db
        self._settings = settings

    def is_diffusion_pr(self, pr: dict[str, Any], files: list[dict[str, Any]] | None = None) -> bool:
        """Check if a PR is diffusion-related using multi-signal OR logic.

        Signals (any match → True):
        1. Title contains "diffusion" but NOT "diffusion-llm" / "diffusion llm"
        2. Label named "diffusion"
        3. File paths match configured diffusion_paths prefixes

        Title/label checks are cheap (no API call). File-path check requires
        the *files* parameter; if ``None`` is passed, only title/label are used.
        """
        # Signal 1: title
        title = pr.get("title", "").lower()
        if "diffusion" in title and "diffusion-llm" not in title and "diffusion llm" not in title:
            return True

        # Signal 2: labels
        labels = [l["name"].lower() for l in pr.get("labels", [])]
        if "diffusion" in labels:
            return True

        # Signal 3: file paths (most precise, requires files)
        if files is not None:
            return any(
                f.get("filename", "").startswith(tuple(self._settings.diffusion_paths))
                for f in files
            )

        return False

    async def poll(self) -> list[PRChange]:
        """Poll for PR changes and return detected events.

        Uses a classification cache to avoid calling ``get_pull_files`` for
        every open PR on each poll cycle.  Only new PRs or PRs whose
        ``head_sha`` changed since the last poll trigger a files lookup.

        On cold start (empty classification cache), fetches up to
        ``cold_start_max_prs`` PRs to seed the cache.
        """
        changes: list[PRChange] = []
        cache = await queries.get_pr_classifications(self._db.conn)

        # Cold start: fetch more PRs when cache is empty
        if not cache:
            log.info("pr_tracker.cold_start", max_prs=self._settings.cold_start_max_prs)
            pulls = await self._client.get_open_pulls_all(self._settings.cold_start_max_prs)
        else:
            pulls = await self._client.get_open_pulls()

        for pr_data in pulls:
            pr_num = pr_data["number"]
            sha = pr_data["head"]["sha"]
            cached = cache.get(pr_num)

            if cached:
                cached_sha, was_diffusion = cached
                if cached_sha == sha:
                    # Classification is still valid for this SHA.
                    if not was_diffusion:
                        continue  # known non-diffusion — skip
                    # Known diffusion — proceed to upsert & change detection
                    pr_record = self._normalize_pr(pr_data)
                    old = await queries.upsert_pr(self._db.conn, pr_record)
                    if old is None:
                        changes.append(PRChange(event=PREvent.OPENED, pr=pr_record))
                        log.info("pr.opened", pr=pr_record["pr_number"], title=pr_record["title"])
                    else:
                        detected = self._detect_changes(old, pr_record, pr_data)
                        changes.extend(detected)
                    continue

            # New PR or SHA changed — try title/label first (cheap), then files.
            is_diff = self.is_diffusion_pr(pr_data)  # title/label only
            if not is_diff:
                # Title/label didn't match — fetch files for precise check.
                try:
                    files = await self._client.get_pull_files(pr_num)
                except Exception:
                    files = []
                is_diff = self.is_diffusion_pr(pr_data, files)
            await queries.upsert_pr_classification(self._db.conn, pr_num, sha, is_diff)

            if not is_diff:
                continue

            pr_record = self._normalize_pr(pr_data)
            old = await queries.upsert_pr(self._db.conn, pr_record)

            if old is None:
                changes.append(PRChange(event=PREvent.OPENED, pr=pr_record))
                log.info("pr.opened", pr=pr_record["pr_number"], title=pr_record["title"])
            else:
                detected = self._detect_changes(old, pr_record, pr_data)
                changes.extend(detected)

        # Check for PRs that were merged/closed since last poll
        closed_changes = await self._check_closed_prs()
        changes.extend(closed_changes)

        return changes

    async def _check_closed_prs(self) -> list[PRChange]:
        """Detect PRs that transitioned to merged/closed state."""
        changes: list[PRChange] = []
        open_prs = await queries.get_open_prs(self._db.conn)

        for stored_pr in open_prs:
            try:
                current = await self._client.get_pull(stored_pr["pr_number"])
            except Exception:
                continue

            if current["state"] == "closed":
                pr_record = self._normalize_pr(current)
                old = await queries.upsert_pr(self._db.conn, pr_record)

                if current.get("merged"):
                    changes.append(
                        PRChange(event=PREvent.MERGED, pr=pr_record, old_state=old)
                    )
                    log.info("pr.merged", pr=pr_record["pr_number"])
                else:
                    changes.append(
                        PRChange(event=PREvent.CLOSED, pr=pr_record, old_state=old)
                    )
                    log.info("pr.closed", pr=pr_record["pr_number"])

        return changes

    def _detect_changes(
        self,
        old: dict[str, Any],
        new: dict[str, Any],
        raw_pr: dict[str, Any],
    ) -> list[PRChange]:
        """Compare old and new PR state to detect change events."""
        changes: list[PRChange] = []

        # New commits pushed
        if old["head_sha"] != new["head_sha"]:
            changes.append(PRChange(event=PREvent.UPDATED, pr=new, old_state=old))
            log.info(
                "pr.updated",
                pr=new["pr_number"],
                old_sha=old["head_sha"][:8],
                new_sha=new["head_sha"][:8],
            )

        return changes

    def _normalize_pr(self, pr_data: dict[str, Any]) -> dict[str, Any]:
        """Convert GitHub API PR response to our DB record format."""
        merged = pr_data.get("merged", False) or pr_data.get("merged_at") is not None
        state = "merged" if merged else pr_data["state"]
        return {
            "pr_number": pr_data["number"],
            "title": pr_data["title"],
            "author": pr_data["user"]["login"],
            "state": state,
            "head_sha": pr_data["head"]["sha"],
            "updated_at": pr_data["updated_at"],
            "changed_files": pr_data.get("changed_files", 0),
            "labels": [l["name"] for l in pr_data.get("labels", [])],
        }
