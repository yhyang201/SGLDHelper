"""CI status monitoring and automatic retry for tracked PRs."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import structlog

from sgldhelper.config import Settings
from sgldhelper.db import queries
from sgldhelper.db.engine import Database
from sgldhelper.github.client import GitHubClient

log = structlog.get_logger()


class CIOverallStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    NO_CI = "no_ci"


@dataclass
class CIJobResult:
    job_name: str
    workflow_name: str
    status: str  # queued, in_progress, completed
    conclusion: str | None  # success, failure, cancelled, skipped, None
    run_id: int
    job_id: int


@dataclass
class CIStatus:
    pr_number: int
    head_sha: str
    overall: CIOverallStatus
    has_run_ci_label: bool
    nvidia_jobs: list[CIJobResult] = field(default_factory=list)
    amd_jobs: list[CIJobResult] = field(default_factory=list)
    failed_jobs: list[CIJobResult] = field(default_factory=list)
    all_runs_completed: bool = False
    has_high_priority_label: bool = False


class CIMonitor:
    """Monitor CI status for tracked PRs and handle automatic retries."""

    def __init__(
        self,
        gh: GitHubClient,
        db: Database,
        settings: Settings,
    ) -> None:
        self._gh = gh
        self._db = db
        self._settings = settings
        self._trigger_lock = asyncio.Lock()
        # Callbacks set by __main__.py
        self._on_ci_passed: Any = None
        self._on_ci_failed_retrying: Any = None
        self._on_ci_failed_permanent: Any = None
        self._on_ci_passed_approved: Any = None
        self._is_merge_pending: Any = None

    def set_callbacks(
        self,
        *,
        on_ci_passed: Any = None,
        on_ci_failed_retrying: Any = None,
        on_ci_failed_permanent: Any = None,
        on_merge_ready_check: Any = None,
        on_ci_passed_approved: Any = None,
        is_merge_pending: Any = None,
    ) -> None:
        self._on_ci_passed = on_ci_passed
        self._on_ci_failed_retrying = on_ci_failed_retrying
        self._on_ci_failed_permanent = on_ci_failed_permanent
        self._on_merge_ready_check = on_merge_ready_check
        self._on_ci_passed_approved = on_ci_passed_approved
        self._is_merge_pending = is_merge_pending

    async def check_pr_ci(
        self,
        pr_number: int,
        head_sha: str,
        *,
        pr_data: dict[str, Any] | None = None,
    ) -> CIStatus:
        """Check CI status for a PR at a specific SHA.

        Pass *pr_data* to reuse an already-fetched PR payload and avoid
        a redundant ``get_pull`` call.
        """
        if pr_data is None:
            pr_data = await self._gh.get_pull(pr_number)
        labels = [l["name"].lower() for l in pr_data.get("labels", [])]
        has_run_ci = "run-ci" in labels
        has_high_priority = self._settings.ci_high_priority_label.lower() in labels

        # Get workflow runs for this SHA
        runs = await self._gh.get_workflow_runs_for_ref(head_sha)

        nvidia_wf = self._settings.ci_nvidia_workflow_id
        amd_wf = self._settings.ci_amd_workflow_id

        nvidia_jobs: list[CIJobResult] = []
        amd_jobs: list[CIJobResult] = []
        all_completed = True
        found_any_run = False

        for run in runs:
            wf_id = run.get("workflow_id")
            if wf_id not in (nvidia_wf, amd_wf):
                continue

            found_any_run = True
            wf_name = "nvidia" if wf_id == nvidia_wf else "amd"

            if run["status"] != "completed":
                all_completed = False

            # Get jobs for this run
            jobs = await self._gh.get_workflow_run_jobs(run["id"])
            for job in jobs:
                # Skip skipped jobs
                if job.get("conclusion") == "skipped":
                    continue

                result = CIJobResult(
                    job_name=job["name"],
                    workflow_name=wf_name,
                    status=job["status"],
                    conclusion=job.get("conclusion"),
                    run_id=run["id"],
                    job_id=job["id"],
                )

                if wf_id == nvidia_wf:
                    nvidia_jobs.append(result)
                else:
                    amd_jobs.append(result)

        if not found_any_run:
            return CIStatus(
                pr_number=pr_number,
                head_sha=head_sha,
                overall=CIOverallStatus.NO_CI,
                has_run_ci_label=has_run_ci,
                all_runs_completed=True,
                has_high_priority_label=has_high_priority,
            )

        # Determine overall status
        all_jobs = nvidia_jobs + amd_jobs
        failed = [j for j in all_jobs if j.conclusion == "failure"]

        if not all_completed:
            has_failure = bool(failed)
            overall = CIOverallStatus.RUNNING if not has_failure else CIOverallStatus.RUNNING
        elif failed:
            overall = CIOverallStatus.FAILED
        elif all(j.conclusion == "success" for j in all_jobs if j.status == "completed"):
            overall = CIOverallStatus.PASSED
        else:
            overall = CIOverallStatus.PENDING

        return CIStatus(
            pr_number=pr_number,
            head_sha=head_sha,
            overall=overall,
            has_run_ci_label=has_run_ci,
            nvidia_jobs=nvidia_jobs,
            amd_jobs=amd_jobs,
            failed_jobs=failed,
            all_runs_completed=all_completed,
            has_high_priority_label=has_high_priority,
        )

    def _all_ci_passed(self, ci_status: CIStatus) -> bool:
        """Check if both Nvidia AND AMD CI jobs passed."""
        if not ci_status.nvidia_jobs or not ci_status.amd_jobs:
            return False
        return all(
            j.status == "completed" and j.conclusion == "success"
            for j in ci_status.nvidia_jobs + ci_status.amd_jobs
        )

    async def should_retry(
        self,
        pr_number: int,
        head_sha: str,
        job: CIJobResult,
        *,
        is_high_priority: bool = False,
    ) -> bool:
        """Check if a failed job should be retried (under max retries)."""
        count = await queries.get_ci_retry_count(
            self._db.conn, pr_number, head_sha, job.job_name
        )
        max_retries = (
            self._settings.ci_high_priority_max_retries
            if is_high_priority
            else self._settings.ci_max_retries
        )
        return count < max_retries

    async def trigger_ci(self, pr_number: int, has_label: bool) -> dict[str, Any]:
        """Trigger CI via PR comment. Uses lock to prevent concurrent triggers.

        Returns a dict describing what actually happened:
        - method: "comment" (always comments now)
        - rerun_ids: list of run IDs that had failures (for which /rerun-failed-ci was posted)
        - skipped_runs: list of {run_id, status, conclusion} for runs that were
          skipped (matched workflow but didn't meet rerun criteria)
        """
        async with self._trigger_lock:
            if not has_label:
                await self._gh.create_issue_comment(pr_number, "/tag-and-rerun-ci")
                log.info("ci.triggered_via_comment", pr=pr_number)
                return {"method": "comment", "rerun_ids": [], "skipped_runs": []}

            # If label exists, find runs and rerun failed
            pr_data = await self._gh.get_pull(pr_number)
            sha = pr_data["head"]["sha"]
            runs = await self._gh.get_workflow_runs_for_ref(sha)

            rerun_ids: list[int] = []
            skipped_runs: list[dict[str, Any]] = []

            has_failed = False
            for run in runs:
                wf_id = run.get("workflow_id")
                if wf_id not in (self._settings.ci_nvidia_workflow_id, self._settings.ci_amd_workflow_id):
                    continue

                run_id = run["id"]
                status = run["status"]
                conclusion = run.get("conclusion")

                if status == "completed" and conclusion in ("failure", "cancelled", "timed_out"):
                    has_failed = True
                    rerun_ids.append(run_id)
                else:
                    skipped_runs.append({
                        "run_id": run_id,
                        "status": status,
                        "conclusion": conclusion,
                        "workflow_id": wf_id,
                    })
                    log.info(
                        "ci.rerun_skipped",
                        pr=pr_number,
                        run_id=run_id,
                        status=status,
                        conclusion=conclusion,
                    )

            if has_failed:
                await self._gh.create_issue_comment(pr_number, "/rerun-failed-ci")
                log.info("ci.rerun_via_comment", pr=pr_number, run_ids=rerun_ids)
            elif not skipped_runs:
                log.warning(
                    "ci.no_matching_runs",
                    pr=pr_number,
                    sha=sha,
                    total_runs=len(runs),
                )

            return {"method": "comment", "rerun_ids": rerun_ids, "skipped_runs": skipped_runs}

    async def poll_all_tracked_prs(self) -> None:
        """Poll CI for all tracked PRs. Called by the CI poller."""
        tracked = await queries.get_all_tracked_prs(self._db.conn)
        if not tracked:
            return

        for pr_number, user_ids in tracked.items():
            try:
                await self._poll_single_pr(pr_number, user_ids)
            except Exception:
                log.exception("ci_monitor.poll_error", pr=pr_number)

    async def poll_all_open_prs(self) -> None:
        """Check all open diffusion PRs (not tracked) for:

        1. All CI passed + approved → ping
        2. Owner-rerun: mickqian commented /tag-and-rerun-ci → auto-retry on failure

        Tracked PRs are already covered by ``_poll_single_pr``.
        """
        tracked = await queries.get_all_tracked_prs(self._db.conn)
        tracked_set = set(tracked.keys()) if tracked else set()

        open_prs = await queries.get_open_prs(self._db.conn)
        for db_pr in open_prs:
            pr_number = db_pr["pr_number"]
            if pr_number in tracked_set:
                continue
            try:
                await self._check_untracked_pr(pr_number)
            except Exception:
                log.exception("ci_monitor.untracked_pr_error", pr=pr_number)

    async def _check_untracked_pr(self, pr_number: int) -> None:
        """Check an untracked PR for CI-passed ping and owner-rerun retry."""
        try:
            pr_data = await self._gh.get_pull(pr_number)
        except Exception:
            return

        if pr_data["state"] != "open":
            return

        head_sha = pr_data["head"]["sha"]
        ci_status = await self.check_pr_ci(pr_number, head_sha, pr_data=pr_data)

        # --- All CI passed + approved → ping ---
        if self._on_ci_passed_approved:
            await self._maybe_ci_ping(pr_number, head_sha, ci_status)

        # --- Approve auto-CI: start or retry CI when approved by configured users ---
        await self._maybe_approve_auto_ci(pr_number, head_sha, ci_status)

        # --- Owner rerun: auto-retry on failure ---
        if (
            ci_status.overall == CIOverallStatus.FAILED
            and ci_status.all_runs_completed
            and ci_status.has_run_ci_label
        ):
            await self._maybe_owner_rerun(pr_number, head_sha, ci_status)

    async def _maybe_ci_ping(
        self, pr_number: int, head_sha: str, ci_status: CIStatus
    ) -> None:
        """Ping once when all CI passed + approved (untracked PRs)."""
        if not self._all_ci_passed(ci_status):
            return

        reviews = await self._gh.get_pull_reviews(pr_number)
        review_state = "none"
        for r in reviews:
            if r.get("state") == "APPROVED":
                review_state = "approved"
                break

        if review_state != "approved":
            return

        if self._is_merge_pending and self._is_merge_pending(pr_number):
            return

        prev_snapshot = await queries.get_ci_snapshot(self._db.conn, pr_number, head_sha)
        snapshot_data = self._load_snapshot_data(prev_snapshot)

        if snapshot_data.get("hp_nvidia_pinged"):
            return

        await self._on_ci_passed_approved(pr_number, [], review_state)
        snapshot_data["hp_nvidia_pinged"] = True
        failed_names = [j.job_name for j in ci_status.failed_jobs]
        await queries.upsert_ci_snapshot(
            self._db.conn,
            pr_number,
            head_sha,
            overall_status=ci_status.overall.value,
            has_run_ci_label=ci_status.has_run_ci_label,
            failed_jobs=json.dumps(failed_names),
            review_state=review_state,
            snapshot_data=json.dumps(snapshot_data),
        )

    async def _maybe_owner_rerun(
        self, pr_number: int, head_sha: str, ci_status: CIStatus
    ) -> None:
        """Auto-retry failed CI if the owner (ci_high_priority_ping_user) has
        commented /tag-and-rerun-ci on this PR. Max ci_owner_rerun_max_retries."""
        owner = self._settings.ci_high_priority_ping_user

        # Check snapshot_data cache first to avoid re-fetching comments
        prev_snapshot = await queries.get_ci_snapshot(self._db.conn, pr_number, head_sha)
        snapshot_data = self._load_snapshot_data(prev_snapshot)

        # Cache whether owner commented /tag-and-rerun-ci (per SHA)
        if "owner_rerun" not in snapshot_data:
            comments = await self._gh.get_issue_comments(pr_number)
            has_owner_rerun = any(
                c.get("user", {}).get("login") == owner
                and "/tag-and-rerun-ci" in (c.get("body") or "")
                for c in comments
            )
            snapshot_data["owner_rerun"] = has_owner_rerun
            await queries.upsert_ci_snapshot(
                self._db.conn,
                pr_number,
                head_sha,
                overall_status=ci_status.overall.value,
                has_run_ci_label=ci_status.has_run_ci_label,
                failed_jobs=json.dumps([j.job_name for j in ci_status.failed_jobs]),
                snapshot_data=json.dumps(snapshot_data),
            )

        if not snapshot_data["owner_rerun"]:
            return

        # Retry failed jobs up to ci_owner_rerun_max_retries
        max_retries = self._settings.ci_owner_rerun_max_retries
        should_rerun = False
        for job in ci_status.failed_jobs:
            count = await queries.get_ci_retry_count(
                self._db.conn, pr_number, head_sha, job.job_name
            )
            if count < max_retries:
                should_rerun = True
                await queries.increment_ci_retry(
                    self._db.conn, pr_number, head_sha, job.job_name
                )

        if should_rerun:
            try:
                await self._gh.create_issue_comment(pr_number, "/rerun-failed-ci")
            except Exception:
                log.exception("ci_monitor.owner_rerun_comment_failed", pr=pr_number)
            log.info("ci_monitor.owner_rerun_triggered",
                     pr=pr_number, owner=owner)

    async def _maybe_approve_auto_ci(
        self, pr_number: int, head_sha: str, ci_status: CIStatus,
    ) -> None:
        """If an approved-by user (mickqian/bbuf) approved this PR:
        - No run-ci label → comment /tag-and-rerun-ci to start CI
        - CI failed → comment /rerun-failed-ci (up to ci_approve_auto_ci_max_retries)
        """
        # Check if any configured user approved
        reviews = await self._gh.get_pull_reviews(pr_number)
        approved_users = {r["user"]["login"] for r in reviews if r.get("state") == "APPROVED"}
        auto_ci_users = set(self._settings.ci_approve_auto_ci_users)
        if not approved_users & auto_ci_users:
            return

        # Case 1: No run-ci label → start CI
        if not ci_status.has_run_ci_label:
            snapshot = await queries.get_ci_snapshot(self._db.conn, pr_number, head_sha)
            snapshot_data = self._load_snapshot_data(snapshot)
            if snapshot_data.get("approve_auto_ci_started"):
                return  # Already commented once for this SHA
            try:
                await self._gh.create_issue_comment(pr_number, "/tag-and-rerun-ci")
                log.info("ci_monitor.approve_auto_ci_started", pr=pr_number)
            except Exception:
                log.exception("ci_monitor.approve_auto_ci_comment_failed", pr=pr_number)
                return
            snapshot_data["approve_auto_ci_started"] = True
            await queries.upsert_ci_snapshot(
                self._db.conn, pr_number, head_sha,
                overall_status=ci_status.overall.value,
                has_run_ci_label=ci_status.has_run_ci_label,
                failed_jobs=json.dumps([j.job_name for j in ci_status.failed_jobs]),
                snapshot_data=json.dumps(snapshot_data),
            )
            return

        # Case 2: CI failed → retry
        if (
            ci_status.overall == CIOverallStatus.FAILED
            and ci_status.all_runs_completed
        ):
            max_retries = self._settings.ci_approve_auto_ci_max_retries
            should_rerun = False
            for job in ci_status.failed_jobs:
                count = await queries.get_ci_retry_count(
                    self._db.conn, pr_number, head_sha, job.job_name
                )
                if count < max_retries:
                    should_rerun = True
                    await queries.increment_ci_retry(
                        self._db.conn, pr_number, head_sha, job.job_name
                    )
            if should_rerun:
                try:
                    await self._gh.create_issue_comment(pr_number, "/rerun-failed-ci")
                    log.info("ci_monitor.approve_auto_ci_retried", pr=pr_number)
                except Exception:
                    log.exception("ci_monitor.approve_auto_ci_retry_failed", pr=pr_number)

    @staticmethod
    def _load_snapshot_data(snapshot: dict[str, Any] | None) -> dict[str, Any]:
        """Load snapshot_data JSON from a ci_snapshot row."""
        if snapshot and snapshot.get("snapshot_data"):
            try:
                return json.loads(snapshot["snapshot_data"])
            except (json.JSONDecodeError, TypeError):
                pass
        return {}

    async def _poll_single_pr(self, pr_number: int, user_ids: list[str]) -> None:
        """Poll CI status for a single tracked PR."""
        # Get current PR info
        try:
            pr_data = await self._gh.get_pull(pr_number)
        except Exception:
            log.warning("ci_monitor.pr_fetch_failed", pr=pr_number)
            return

        # Skip closed/merged PRs (lifecycle handled in __main__.py)
        if pr_data["state"] != "open":
            return

        head_sha = pr_data["head"]["sha"]
        ci_status = await self.check_pr_ci(pr_number, head_sha, pr_data=pr_data)

        # Get previous snapshot to detect changes
        prev_snapshot = await queries.get_ci_snapshot(self._db.conn, pr_number, head_sha)
        prev_overall = prev_snapshot["overall_status"] if prev_snapshot else None

        # Get review state
        reviews = await self._gh.get_pull_reviews(pr_number)
        review_state = "none"
        for r in reviews:
            if r.get("state") == "APPROVED":
                review_state = "approved"
                break
            elif r.get("state") == "CHANGES_REQUESTED":
                review_state = "changes_requested"

        # Get commit count
        try:
            commits = await self._gh.get_pull_commits(pr_number)
            commit_count = len(commits)
        except Exception:
            commit_count = 0

        # Save snapshot
        failed_names = [j.job_name for j in ci_status.failed_jobs]
        await queries.upsert_ci_snapshot(
            self._db.conn,
            pr_number,
            head_sha,
            overall_status=ci_status.overall.value,
            has_run_ci_label=ci_status.has_run_ci_label,
            failed_jobs=json.dumps(failed_names),
            review_state=review_state,
            commit_count=commit_count,
        )

        # --- Approve auto-CI: start or retry CI when approved by configured users ---
        await self._maybe_approve_auto_ci(pr_number, head_sha, ci_status)

        # --- All CI passed + approved → ping (once) ---
        # Skip if auto-merge is already pending for this PR
        merge_pending = self._is_merge_pending and self._is_merge_pending(pr_number)
        if (
            self._all_ci_passed(ci_status)
            and review_state == "approved"
            and self._on_ci_passed_approved
            and not merge_pending
        ):
            # Check snapshot_data to avoid duplicate pings
            snapshot_data = self._load_snapshot_data(prev_snapshot)
            if not snapshot_data.get("hp_nvidia_pinged"):
                await self._on_ci_passed_approved(pr_number, user_ids, review_state)
                snapshot_data["hp_nvidia_pinged"] = True
                await queries.upsert_ci_snapshot(
                    self._db.conn,
                    pr_number,
                    head_sha,
                    overall_status=ci_status.overall.value,
                    has_run_ci_label=ci_status.has_run_ci_label,
                    failed_jobs=json.dumps(failed_names),
                    review_state=review_state,
                    commit_count=commit_count,
                    snapshot_data=json.dumps(snapshot_data),
                )

        # --- Merge-ready check: runs EVERY poll when CI is PASSED ---
        # This catches: approve came after CI passed, PR became mergeable
        # after conflicts resolved, PR tracked when already CI-passed, etc.
        if ci_status.overall == CIOverallStatus.PASSED:
            if self._on_merge_ready_check:
                await self._on_merge_ready_check(pr_number, user_ids, review_state)

        # --- CI status change notifications ---
        if prev_overall == ci_status.overall.value:
            return

        if ci_status.overall == CIOverallStatus.PASSED:
            if self._on_ci_passed:
                await self._on_ci_passed(pr_number, user_ids, ci_status, review_state)

        elif ci_status.overall == CIOverallStatus.FAILED and ci_status.all_runs_completed:
            if not ci_status.has_run_ci_label:
                return  # Don't retry if no run-ci label

            retryable: list[CIJobResult] = []
            permanent: list[CIJobResult] = []

            for job in ci_status.failed_jobs:
                if await self.should_retry(
                    pr_number, head_sha, job,
                    is_high_priority=ci_status.has_high_priority_label,
                ):
                    retryable.append(job)
                else:
                    permanent.append(job)

            if retryable:
                # Comment to trigger rerun
                for job in retryable:
                    await queries.increment_ci_retry(
                        self._db.conn, pr_number, head_sha, job.job_name
                    )
                try:
                    await self._gh.create_issue_comment(pr_number, "/rerun-failed-ci")
                    log.info("ci_monitor.rerun_via_comment", pr=pr_number)
                except Exception:
                    log.exception("ci_monitor.rerun_comment_failed", pr=pr_number)

                if self._on_ci_failed_retrying:
                    await self._on_ci_failed_retrying(pr_number, user_ids, retryable)

            if permanent:
                if self._on_ci_failed_permanent:
                    await self._on_ci_failed_permanent(pr_number, user_ids, permanent)
