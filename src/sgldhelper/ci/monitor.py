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

    def set_callbacks(
        self,
        *,
        on_ci_passed: Any = None,
        on_ci_failed_retrying: Any = None,
        on_ci_failed_permanent: Any = None,
        on_merge_ready_check: Any = None,
    ) -> None:
        self._on_ci_passed = on_ci_passed
        self._on_ci_failed_retrying = on_ci_failed_retrying
        self._on_ci_failed_permanent = on_ci_failed_permanent
        self._on_merge_ready_check = on_merge_ready_check

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
        )

    async def should_retry(
        self, pr_number: int, head_sha: str, job: CIJobResult
    ) -> bool:
        """Check if a failed job should be retried (under max retries)."""
        count = await queries.get_ci_retry_count(
            self._db.conn, pr_number, head_sha, job.job_name
        )
        return count < self._settings.ci_max_retries

    async def trigger_ci(self, pr_number: int, has_label: bool) -> None:
        """Trigger CI by adding run-ci label comment. Uses lock to prevent concurrent triggers."""
        async with self._trigger_lock:
            if not has_label:
                await self._gh.create_issue_comment(pr_number, "/tag-and-rerun-ci")
                log.info("ci.triggered_via_comment", pr=pr_number)
            else:
                # If label exists, find runs and rerun failed
                pr_data = await self._gh.get_pull(pr_number)
                sha = pr_data["head"]["sha"]
                runs = await self._gh.get_workflow_runs_for_ref(sha)
                for run in runs:
                    wf_id = run.get("workflow_id")
                    if wf_id in (self._settings.ci_nvidia_workflow_id, self._settings.ci_amd_workflow_id):
                        if run["status"] == "completed" and run["conclusion"] == "failure":
                            await self._gh.rerun_failed_jobs(run["id"])
                            log.info("ci.rerun_triggered", pr=pr_number, run_id=run["id"])

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
                if await self.should_retry(pr_number, head_sha, job):
                    retryable.append(job)
                else:
                    permanent.append(job)

            if retryable:
                # Rerun the failed runs
                rerun_ids: set[int] = set()
                for job in retryable:
                    if job.run_id not in rerun_ids:
                        try:
                            await self._gh.rerun_failed_jobs(job.run_id)
                            rerun_ids.add(job.run_id)
                        except Exception:
                            log.exception("ci_monitor.rerun_failed", pr=pr_number, run_id=job.run_id)
                    await queries.increment_ci_retry(
                        self._db.conn, pr_number, head_sha, job.job_name
                    )

                if self._on_ci_failed_retrying:
                    await self._on_ci_failed_retrying(pr_number, user_ids, retryable)

            if permanent:
                if self._on_ci_failed_permanent:
                    await self._on_ci_failed_permanent(pr_number, user_ids, permanent)
