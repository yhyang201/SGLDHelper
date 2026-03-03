"""CI failure analysis and classification for diffusion jobs."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any

import structlog

from sgldhelper.config import Settings
from sgldhelper.db.engine import Database
from sgldhelper.db import queries
from sgldhelper.github.client import GitHubClient

log = structlog.get_logger()


class FailureCategory(str, Enum):
    FLAKY = "flaky"
    INFRA = "infra"
    PERF_REGRESSION = "perf_regression"
    CODE = "code"
    UNKNOWN = "unknown"


# Patterns referenced from SGLang's run_suite.py and ci_failures_analysis.py
FLAKY_PATTERNS = [
    r"SafetensorError",
    r"FileNotFoundError.*safetensors",
    r"torch\.cuda\.OutOfMemoryError",
    r"Connection refused",
    r"asyncio\.TimeoutError",
    r"ServerDisconnectedError",
    r"CUDA error: device-side assert triggered",
    r"RuntimeError: NCCL",
]

INFRA_PATTERNS = [
    r"out of memory",
    r"oom.killer",
    r"OOM",
    r"Connection reset by peer",
    r"No space left on device",
    r"Read timed out",
    r"Could not resolve host",
    r"Failed to download",
    r"HTTP Error 503",
    r"runner environment",
]

PERF_REGRESSION_PATTERNS = [
    r"AssertionError.*test_server_utils",
    r"AssertionError.*throughput",
    r"AssertionError.*latency",
    r"performance regression",
]


@dataclass
class CIResult:
    run_id: int
    pr_number: int
    job_name: str
    head_sha: str
    status: str
    conclusion: str | None
    failure_category: FailureCategory | None
    failure_summary: str | None
    html_url: str | None
    auto_rerun_count: int


class CIAnalyzer:
    """Analyze CI runs for diffusion-related jobs, classify failures."""

    def __init__(
        self, client: GitHubClient, db: Database, settings: Settings
    ) -> None:
        self._client = client
        self._db = db
        self._settings = settings

    async def analyze_pr(self, pr_number: int, head_sha: str) -> list[CIResult]:
        """Analyze all diffusion CI jobs for a given PR's head SHA."""
        results: list[CIResult] = []

        try:
            workflow_runs = await self._client.get_workflow_runs_for_ref(head_sha)
        except Exception as e:
            log.error("ci.workflow_runs_failed", pr=pr_number, error=str(e))
            return results

        for wf_run in workflow_runs:
            try:
                jobs = await self._client.get_workflow_run_jobs(wf_run["id"])
            except Exception as e:
                log.error("ci.jobs_fetch_failed", run_id=wf_run["id"], error=str(e))
                continue

            for job in jobs:
                if not self._is_diffusion_job(job["name"]):
                    continue

                category = None
                summary = None
                if job["conclusion"] == "failure":
                    category, summary = await self._classify_failure(job)

                result = CIResult(
                    run_id=wf_run["id"],
                    pr_number=pr_number,
                    job_name=job["name"],
                    head_sha=head_sha,
                    status=job["status"],
                    conclusion=job["conclusion"],
                    failure_category=category,
                    failure_summary=summary,
                    html_url=job.get("html_url"),
                    auto_rerun_count=0,
                )

                # Check existing record for rerun count
                existing = await queries.get_ci_run(self._db.conn, wf_run["id"])
                if existing:
                    result.auto_rerun_count = existing["auto_rerun_count"]

                await queries.upsert_ci_run(self._db.conn, {
                    "run_id": result.run_id,
                    "pr_number": result.pr_number,
                    "job_name": result.job_name,
                    "head_sha": result.head_sha,
                    "status": result.status,
                    "conclusion": result.conclusion,
                    "failure_category": result.failure_category.value if result.failure_category else None,
                    "failure_summary": result.failure_summary,
                    "auto_rerun_count": result.auto_rerun_count,
                    "html_url": result.html_url,
                })

                results.append(result)

        return results

    def _is_diffusion_job(self, job_name: str) -> bool:
        """Check if a job name matches diffusion CI patterns."""
        name_lower = job_name.lower()
        return any(
            pattern.lower() in name_lower
            for pattern in self._settings.diffusion_ci_jobs
        )

    async def _classify_failure(
        self, job: dict[str, Any]
    ) -> tuple[FailureCategory, str]:
        """Classify a CI failure by analyzing job logs."""
        log_text = ""
        try:
            log_text = await self._client.get_job_logs(job["id"])
        except Exception as e:
            log.warning("ci.log_fetch_failed", job_id=job["id"], error=str(e))
            return FailureCategory.UNKNOWN, f"Could not fetch logs: {e}"

        # Truncate to last 5000 chars for pattern matching
        log_tail = log_text[-5000:] if len(log_text) > 5000 else log_text

        for pattern in FLAKY_PATTERNS:
            match = re.search(pattern, log_tail, re.IGNORECASE)
            if match:
                return FailureCategory.FLAKY, f"Flaky: {match.group(0)}"

        for pattern in INFRA_PATTERNS:
            match = re.search(pattern, log_tail, re.IGNORECASE)
            if match:
                return FailureCategory.INFRA, f"Infra: {match.group(0)}"

        for pattern in PERF_REGRESSION_PATTERNS:
            match = re.search(pattern, log_tail, re.IGNORECASE)
            if match:
                return FailureCategory.PERF_REGRESSION, f"Perf: {match.group(0)}"

        # Extract last error line for code failures
        error_lines = [
            line for line in log_tail.splitlines()
            if "error" in line.lower() or "Error" in line or "FAILED" in line
        ]
        summary = error_lines[-1].strip()[:200] if error_lines else "Unknown failure"
        return FailureCategory.CODE, summary
