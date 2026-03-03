"""Auto/manual CI rerun logic for flaky and infra failures."""

from __future__ import annotations

from dataclasses import dataclass

import structlog

from sgldhelper.config import Settings
from sgldhelper.db.engine import Database
from sgldhelper.db import queries
from sgldhelper.github.ci_analyzer import CIResult, FailureCategory
from sgldhelper.github.client import GitHubClient

log = structlog.get_logger()


@dataclass
class RerunResult:
    run_id: int
    pr_number: int
    triggered: bool
    reason: str
    triggered_by: str = "auto"


class CIRerunner:
    """Decide whether to auto-rerun failed CI and execute reruns."""

    def __init__(
        self, client: GitHubClient, db: Database, settings: Settings
    ) -> None:
        self._client = client
        self._db = db
        self._settings = settings

    def should_auto_rerun(self, result: CIResult) -> bool:
        """Determine if a failed CI run should be automatically rerun."""
        if not self._settings.auto_rerun_enabled:
            return False
        if result.conclusion != "failure":
            return False
        if result.failure_category not in (FailureCategory.FLAKY, FailureCategory.INFRA):
            return False
        if result.auto_rerun_count >= self._settings.max_auto_reruns:
            return False
        return True

    async def auto_rerun(self, result: CIResult) -> RerunResult:
        """Attempt auto-rerun for a failed CI run."""
        if not self.should_auto_rerun(result):
            return RerunResult(
                run_id=result.run_id,
                pr_number=result.pr_number,
                triggered=False,
                reason=f"Auto-rerun not applicable (category={result.failure_category}, reruns={result.auto_rerun_count})",
            )

        try:
            await self._client.create_issue_comment(
                result.pr_number, "/rerun-failed-ci"
            )
        except Exception as e:
            log.error("ci.rerun_failed", run_id=result.run_id, error=str(e))
            return RerunResult(
                run_id=result.run_id,
                pr_number=result.pr_number,
                triggered=False,
                reason=f"API call failed: {e}",
            )

        await queries.increment_rerun_count(self._db.conn, result.run_id)
        await queries.log_rerun(
            self._db.conn,
            run_id=result.run_id,
            new_run_id=None,
            triggered_by="auto",
            reason=f"{result.failure_category.value}: {result.failure_summary}",
        )

        log.info(
            "ci.auto_rerun_triggered",
            run_id=result.run_id,
            pr=result.pr_number,
            category=result.failure_category.value,
        )

        return RerunResult(
            run_id=result.run_id,
            pr_number=result.pr_number,
            triggered=True,
            reason=f"Auto-rerun #{result.auto_rerun_count + 1} for {result.failure_category.value}",
        )

    async def manual_rerun(self, pr_number: int) -> list[RerunResult]:
        """Manually rerun failed CI for a PR by commenting /rerun-failed-ci."""
        ci_runs = await queries.get_ci_runs_for_pr(self._db.conn, pr_number)

        failed_runs = [r for r in ci_runs if r["conclusion"] == "failure"]
        if not failed_runs:
            return []

        # One comment triggers all failed jobs — no need to comment per run.
        try:
            await self._client.create_issue_comment(pr_number, "/rerun-failed-ci")
        except Exception as e:
            return [
                RerunResult(
                    run_id=r["run_id"],
                    pr_number=pr_number,
                    triggered=False,
                    reason=f"Failed: {e}",
                    triggered_by="manual",
                )
                for r in failed_runs
            ]

        results: list[RerunResult] = []
        for run in failed_runs:
            await queries.log_rerun(
                self._db.conn,
                run_id=run["run_id"],
                new_run_id=None,
                triggered_by="manual",
                reason="Manual rerun via Slack",
            )
            results.append(RerunResult(
                run_id=run["run_id"],
                pr_number=pr_number,
                triggered=True,
                reason="Manual rerun triggered",
                triggered_by="manual",
            ))
            log.info("ci.manual_rerun", run_id=run["run_id"], pr=pr_number)

        return results
