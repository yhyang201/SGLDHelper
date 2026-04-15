"""CI event notifications to Slack."""

from __future__ import annotations

from typing import Any

import structlog

from sgldhelper.ci.monitor import CIJobResult, CIStatus
from sgldhelper.config import Settings
from sgldhelper.db.engine import Database
from sgldhelper.slack.app import SlackApp
from sgldhelper.slack.channels import ChannelRouter
from sgldhelper.slack import messages

log = structlog.get_logger()


class CIEventHandler:
    """Send Slack notifications for CI events on tracked PRs."""

    def __init__(
        self,
        slack_app: SlackApp,
        channels: ChannelRouter,
        settings: Settings,
        db: Database,
    ) -> None:
        self._slack = slack_app
        self._channels = channels
        self._settings = settings
        self._db = db

    async def _post(self, msg: dict[str, Any]) -> None:
        """Post a message and save to conversation history for thread context."""
        await self._slack.post_message_with_context(
            self._channels.ci_channel,
            text=msg["text"],
            blocks=msg.get("blocks"),
            db_conn=self._db.conn,
        )

    async def notify_ci_passed(
        self,
        pr_number: int,
        user_ids: list[str],
        ci_status: CIStatus,
        review_state: str,
    ) -> None:
        msg = messages.build_ci_passed(pr_number, ci_status, user_ids, self._settings.github_repo)
        await self._post(msg)
        log.info("notification.ci_passed", pr=pr_number)

    async def notify_ci_failed_retrying(
        self,
        pr_number: int,
        user_ids: list[str],
        retryable_jobs: list[CIJobResult],
    ) -> None:
        msg = messages.build_ci_failed_retrying(
            pr_number, retryable_jobs, user_ids, self._settings.github_repo,
        )
        await self._post(msg)
        log.info("notification.ci_failed_retrying", pr=pr_number, jobs=len(retryable_jobs))

    async def notify_ci_failed_permanent(
        self,
        pr_number: int,
        user_ids: list[str],
        permanent_jobs: list[CIJobResult],
    ) -> None:
        msg = messages.build_ci_failed_permanent(
            pr_number, permanent_jobs, user_ids, self._settings.github_repo,
        )
        await self._post(msg)
        log.info("notification.ci_failed_permanent", pr=pr_number, jobs=len(permanent_jobs))

    async def notify_pr_untracked(
        self,
        pr_number: int,
        user_ids: list[str],
        reason: str,
    ) -> None:
        msg = messages.build_pr_untracked(
            pr_number, user_ids, reason, self._settings.github_repo,
        )
        await self._post(msg)
        log.info("notification.pr_untracked", pr=pr_number, reason=reason)

    async def notify_merge_countdown(
        self, pr_number: int, user_ids: list[str], delay_seconds: int
    ) -> None:
        msg = messages.build_merge_countdown(
            pr_number, user_ids, delay_seconds, self._settings.github_repo,
        )
        await self._post(msg)

    async def notify_merge_complete(self, pr_number: int, user_ids: list[str]) -> None:
        msg = messages.build_merge_complete(pr_number, user_ids, self._settings.github_repo)
        await self._post(msg)

    async def notify_merge_cancelled(self, pr_number: int, user_ids: list[str]) -> None:
        msg = messages.build_merge_cancelled(pr_number, user_ids, self._settings.github_repo)
        await self._post(msg)

    async def notify_code_quality_report(
        self, report: str, pr_count: int, alert_prs: list[dict],
    ) -> None:
        alert_user_ids = self._settings.code_quality_alert_user_ids
        msg = messages.build_code_quality_report(
            report, pr_count, alert_prs, alert_user_ids, self._settings.github_repo,
        )
        await self._post(msg)
        log.info(
            "notification.code_quality_report",
            pr_count=pr_count,
            alerts=len(alert_prs),
        )

    async def notify_tracked_pr_summary(
        self, pr_number: int, user_ids: list[str], summary: str
    ) -> None:
        msg = messages.build_tracked_pr_summary(pr_number, user_ids, summary, self._settings.github_repo)
        await self._post(msg)
