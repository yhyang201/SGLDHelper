"""CI event notifications to Slack."""

from __future__ import annotations

import structlog

from sgldhelper.config import Settings
from sgldhelper.db.engine import Database
from sgldhelper.db import queries
from sgldhelper.github.ci_analyzer import CIResult
from sgldhelper.github.ci_rerunner import RerunResult
from sgldhelper.slack.app import SlackApp
from sgldhelper.slack.channels import ChannelRouter
from sgldhelper.slack import messages

log = structlog.get_logger()


class CIEventHandler:
    """Send Slack notifications for CI events."""

    def __init__(
        self,
        slack_app: SlackApp,
        channels: ChannelRouter,
        settings: Settings,
    ) -> None:
        self._slack = slack_app
        self._channels = channels
        self._settings = settings

    async def handle_failure(self, result: CIResult, db: Database) -> None:
        """Notify about a CI failure."""
        msg = messages.build_ci_failure(result, self._settings.github_repo)

        # Try to thread under the PR's main message
        stored = await queries.get_pr(db.conn, result.pr_number)
        thread_ts = stored.get("slack_thread_ts") if stored else None

        await self._slack.post_message(
            self._channels.ci_channel,
            text=msg["text"],
            blocks=msg.get("blocks"),
            thread_ts=thread_ts,
        )
        log.info("notification.ci_failure", pr=result.pr_number, run=result.run_id)

    async def handle_success(self, pr_number: int, db: Database) -> None:
        """Notify about CI passing."""
        msg = messages.build_ci_success(pr_number, self._settings.github_repo)

        stored = await queries.get_pr(db.conn, pr_number)
        thread_ts = stored.get("slack_thread_ts") if stored else None

        await self._slack.post_message(
            self._channels.ci_channel,
            text=msg["text"],
            blocks=msg.get("blocks"),
            thread_ts=thread_ts,
        )
        log.info("notification.ci_success", pr=pr_number)

    async def handle_rerun(
        self, result: CIResult, rerun: RerunResult, db: Database
    ) -> None:
        """Notify about a CI rerun (auto or manual)."""
        msg = messages.build_ci_rerun(
            result,
            auto=(rerun.triggered_by == "auto"),
            repo=self._settings.github_repo,
        )

        stored = await queries.get_pr(db.conn, result.pr_number)
        thread_ts = stored.get("slack_thread_ts") if stored else None

        await self._slack.post_message(
            self._channels.ci_channel,
            text=msg["text"],
            blocks=msg.get("blocks"),
            thread_ts=thread_ts,
        )
        log.info("notification.ci_rerun", pr=result.pr_number, auto=rerun.triggered_by == "auto")
