"""Feature progress notifications to Slack."""

from __future__ import annotations

import structlog

from sgldhelper.config import Settings
from sgldhelper.db.engine import Database
from sgldhelper.db import queries
from sgldhelper.github.issue_tracker import FeatureProgress
from sgldhelper.slack.app import SlackApp
from sgldhelper.slack.channels import ChannelRouter
from sgldhelper.slack import messages

log = structlog.get_logger()


class FeatureEventHandler:
    """Send Slack notifications for feature progress updates."""

    def __init__(
        self,
        slack_app: SlackApp,
        channels: ChannelRouter,
        settings: Settings,
    ) -> None:
        self._slack = slack_app
        self._channels = channels
        self._settings = settings

    async def handle_progress(self, progress: FeatureProgress) -> None:
        """Send a feature progress update."""
        msg = messages.build_feature_progress(progress, self._settings.github_repo)
        await self._slack.post_message(
            self._channels.features_channel,
            text=msg["text"],
            blocks=msg.get("blocks"),
        )
        log.info(
            "notification.feature_progress",
            issue=progress.issue_number,
            pct=f"{progress.percent:.0f}%",
        )

    async def send_daily_digest(
        self,
        db: Database,
        progress_list: list[FeatureProgress],
    ) -> None:
        """Send the daily digest summarising open PRs and feature progress."""
        open_prs = await queries.get_open_prs(db.conn)
        msg = messages.build_daily_digest(
            open_prs, progress_list, self._settings.github_repo
        )
        await self._slack.post_message(
            self._channels.features_channel,
            text=msg["text"],
            blocks=msg.get("blocks"),
        )
        log.info("notification.daily_digest", prs=len(open_prs))
