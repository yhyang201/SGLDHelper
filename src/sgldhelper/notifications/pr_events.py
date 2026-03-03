"""PR event notifications to Slack."""

from __future__ import annotations

import structlog

from sgldhelper.config import Settings
from sgldhelper.db.engine import Database
from sgldhelper.db import queries
from sgldhelper.github.pr_tracker import PRChange, PREvent
from sgldhelper.slack.app import SlackApp
from sgldhelper.slack.channels import ChannelRouter
from sgldhelper.slack import messages

log = structlog.get_logger()


class PREventHandler:
    """Send Slack notifications for PR lifecycle events."""

    def __init__(
        self,
        slack_app: SlackApp,
        channels: ChannelRouter,
        settings: Settings,
    ) -> None:
        self._slack = slack_app
        self._channels = channels
        self._settings = settings

    async def handle(self, change: PRChange, db: Database) -> None:
        """Dispatch a PR change to the appropriate Slack notification."""
        builder = messages.PR_MESSAGE_BUILDERS.get(change.event)
        if not builder:
            return

        msg = builder(change, self._settings.github_repo)
        pr_number = change.pr["pr_number"]
        stored = await queries.get_pr(db.conn, pr_number)

        if change.event == PREvent.OPENED:
            # Create a new top-level message
            result = await self._slack.post_message(
                self._channels.pr_channel,
                text=msg["text"],
                blocks=msg.get("blocks"),
            )
            thread_ts = result.get("ts")
            if thread_ts:
                await queries.set_pr_slack_thread(db.conn, pr_number, thread_ts)
            log.info("notification.pr_opened", pr=pr_number)
        else:
            # Reply in thread if we have one
            thread_ts = stored.get("slack_thread_ts") if stored else None
            await self._slack.post_message(
                self._channels.pr_channel,
                text=msg["text"],
                blocks=msg.get("blocks"),
                thread_ts=thread_ts,
            )
            log.info("notification.pr_event", pr=pr_number, event=change.event.value)

        await queries.set_pr_notified_state(db.conn, pr_number, change.event.value)
