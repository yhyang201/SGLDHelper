"""Central event dispatcher routing changes to Slack notifications."""

from __future__ import annotations

import structlog

from sgldhelper.config import Settings
from sgldhelper.notifications.ci_events import CIEventHandler
from sgldhelper.notifications.feature_events import FeatureEventHandler
from sgldhelper.notifications.pr_events import PREventHandler
from sgldhelper.slack.app import SlackApp
from sgldhelper.slack.channels import ChannelRouter

log = structlog.get_logger()


class NotificationDispatcher:
    """Facade that owns all event handlers and provides a unified interface."""

    def __init__(
        self,
        slack_app: SlackApp,
        channels: ChannelRouter,
        settings: Settings,
    ) -> None:
        self.pr = PREventHandler(slack_app, channels, settings)
        self.ci = CIEventHandler(slack_app, channels, settings)
        self.feature = FeatureEventHandler(slack_app, channels, settings)
