"""Channel routing configuration for Slack notifications."""

from __future__ import annotations

from dataclasses import dataclass

from sgldhelper.config import Settings


@dataclass
class ChannelRouter:
    """Route notifications to the appropriate Slack channel."""

    pr_channel: str
    ci_channel: str
    features_channel: str

    @classmethod
    def from_settings(cls, settings: Settings) -> ChannelRouter:
        return cls(
            pr_channel=settings.slack_pr_channel,
            ci_channel=settings.slack_ci_channel,
            features_channel=settings.slack_features_channel,
        )
