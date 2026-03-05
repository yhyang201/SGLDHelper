"""Slash command and interactive button handlers for Slack."""

from __future__ import annotations

import re
import time
from collections import defaultdict

import structlog

from sgldhelper.config import Settings
from sgldhelper.db.engine import Database
from sgldhelper.db import queries
from sgldhelper.slack.app import SlackApp
from sgldhelper.slack.channels import ChannelRouter
from sgldhelper.slack import messages
from sgldhelper.ai.classifier import MessageClassifier
from sgldhelper.ai.conversation import ConversationManager
from sgldhelper.ai.summaries import SummaryGenerator

_GITHUB_NOT_CONFIGURED_MSG = (
    ":warning: *GITHUB_TOKEN is not configured.* "
    "This operation requires GitHub access. "
    "Please set `GITHUB_TOKEN` in your `.env` file and restart."
)

log = structlog.get_logger()

# Per-user rate limiting for AI interactions
_user_timestamps: dict[str, list[float]] = defaultdict(list)


def _check_user_rate_limit(user_id: str, max_calls: int, window_seconds: int) -> bool:
    """Return True if the user is within rate limits."""
    now = time.monotonic()
    timestamps = _user_timestamps[user_id]
    # Prune old entries
    _user_timestamps[user_id] = [t for t in timestamps if now - t < window_seconds]
    if len(_user_timestamps[user_id]) >= max_calls:
        return False
    _user_timestamps[user_id].append(now)
    return True


def register_handlers(
    slack_app: SlackApp,
    db: Database,
    channels: ChannelRouter,
    settings: Settings,
    *,
    conversation_manager: ConversationManager,
    classifier: MessageClassifier,
    summary_generator: SummaryGenerator,
    auto_merge: object | None = None,
) -> None:
    """Register all slash commands and action handlers on the Slack app."""
    app = slack_app.app

    # ------------------------------------------------------------------
    # AI-powered handlers
    # ------------------------------------------------------------------

    monitored_channels = {
        channels.pr_channel, channels.ci_channel
    }
    log.info(
        "handlers.registered",
        monitored_channels=list(monitored_channels),
    )

    @app.event("message")
    async def handle_message(event, say):
        """Respond to ALL messages in monitored channels via AI."""
        # Skip subtypes: bot messages, edits, joins, etc.
        if event.get("subtype"):
            return
        # Skip bot's own messages to avoid infinite loops
        if event.get("bot_id") or event.get("user") == slack_app.bot_user_id:
            return

        channel = event.get("channel", "")
        if channel not in monitored_channels:
            return

        user_id = event.get("user", "")
        text = event.get("text", "")
        message_ts = event.get("ts", "")
        thread_ts = event.get("thread_ts") or message_ts

        # Strip any bot mentions from text
        text = re.sub(r"<@[A-Z0-9]+>\s*", "", text).strip()
        if not text:
            return

        # Rate limit check
        if not _check_user_rate_limit(
            user_id, settings.ai_user_cooldown_max, settings.ai_user_cooldown_seconds
        ):
            return  # Silently drop when rate-limited on passive replies

        # Check for auto-merge cancel keywords
        if auto_merge is not None:
            from sgldhelper.ci.auto_merge import AutoMergeManager
            if isinstance(auto_merge, AutoMergeManager):
                cancel_pr = auto_merge.check_cancel_keywords(text)
                if cancel_pr is not None:
                    import asyncio
                    cancelled = await auto_merge.cancel(cancel_pr)
                    if cancelled:
                        await say(
                            f":no_entry_sign: PR #{cancel_pr} 的自动合并已取消。",
                            thread_ts=thread_ts,
                        )
                        return

        # Add thinking reaction
        try:
            await app.client.reactions_add(
                channel=channel, timestamp=message_ts, name="thinking_face"
            )
        except Exception:
            pass

        try:
            # Run classifier in parallel
            try:
                cls_result = await classifier.classify_message(
                    text=text, user_id=user_id,
                    channel_id=channel, message_ts=message_ts,
                )
                if cls_result:
                    msg = messages.build_progress_confirmation(cls_result)
                    await say(
                        blocks=msg["blocks"], text=msg["text"],
                        thread_ts=thread_ts,
                    )
            except Exception as e:
                log.error("ai.classify_failed", error=str(e))

            # Always reply via conversation manager
            reply = await conversation_manager.handle_mention(
                text=text,
                thread_ts=thread_ts,
                channel_id=channel,
                user_id=user_id,
            )
            await say(reply, thread_ts=thread_ts)
        except Exception as e:
            log.error("ai.reply_failed", error=str(e), user=user_id)
            await say(
                "Sorry, I encountered an error processing your message.",
                thread_ts=thread_ts,
            )
        finally:
            try:
                await app.client.reactions_remove(
                    channel=channel, timestamp=message_ts, name="thinking_face"
                )
            except Exception:
                pass

    @app.event("app_mention")
    async def handle_app_mention(event, say):
        """Handle @mentions — also routed through conversation manager.

        In channels outside the monitored set, @mention still works.
        In monitored channels, the message handler already handles it,
        so we only process mentions from non-monitored channels here.
        """
        channel = event.get("channel", "")
        if channel in monitored_channels:
            return  # Already handled by message handler

        user_id = event.get("user", "")
        text = event.get("text", "")
        message_ts = event.get("ts", "")
        thread_ts = event.get("thread_ts") or message_ts

        text = re.sub(r"<@[A-Z0-9]+>\s*", "", text).strip()
        if not text:
            await say("How can I help? Ask me about PRs or CI.", thread_ts=thread_ts)
            return

        if not _check_user_rate_limit(
            user_id, settings.ai_user_cooldown_max, settings.ai_user_cooldown_seconds
        ):
            await say("You're sending messages too fast. Please wait a moment.", thread_ts=thread_ts)
            return

        try:
            await app.client.reactions_add(channel=channel, timestamp=message_ts, name="thinking_face")
        except Exception:
            pass

        try:
            reply = await conversation_manager.handle_mention(
                text=text, thread_ts=thread_ts, channel_id=channel, user_id=user_id,
            )
            await say(reply, thread_ts=thread_ts)
        except Exception as e:
            log.error("ai.mention_failed", error=str(e), user=user_id)
            await say("Sorry, I encountered an error.", thread_ts=thread_ts)
        finally:
            try:
                await app.client.reactions_remove(channel=channel, timestamp=message_ts, name="thinking_face")
            except Exception:
                pass

    @app.action("confirm_update")
    async def handle_confirm_update(ack, body, respond):
        """User confirmed a detected progress update."""
        await ack()
        update_id = int(body["actions"][0]["value"])
        user = body["user"]["username"]
        await queries.confirm_detected_update(db.conn, update_id)
        await respond(f":white_check_mark: Update confirmed by @{user}. Progress recorded.")

    @app.action("dismiss_update")
    async def handle_dismiss_update(ack, body, respond):
        """User dismissed a detected update (false positive)."""
        await ack()
        await respond(":no_entry_sign: Update dismissed. Thanks for the feedback!")

    @app.command("/diffusion-standup")
    async def handle_standup(ack, respond):
        """Generate a daily standup summary."""
        await ack()
        try:
            summary = await summary_generator.generate_standup()
            await respond(summary)
        except Exception as e:
            log.error("ai.standup_failed", error=str(e))
            await respond(f"Failed to generate standup summary: {e}")
