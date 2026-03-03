"""Slash command and interactive button handlers for Slack."""

from __future__ import annotations

import re
import time
from collections import defaultdict
from typing import TYPE_CHECKING

import structlog

from sgldhelper.config import Settings
from sgldhelper.db.engine import Database
from sgldhelper.db import queries
from sgldhelper.github.ci_rerunner import CIRerunner
from sgldhelper.github.issue_tracker import IssueTracker
from sgldhelper.slack.app import SlackApp
from sgldhelper.slack.channels import ChannelRouter
from sgldhelper.slack import messages

if TYPE_CHECKING:
    from sgldhelper.ai.classifier import MessageClassifier
    from sgldhelper.ai.conversation import ConversationManager
    from sgldhelper.ai.summaries import SummaryGenerator

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
    rerunner: CIRerunner,
    issue_tracker: IssueTracker,
    channels: ChannelRouter,
    settings: Settings,
    *,
    conversation_manager: ConversationManager | None = None,
    classifier: MessageClassifier | None = None,
    summary_generator: SummaryGenerator | None = None,
) -> None:
    """Register all slash commands and action handlers on the Slack app."""
    app = slack_app.app

    @app.command("/diffusion-status")
    async def handle_diffusion_status(ack, respond):
        await ack()
        open_prs = await queries.get_open_prs(db.conn)

        if not open_prs:
            await respond("No open diffusion PRs at the moment.")
            return

        lines = [f"*Open Diffusion PRs ({len(open_prs)}):*\n"]
        for pr in open_prs:
            url = f"https://github.com/{settings.github_repo}/pull/{pr['pr_number']}"
            lines.append(
                f"- <{url}|#{pr['pr_number']}> {pr['title']} "
                f"(`{pr['author']}`, `{pr['head_sha'][:8]}`)"
            )
        await respond("\n".join(lines))

    @app.command("/diffusion-rerun")
    async def handle_diffusion_rerun(ack, respond, command):
        await ack()
        text = command.get("text", "").strip()
        if not text:
            await respond("Usage: `/diffusion-rerun <PR#>`")
            return

        try:
            pr_number = int(text.lstrip("#"))
        except ValueError:
            await respond(f"Invalid PR number: `{text}`")
            return

        results = await rerunner.manual_rerun(pr_number)
        if not results:
            await respond(f"No failed CI runs found for PR #{pr_number}")
            return

        lines = [f"*Rerun results for PR #{pr_number}:*\n"]
        for r in results:
            status = ":white_check_mark: Triggered" if r.triggered else ":x: Failed"
            lines.append(f"- Run {r.run_id}: {status} - {r.reason}")
        await respond("\n".join(lines))

    @app.command("/diffusion-features")
    async def handle_diffusion_features(ack, respond):
        await ack()
        for issue_num in settings.roadmap_issue_numbers:
            try:
                progress = await issue_tracker.get_progress(issue_num)
                msg = messages.build_feature_progress(progress, settings.github_repo)
                await respond(blocks=msg["blocks"], text=msg["text"])
            except Exception as e:
                await respond(f"Error fetching issue #{issue_num}: {e}")

    @app.action("rerun_ci")
    async def handle_rerun_button(ack, body, respond):
        await ack()
        value = body["actions"][0]["value"]  # "run_id:pr_number"
        run_id_str, pr_number_str = value.split(":")
        run_id = int(run_id_str)
        pr_number = int(pr_number_str)

        user = body["user"]["username"]
        log.info("ci.manual_rerun_button", user=user, run_id=run_id, pr=pr_number)

        results = await rerunner.manual_rerun(pr_number)
        triggered = [r for r in results if r.triggered]
        if triggered:
            await respond(
                f":repeat: Rerun triggered by @{user} for PR #{pr_number} "
                f"({len(triggered)} run(s))"
            )
        else:
            await respond(f"No failed runs to rerun for PR #{pr_number}")

    @app.action("ci_status")
    async def handle_ci_status_button(ack, body, respond):
        await ack()
        pr_number = int(body["actions"][0]["value"])
        ci_runs = await queries.get_ci_runs_for_pr(db.conn, pr_number)

        if not ci_runs:
            await respond(f"No CI runs recorded for PR #{pr_number}")
            return

        lines = [f"*CI Status for PR #{pr_number}:*\n"]
        for run in ci_runs[:10]:
            status_emoji = {
                "success": ":white_check_mark:",
                "failure": ":x:",
                "in_progress": ":hourglass:",
            }.get(run["conclusion"] or run["status"], ":grey_question:")
            category = f" ({run['failure_category']})" if run.get("failure_category") else ""
            lines.append(
                f"{status_emoji} `{run['job_name']}` - {run['conclusion'] or run['status']}{category}"
            )
        await respond("\n".join(lines))

    # ------------------------------------------------------------------
    # AI-powered handlers (only registered when AI is enabled)
    # ------------------------------------------------------------------

    if settings.ai_enabled and conversation_manager:

        @app.event("app_mention")
        async def handle_app_mention(event, say):
            """Handle @bot mentions — route to Kimi K2.5 conversation."""
            user_id = event.get("user", "")
            text = event.get("text", "")
            channel = event.get("channel", "")
            thread_ts = event.get("thread_ts") or event.get("ts", "")

            # Strip the bot mention from text
            text = re.sub(r"<@[A-Z0-9]+>\s*", "", text).strip()
            if not text:
                await say("How can I help? Ask me about PRs, CI, or features.", thread_ts=thread_ts)
                return

            # Rate limit check
            if not _check_user_rate_limit(
                user_id, settings.ai_user_cooldown_max, settings.ai_user_cooldown_seconds
            ):
                await say(
                    "You're sending messages too fast. Please wait a moment.",
                    thread_ts=thread_ts,
                )
                return

            # Add thinking reaction
            try:
                await app.client.reactions_add(
                    channel=channel, timestamp=event["ts"], name="thinking_face"
                )
            except Exception:
                pass

            try:
                reply = await conversation_manager.handle_mention(
                    text=text,
                    thread_ts=thread_ts,
                    channel_id=channel,
                    user_id=user_id,
                )
                await say(reply, thread_ts=thread_ts)
            except Exception as e:
                log.error("ai.mention_failed", error=str(e), user=user_id)
                await say(
                    "Sorry, I encountered an error. Try a slash command instead:\n"
                    "- `/diffusion-status` — open PRs\n"
                    "- `/diffusion-features` — feature progress",
                    thread_ts=thread_ts,
                )
            finally:
                # Remove thinking reaction
                try:
                    await app.client.reactions_remove(
                        channel=channel, timestamp=event["ts"], name="thinking_face"
                    )
                except Exception:
                    pass

    if settings.ai_enabled and classifier:
        monitored_channels = {
            channels.pr_channel, channels.ci_channel, channels.features_channel
        }

        @app.event("message")
        async def handle_message(event, say):
            """Passively classify messages in monitored channels."""
            # Skip bot messages and message changes
            if event.get("subtype"):
                return
            channel = event.get("channel", "")
            if channel not in monitored_channels:
                return

            text = event.get("text", "")
            user_id = event.get("user", "")
            message_ts = event.get("ts", "")

            try:
                result = await classifier.classify_message(
                    text=text,
                    user_id=user_id,
                    channel_id=channel,
                    message_ts=message_ts,
                )
            except Exception as e:
                log.error("ai.classify_failed", error=str(e))
                return

            if not result:
                return

            # Send confirmation button
            msg = messages.build_progress_confirmation(result)
            await say(
                blocks=msg["blocks"],
                text=msg["text"],
                thread_ts=message_ts,
            )

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

    if settings.ai_enabled and summary_generator:

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
