"""Entry point: start all subsystems (pollers + Slack) under asyncio."""

from __future__ import annotations

import asyncio
import signal
import sys

import structlog

from sgldhelper.config import Settings
from sgldhelper.db.engine import Database
from sgldhelper.github.client import GitHubClient
from sgldhelper.github.poller import Poller
from sgldhelper.github.pr_tracker import PRTracker
from sgldhelper.notifications.dispatcher import NotificationDispatcher
from sgldhelper.slack.app import SlackApp
from sgldhelper.slack.channels import ChannelRouter
from sgldhelper.slack.handlers import register_handlers
from sgldhelper.db import queries
from sgldhelper.slack import messages as slack_messages
from sgldhelper.utils.logging_setup import setup_logging
from sgldhelper.ai.client import KimiClient
from sgldhelper.ai.tools import ToolRegistry
from sgldhelper.ai.conversation import ConversationManager
from sgldhelper.ai.classifier import MessageClassifier
from sgldhelper.ai.summaries import SummaryGenerator
from sgldhelper.ci.monitor import CIMonitor
from sgldhelper.ci.auto_merge import AutoMergeManager
from sgldhelper.ci.tracked_pr_summary import TrackedPRSummaryGenerator
from sgldhelper.ci.health_check import PRHealthChecker


async def _run() -> None:
    settings = Settings()  # type: ignore[call-arg]
    setup_logging(settings.log_level, settings.log_dir)
    log = structlog.get_logger()
    log.info("starting", repo=settings.github_repo)

    # --- Initialise components ---
    db = Database(settings.db_path)
    await db.connect()

    gh = GitHubClient(
        settings.github_token,
        settings.github_owner,
        settings.github_repo_name,
    )

    if settings.github_configured:
        log.info("github.enabled")
    else:
        log.warning("github.disabled", reason="GITHUB_TOKEN not set")

    pr_tracker = PRTracker(gh, db, settings)

    slack_app = SlackApp(settings)
    channels = ChannelRouter.from_settings(settings)
    dispatcher = NotificationDispatcher(slack_app, channels, settings, db)

    # --- Initialise CI components ---
    ci_monitor = CIMonitor(gh, db, settings)
    auto_merge = AutoMergeManager(gh, db, settings)

    # Wire CI callbacks to notification dispatcher
    ci_monitor.set_callbacks(
        on_ci_passed=dispatcher.ci.notify_ci_passed,
        on_ci_failed_retrying=dispatcher.ci.notify_ci_failed_retrying,
        on_ci_failed_permanent=dispatcher.ci.notify_ci_failed_permanent,
        on_merge_ready_check=_make_merge_ready_handler(auto_merge),
        on_high_priority_nvidia_passed=_make_hp_nvidia_handler(gh, settings),
    )
    auto_merge.set_callbacks(
        on_countdown=dispatcher.ci.notify_merge_countdown,
        on_complete=dispatcher.ci.notify_merge_complete,
        on_cancelled=dispatcher.ci.notify_merge_cancelled,
    )

    # --- Initialise AI components ---
    kimi = KimiClient(settings)
    tool_registry = ToolRegistry(db, gh, settings)
    tool_registry.set_ci_components(ci_monitor, auto_merge)
    conversation_manager = ConversationManager(kimi, tool_registry, db, settings)
    classifier = MessageClassifier(kimi, db, settings)
    summary_generator = SummaryGenerator(kimi, db, settings)
    log.info("ai.enabled", model=settings.kimi_model, base_url=settings.kimi_base_url)

    # --- Tracked PR summary generator ---
    tracked_pr_summary = TrackedPRSummaryGenerator(kimi, gh, db, settings)
    tracked_pr_summary.set_callback(dispatcher.ci.notify_tracked_pr_summary)

    # --- PR health checker ---
    health_checker = PRHealthChecker(gh, db, ci_monitor, slack_app, channels, settings)

    register_handlers(
        slack_app, db, channels, settings,
        conversation_manager=conversation_manager,
        classifier=classifier,
        summary_generator=summary_generator,
        auto_merge=auto_merge,
    )

    # --- Define poll callbacks ---
    async def poll_prs() -> None:
        from sgldhelper.github.pr_tracker import PREvent

        changes = await pr_tracker.poll()
        for change in changes:
            if change.event == PREvent.OPENED:
                try:
                    await gh.create_issue_comment(
                        change.pr["pr_number"], "/tag-and-rerun-ci"
                    )
                    log.info(
                        "ci.initial_trigger",
                        pr=change.pr["pr_number"],
                        comment="/tag-and-rerun-ci",
                    )
                except Exception as exc:
                    log.error(
                        "ci.initial_trigger_failed",
                        pr=change.pr["pr_number"],
                        error=str(exc),
                    )

            # Phase 7: PR lifecycle — untrack merged/closed PRs
            if change.event in (PREvent.MERGED, PREvent.CLOSED):
                pr_num = change.pr["pr_number"]
                reason = "merged" if change.event == PREvent.MERGED else "closed"
                try:
                    affected_users = await queries.remove_tracked_pr_all_users(db.conn, pr_num)
                    if affected_users:
                        await dispatcher.ci.notify_pr_untracked(pr_num, affected_users, reason)
                        log.info("lifecycle.untracked", pr=pr_num, reason=reason, users=affected_users)
                    # Cancel any pending auto-merge
                    await auto_merge.cancel(pr_num)
                except Exception as exc:
                    log.error("lifecycle.untrack_failed", pr=pr_num, error=str(exc))

            await dispatcher.pr.handle(change, db)

    async def poll_ci() -> None:
        """Poll CI status for all tracked PRs."""
        await ci_monitor.poll_all_tracked_prs()

    async def poll_tracked_pr_summary() -> None:
        """Generate periodic summaries for tracked PRs."""
        await tracked_pr_summary.poll()

    async def poll_diffusion_summary() -> None:
        """Generate and post a periodic diffusion PR summary."""
        summary = await summary_generator.generate_diffusion_summary()
        if summary is None:
            log.debug("diffusion_summary.no_activity")
            return
        await slack_app.post_message(
            channels.pr_channel,
            text=summary,
        )
        log.info("diffusion_summary.posted")

    async def poll_health_check() -> None:
        """Periodic health check of all open diffusion PRs."""
        await health_checker.poll()

    pollers: list[Poller] = []

    # Pollers work with public repos even without GITHUB_TOKEN (lower rate limit)
    pr_poller = Poller("pr", settings.pr_poll_interval, poll_prs)
    ci_poller = Poller("ci", settings.ci_poll_interval, poll_ci)
    tracked_summary_poller = Poller(
        "tracked_pr_summary", settings.tracked_pr_summary_interval, poll_tracked_pr_summary
    )
    diffusion_summary_poller = Poller(
        "diffusion_summary", settings.diffusion_summary_interval, poll_diffusion_summary
    )
    health_check_poller = Poller(
        "health_check", settings.pr_health_check_interval, poll_health_check
    )
    pollers.extend([
        pr_poller, ci_poller, tracked_summary_poller,
        diffusion_summary_poller, health_check_poller,
    ])

    # --- Graceful shutdown ---
    running_tasks: list[asyncio.Task[None]] = []

    def _signal_handler() -> None:
        log.info("shutdown.signal_received")
        for p in pollers:
            p.stop()
        # Cancel all running tasks so blocked HTTP calls are interrupted
        for t in running_tasks:
            t.cancel()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _signal_handler)

    # --- Start ---
    await slack_app.start()

    try:
        async with asyncio.TaskGroup() as tg:
            for p in pollers:
                running_tasks.append(tg.create_task(p.run()))
    except* asyncio.CancelledError:
        pass
    finally:
        await slack_app.stop()
        await gh.close()
        await db.close()
        log.info("shutdown.complete")


def _make_hp_nvidia_handler(gh, settings):
    """Create a callback that pings on GitHub when Nvidia CI passes for high-priority PRs."""
    async def handler(pr_number, user_ids, review_state):
        await gh.create_issue_comment(
            pr_number,
            f"@{settings.ci_high_priority_ping_user} Nvidia CI passed and PR is approved, ready for merge",
        )
    return handler


def _make_merge_ready_handler(auto_merge):
    """Create a merge-ready callback that runs every poll when CI is PASSED.

    This catches all cases: approve after CI pass, PR becomes mergeable
    after conflict resolution, PR tracked when already CI-passed, etc.
    """
    async def handler(pr_number, user_ids, review_state):
        await auto_merge.check_and_start(pr_number, user_ids, review_state)
    return handler


def main() -> None:
    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
