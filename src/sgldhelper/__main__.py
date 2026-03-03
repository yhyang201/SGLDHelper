"""Entry point: start all subsystems (pollers + Slack) under asyncio."""

from __future__ import annotations

import asyncio
import signal
import sys

import structlog

from sgldhelper.config import Settings
from sgldhelper.db.engine import Database
from sgldhelper.github.ci_analyzer import CIAnalyzer
from sgldhelper.github.ci_rerunner import CIRerunner
from sgldhelper.github.client import GitHubClient
from sgldhelper.github.issue_tracker import IssueTracker
from sgldhelper.github.poller import Poller
from sgldhelper.github.pr_tracker import PRTracker
from sgldhelper.notifications.dispatcher import NotificationDispatcher
from sgldhelper.slack.app import SlackApp
from sgldhelper.slack.channels import ChannelRouter
from sgldhelper.slack.handlers import register_handlers
from sgldhelper.slack import messages as slack_messages
from sgldhelper.utils.logging_setup import setup_logging


async def _run() -> None:
    settings = Settings()  # type: ignore[call-arg]
    setup_logging(settings.log_level)
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

    pr_tracker = PRTracker(gh, db, settings)
    ci_analyzer = CIAnalyzer(gh, db, settings)
    ci_rerunner = CIRerunner(gh, db, settings)
    issue_tracker = IssueTracker(gh, db, settings)

    slack_app = SlackApp(settings)
    channels = ChannelRouter.from_settings(settings)
    dispatcher = NotificationDispatcher(slack_app, channels, settings)

    # --- Initialise AI components (optional) ---
    conversation_manager = None
    classifier = None
    summary_generator = None
    stall_detector = None

    if settings.ai_enabled:
        from sgldhelper.ai.client import KimiClient
        from sgldhelper.ai.tools import ToolRegistry
        from sgldhelper.ai.conversation import ConversationManager
        from sgldhelper.ai.classifier import MessageClassifier
        from sgldhelper.ai.stall_detector import StallDetector
        from sgldhelper.ai.summaries import SummaryGenerator

        kimi = KimiClient(settings)
        tool_registry = ToolRegistry(db, gh, ci_rerunner, issue_tracker, settings)
        conversation_manager = ConversationManager(kimi, tool_registry, db, settings)
        classifier = MessageClassifier(kimi, db, settings)
        stall_detector = StallDetector(db, gh, settings)
        summary_generator = SummaryGenerator(kimi, db, settings)
        log.info("ai.enabled", model=settings.kimi_model, base_url=settings.kimi_base_url)
    else:
        log.info("ai.disabled")

    register_handlers(
        slack_app, db, ci_rerunner, issue_tracker, channels, settings,
        conversation_manager=conversation_manager,
        classifier=classifier,
        summary_generator=summary_generator,
    )

    # --- Define poll callbacks ---
    async def poll_prs() -> None:
        changes = await pr_tracker.poll()
        for change in changes:
            await dispatcher.pr.handle(change, db)

    async def poll_ci() -> None:
        from sgldhelper.db import queries

        open_prs = await queries.get_open_prs(db.conn)
        for pr in open_prs:
            results = await ci_analyzer.analyze_pr(pr["pr_number"], pr["head_sha"])
            for result in results:
                if result.conclusion == "failure" and result.failure_category:
                    # Notify about failure
                    await dispatcher.ci.handle_failure(result, db)
                    # Attempt auto-rerun
                    rerun_result = await ci_rerunner.auto_rerun(result)
                    if rerun_result.triggered:
                        await dispatcher.ci.handle_rerun(result, rerun_result, db)
                elif result.conclusion == "success":
                    # Check if we previously notified about failure for this PR
                    existing = await queries.get_ci_run(db.conn, result.run_id)
                    if existing and existing.get("conclusion") != "success":
                        await dispatcher.ci.handle_success(result.pr_number, db)

    async def poll_features() -> None:
        progress_list = await issue_tracker.poll()
        for progress in progress_list:
            await dispatcher.feature.handle_progress(progress)

    async def poll_stalls() -> None:
        """Check for stalled features and PRs needing review."""
        if not stall_detector:
            return
        alerts = await stall_detector.check_all()
        for alert in alerts:
            msg = slack_messages.build_stall_alert(alert)
            # Route stall alerts to the features channel
            await slack_app.post_message(
                channels.features_channel,
                text=msg["text"],
                blocks=msg["blocks"],
            )

    pr_poller = Poller("pr", settings.pr_poll_interval, poll_prs)
    ci_poller = Poller("ci", settings.ci_poll_interval, poll_ci)
    feature_poller = Poller("feature", settings.feature_poll_interval, poll_features)

    pollers = [pr_poller, ci_poller, feature_poller]

    if settings.ai_enabled and stall_detector:
        stall_poller = Poller("stall", settings.stall_check_interval, poll_stalls)
        pollers.append(stall_poller)

    # --- Graceful shutdown ---
    shutdown_event = asyncio.Event()

    def _signal_handler() -> None:
        log.info("shutdown.signal_received")
        shutdown_event.set()
        for p in pollers:
            p.stop()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _signal_handler)

    # --- Start ---
    await slack_app.start()

    try:
        async with asyncio.TaskGroup() as tg:
            for p in pollers:
                tg.create_task(p.run())
            tg.create_task(shutdown_event.wait())
    except* asyncio.CancelledError:
        pass
    finally:
        await slack_app.stop()
        await gh.close()
        await db.close()
        log.info("shutdown.complete")


def main() -> None:
    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
