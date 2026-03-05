"""Periodic health check for all open diffusion PRs.

Reports three categories every 2h:
1. Merge-ready: CI passed + approved + mergeable (action needed: merge)
2. Needs review: CI passed (nvidia+amd) but no approval (action needed: review)
3. CI stalled: approved but CI not run / failed with no pending retry (action needed: trigger CI)

Always posts a report (even if all categories are empty — "all clear").
"""

from __future__ import annotations

import json
from typing import Any

import structlog

from sgldhelper.ci.monitor import CIMonitor, CIOverallStatus
from sgldhelper.config import Settings
from sgldhelper.db import queries
from sgldhelper.db.engine import Database
from sgldhelper.github.client import GitHubClient
from sgldhelper.slack.app import SlackApp
from sgldhelper.slack.channels import ChannelRouter

log = structlog.get_logger()


def _pr_url(repo: str, pr_number: int) -> str:
    return f"https://github.com/{repo}/pull/{pr_number}"


class PRHealthChecker:
    """Batch-check all open diffusion PRs and post a health report."""

    def __init__(
        self,
        gh: GitHubClient,
        db: Database,
        ci_monitor: CIMonitor,
        slack_app: SlackApp,
        channels: ChannelRouter,
        settings: Settings,
    ) -> None:
        self._gh = gh
        self._db = db
        self._ci_monitor = ci_monitor
        self._slack = slack_app
        self._channels = channels
        self._settings = settings

    async def poll(self) -> None:
        """Run the health check and post results to Slack."""
        open_prs = await queries.get_open_prs(self._db.conn)
        if not open_prs:
            await self._post_report([], [], [])
            return

        merge_ready: list[dict[str, Any]] = []
        needs_review: list[dict[str, Any]] = []
        ci_stalled: list[dict[str, Any]] = []

        for db_pr in open_prs:
            pr_num = db_pr["pr_number"]
            try:
                pr_data = await self._gh.get_pull(pr_num)
            except Exception:
                continue

            if pr_data["state"] != "open":
                continue

            head_sha = pr_data["head"]["sha"]
            title = pr_data["title"]
            author = pr_data["user"]["login"]

            # Check CI
            ci_status = await self._ci_monitor.check_pr_ci(
                pr_num, head_sha, pr_data=pr_data,
            )

            # Check reviews
            reviews = await self._gh.get_pull_reviews(pr_num)
            has_approval = any(r.get("state") == "APPROVED" for r in reviews)

            ci_passed = ci_status.overall == CIOverallStatus.PASSED
            ci_failed = ci_status.overall == CIOverallStatus.FAILED
            ci_none = ci_status.overall == CIOverallStatus.NO_CI
            mergeable = bool(pr_data.get("mergeable"))

            pr_info = {
                "pr_number": pr_num,
                "title": title,
                "author": author,
            }

            if ci_passed and has_approval and mergeable:
                merge_ready.append(pr_info)
            elif ci_passed and not has_approval:
                needs_review.append(pr_info)
            elif has_approval and (ci_none or (ci_failed and ci_status.all_runs_completed)):
                # Distinguish "CI not started" (no run-ci label) from "CI failed"
                if not ci_status.has_run_ci_label:
                    pr_info["ci_status"] = "not_started"
                else:
                    pr_info["ci_status"] = ci_status.overall.value
                ci_stalled.append(pr_info)

        await self._post_report(merge_ready, needs_review, ci_stalled)

    async def _post_report(
        self,
        merge_ready: list[dict[str, Any]],
        needs_review: list[dict[str, Any]],
        ci_stalled: list[dict[str, Any]],
    ) -> None:
        repo = self._settings.github_repo
        sections: list[str] = []

        # 1. Merge-ready
        if merge_ready:
            lines = [f"• <{_pr_url(repo, p['pr_number'])}|#{p['pr_number']}> {p['title']} (`{p['author']}`)"
                     for p in merge_ready]
            sections.append(
                f":white_check_mark: *可以 Merge ({len(merge_ready)})*\n" + "\n".join(lines)
            )
        else:
            sections.append(":white_check_mark: *可以 Merge*\n无")

        # 2. Needs review
        if needs_review:
            lines = [f"• <{_pr_url(repo, p['pr_number'])}|#{p['pr_number']}> {p['title']} (`{p['author']}`)"
                     for p in needs_review]
            sections.append(
                f":eyes: *CI 通过，等待 Review ({len(needs_review)})*\n" + "\n".join(lines)
            )
        else:
            sections.append(":eyes: *CI 通过，等待 Review*\n无")

        # 3. CI stalled
        if ci_stalled:
            lines = []
            for p in ci_stalled:
                status = p.get("ci_status", "unknown")
                if status == "not_started":
                    label = "未启动（无 run-ci label）"
                elif status == "no_ci":
                    label = "未运行"
                else:
                    label = "失败未重跑"
                lines.append(
                    f"• <{_pr_url(repo, p['pr_number'])}|#{p['pr_number']}> {p['title']} "
                    f"(`{p['author']}`) — CI {label}"
                )
            sections.append(
                f":warning: *已 Approve，CI 需处理 ({len(ci_stalled)})*\n" + "\n".join(lines)
            )
        else:
            sections.append(":warning: *已 Approve，CI 需处理*\n无")

        text = ":bar_chart: *Diffusion PR 健康检查*\n\n" + "\n\n".join(sections)

        await self._slack.post_message_with_context(
            self._channels.ci_channel,
            text=text,
            db_conn=self._db.conn,
        )
        log.info(
            "health_check.posted",
            merge_ready=len(merge_ready),
            needs_review=len(needs_review),
            ci_stalled=len(ci_stalled),
        )
