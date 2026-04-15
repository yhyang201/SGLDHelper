"""Block Kit message builders for Slack notifications."""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from sgldhelper.ci.monitor import CIJobResult, CIStatus

from sgldhelper.github.pr_tracker import PRChange, PREvent


def _pr_url(repo: str, pr_number: int) -> str:
    return f"https://github.com/{repo}/pull/{pr_number}"


# ---------------------------------------------------------------------------
# PR messages
# ---------------------------------------------------------------------------

def build_pr_opened(change: PRChange, repo: str) -> dict[str, Any]:
    pr = change.pr
    url = _pr_url(repo, pr["pr_number"])
    return {
        "text": f"New Diffusion PR #{pr['pr_number']}: {pr['title']}",
        "blocks": [
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": (
                        f":new: *New Diffusion PR #{pr['pr_number']}*\n"
                        f"*<{url}|{pr['title']}>*\n"
                        f"Author: `{pr['author']}` | Files: {pr['changed_files']}"
                    ),
                },
            },
            {
                "type": "actions",
                "elements": [
                    {
                        "type": "button",
                        "text": {"type": "plain_text", "text": "View PR"},
                        "url": url,
                    },
                ],
            },
        ],
    }


def build_pr_updated(change: PRChange, repo: str) -> dict[str, Any]:
    pr = change.pr
    url = _pr_url(repo, pr["pr_number"])
    old_sha = change.old_state["head_sha"][:8] if change.old_state else "?"
    new_sha = pr["head_sha"][:8]
    return {
        "text": f"PR #{pr['pr_number']} updated: new commits pushed",
        "blocks": [
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": (
                        f":arrows_counterclockwise: *<{url}|PR #{pr['pr_number']}>* updated\n"
                        f"New commits: `{old_sha}` -> `{new_sha}`"
                    ),
                },
            },
        ],
    }


def build_pr_merged(change: PRChange, repo: str) -> dict[str, Any]:
    pr = change.pr
    url = _pr_url(repo, pr["pr_number"])
    return {
        "text": f"PR #{pr['pr_number']} merged!",
        "blocks": [
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": (
                        f":merged: *<{url}|PR #{pr['pr_number']}>* merged!\n"
                        f"*{pr['title']}* by `{pr['author']}`"
                    ),
                },
            },
        ],
    }


def build_pr_closed(change: PRChange, repo: str) -> dict[str, Any]:
    pr = change.pr
    url = _pr_url(repo, pr["pr_number"])
    return {
        "text": f"PR #{pr['pr_number']} closed",
        "blocks": [
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": (
                        f":no_entry_sign: *<{url}|PR #{pr['pr_number']}>* closed without merge\n"
                        f"*{pr['title']}* by `{pr['author']}`"
                    ),
                },
            },
        ],
    }


PR_MESSAGE_BUILDERS = {
    PREvent.OPENED: build_pr_opened,
    PREvent.MERGED: build_pr_merged,
    PREvent.CLOSED: build_pr_closed,
}


# ---------------------------------------------------------------------------
# AI-related messages
# ---------------------------------------------------------------------------

def _mention_users(user_ids: list[str]) -> str:
    return " ".join(f"<@{uid}>" for uid in user_ids)


# ---------------------------------------------------------------------------
# CI messages
# ---------------------------------------------------------------------------

def build_ci_passed(
    pr_number: int, ci_status: CIStatus, user_ids: list[str], repo: str
) -> dict[str, Any]:
    url = _pr_url(repo, pr_number)
    mentions = _mention_users(user_ids)
    nvidia_count = len(ci_status.nvidia_jobs)
    amd_count = len(ci_status.amd_jobs)
    text = f"CI passed for PR #{pr_number}"
    return {
        "text": text,
        "blocks": [
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": (
                        f":white_check_mark: *<{url}|PR #{pr_number}>* CI 全部通过!\n"
                        f"Nvidia jobs: {nvidia_count} | AMD jobs: {amd_count}\n"
                        f"cc {mentions}"
                    ),
                },
            },
        ],
    }


def build_ci_failed_retrying(
    pr_number: int,
    retryable_jobs: list[CIJobResult],
    user_ids: list[str],
    repo: str,
) -> dict[str, Any]:
    url = _pr_url(repo, pr_number)
    mentions = _mention_users(user_ids)
    job_lines = "\n".join(
        f"• `{j.job_name}` ({j.workflow_name})" for j in retryable_jobs
    )
    text = f"CI failed for PR #{pr_number}, retrying..."
    return {
        "text": text,
        "blocks": [
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": (
                        f":arrows_counterclockwise: *<{url}|PR #{pr_number}>* CI 失败，正在重试:\n"
                        f"{job_lines}\n"
                        f"cc {mentions}"
                    ),
                },
            },
        ],
    }


def build_ci_failed_permanent(
    pr_number: int,
    permanent_jobs: list[CIJobResult],
    user_ids: list[str],
    repo: str,
) -> dict[str, Any]:
    url = _pr_url(repo, pr_number)
    mentions = _mention_users(user_ids)
    job_lines = "\n".join(
        f"• `{j.job_name}` ({j.workflow_name})" for j in permanent_jobs
    )
    text = f"CI permanently failed for PR #{pr_number}"
    return {
        "text": text,
        "blocks": [
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": (
                        f":x: *<{url}|PR #{pr_number}>* CI 多次重试后仍然失败，疑似代码问题:\n"
                        f"{job_lines}\n"
                        f"已停止自动重试。cc {mentions}"
                    ),
                },
            },
        ],
    }


def build_merge_countdown(
    pr_number: int, user_ids: list[str], delay_seconds: int, repo: str
) -> dict[str, Any]:
    url = _pr_url(repo, pr_number)
    mentions = _mention_users(user_ids)
    minutes = delay_seconds // 60
    text = f"PR #{pr_number} will auto-merge in {minutes} minutes"
    return {
        "text": text,
        "blocks": [
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": (
                        f":hourglass_flowing_sand: *<{url}|PR #{pr_number}>* 将在 {minutes} 分钟后自动 squash merge\n"
                        f"说「取消」或「cancel merge」可以取消。cc {mentions}"
                    ),
                },
            },
        ],
    }


def build_merge_complete(
    pr_number: int, user_ids: list[str], repo: str
) -> dict[str, Any]:
    url = _pr_url(repo, pr_number)
    mentions = _mention_users(user_ids)
    text = f"PR #{pr_number} auto-merged"
    return {
        "text": text,
        "blocks": [
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": (
                        f":merged: *<{url}|PR #{pr_number}>* 已自动 squash merge!\n"
                        f"cc {mentions}"
                    ),
                },
            },
        ],
    }


def build_merge_cancelled(
    pr_number: int, user_ids: list[str], repo: str
) -> dict[str, Any]:
    url = _pr_url(repo, pr_number)
    mentions = _mention_users(user_ids)
    text = f"PR #{pr_number} auto-merge cancelled"
    return {
        "text": text,
        "blocks": [
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": (
                        f":no_entry_sign: *<{url}|PR #{pr_number}>* 自动合并已取消\n"
                        f"cc {mentions}"
                    ),
                },
            },
        ],
    }


def build_pr_untracked(
    pr_number: int, user_ids: list[str], reason: str, repo: str
) -> dict[str, Any]:
    url = _pr_url(repo, pr_number)
    mentions = _mention_users(user_ids)
    text = f"PR #{pr_number} untracked: {reason}"
    return {
        "text": text,
        "blocks": [
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": (
                        f":ballot_box_with_check: *<{url}|PR #{pr_number}>* 已自动取消追踪 ({reason})\n"
                        f"cc {mentions}"
                    ),
                },
            },
        ],
    }


def build_tracked_pr_summary(
    pr_number: int, user_ids: list[str], summary: str, repo: str
) -> dict[str, Any]:
    url = _pr_url(repo, pr_number)
    mentions = _mention_users(user_ids)
    text = f"Tracked PR #{pr_number} summary"
    return {
        "text": text,
        "blocks": [
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": (
                        f":bar_chart: *<{url}|PR #{pr_number}>* 追踪更新:\n\n"
                        f"{summary}\n\n"
                        f"cc {mentions}"
                    ),
                },
            },
        ],
    }


# ---------------------------------------------------------------------------
# AI-related messages
# ---------------------------------------------------------------------------

def build_code_quality_report(
    report: str,
    pr_count: int,
    alert_prs: list[dict[str, Any]] | None = None,
    alert_user_ids: list[str] | None = None,
    repo: str = "",
) -> dict[str, Any]:
    """Build a daily code quality report message.

    If *alert_prs* is non-empty and *alert_user_ids* is configured,
    appends a :rotating_light: alert block with @mentions.
    """
    text = f"Daily Code Quality Report ({pr_count} PRs)"
    blocks: list[dict[str, Any]] = [
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": (
                    f":mag: *Daily Code Quality Report*\n"
                    f"_{pr_count} diffusion PR(s) merged today_\n\n"
                    f"{report}"
                ),
            },
        },
    ]

    if alert_prs and alert_user_ids:
        mentions = _mention_users(alert_user_ids)
        alert_lines: list[str] = []
        for p in alert_prs:
            pr_num = p.get("pr", "?")
            score = p.get("score", "?")
            reason = p.get("reason", "")
            url = _pr_url(repo, pr_num) if repo and isinstance(pr_num, int) else f"#{pr_num}"
            link = f"<{url}|PR #{pr_num}>" if repo and isinstance(pr_num, int) else f"PR #{pr_num}"
            line = f"• {link} — score *{score}/10*"
            if reason:
                line += f": {reason}"
            alert_lines.append(line)

        blocks.append({"type": "divider"})
        blocks.append({
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": (
                    f":rotating_light: *Code Quality Alert*\n"
                    f"以下 PR 评分极低，需要关注:\n"
                    f"{''.join(chr(10) + l for l in alert_lines)}\n\n"
                    f"cc {mentions}"
                ),
            },
        })

    return {"text": text, "blocks": blocks}


def build_progress_confirmation(result: dict[str, Any]) -> dict[str, Any]:
    """Build a confirmation message for a detected progress update or blocker."""
    category = result.get("category", "update")
    summary = result.get("summary", "")
    update_id = result.get("update_id", 0)

    if category == "blocker":
        emoji = ":rotating_light:"
        label = "Blocker Detected"
    else:
        emoji = ":clipboard:"
        label = "Progress Update Detected"

    pr_mention = ""
    if result.get("mentioned_pr"):
        pr_mention = f"\nLinked PR: #{result['mentioned_pr']}"

    feature_mention = ""
    if result.get("mentioned_feature"):
        feature_mention = f"\nFeature: {result['mentioned_feature']}"

    return {
        "text": f"{label}: {summary}",
        "blocks": [
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": (
                        f"{emoji} *{label}*\n"
                        f"{summary}{pr_mention}{feature_mention}\n\n"
                        "_Is this correct?_"
                    ),
                },
            },
            {
                "type": "actions",
                "elements": [
                    {
                        "type": "button",
                        "text": {"type": "plain_text", "text": "Confirm"},
                        "action_id": "confirm_update",
                        "value": str(update_id),
                        "style": "primary",
                    },
                    {
                        "type": "button",
                        "text": {"type": "plain_text", "text": "Dismiss"},
                        "action_id": "dismiss_update",
                        "value": str(update_id),
                    },
                ],
            },
        ],
    }
