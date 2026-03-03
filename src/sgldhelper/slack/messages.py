"""Block Kit message builders for Slack notifications."""

from __future__ import annotations

from typing import Any

from sgldhelper.github.ci_analyzer import CIResult, FailureCategory
from sgldhelper.github.issue_tracker import FeatureProgress
from sgldhelper.github.pr_tracker import PRChange, PREvent


def _pr_url(repo: str, pr_number: int) -> str:
    return f"https://github.com/{repo}/pull/{pr_number}"


def _run_url(repo: str, run_id: int) -> str:
    return f"https://github.com/{repo}/actions/runs/{run_id}"


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
                    {
                        "type": "button",
                        "text": {"type": "plain_text", "text": "CI Status"},
                        "action_id": "ci_status",
                        "value": str(pr["pr_number"]),
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
    PREvent.UPDATED: build_pr_updated,
    PREvent.MERGED: build_pr_merged,
    PREvent.CLOSED: build_pr_closed,
}


# ---------------------------------------------------------------------------
# CI messages
# ---------------------------------------------------------------------------

_CATEGORY_EMOJI = {
    FailureCategory.FLAKY: ":game_die:",
    FailureCategory.INFRA: ":building_construction:",
    FailureCategory.PERF_REGRESSION: ":chart_with_downwards_trend:",
    FailureCategory.CODE: ":bug:",
    FailureCategory.UNKNOWN: ":question:",
}


def build_ci_failure(result: CIResult, repo: str) -> dict[str, Any]:
    emoji = _CATEGORY_EMOJI.get(result.failure_category, ":x:")
    cat = result.failure_category.value if result.failure_category else "unknown"
    url = result.html_url or _run_url(repo, result.run_id)
    summary = result.failure_summary or "No details"
    return {
        "text": f"CI failure on PR #{result.pr_number}: {cat}",
        "blocks": [
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": (
                        f"{emoji} *CI Failure* on PR #{result.pr_number}\n"
                        f"Job: `{result.job_name}` | Category: *{cat}*\n"
                        f"```{summary[:500]}```"
                    ),
                },
            },
            {
                "type": "actions",
                "elements": [
                    {
                        "type": "button",
                        "text": {"type": "plain_text", "text": "View Logs"},
                        "url": url,
                    },
                    {
                        "type": "button",
                        "text": {"type": "plain_text", "text": "Rerun CI"},
                        "action_id": "rerun_ci",
                        "value": f"{result.run_id}:{result.pr_number}",
                        "style": "danger",
                    },
                ],
            },
        ],
    }


def build_ci_success(pr_number: int, repo: str) -> dict[str, Any]:
    url = _pr_url(repo, pr_number)
    return {
        "text": f"CI passed on PR #{pr_number}",
        "blocks": [
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f":white_check_mark: *<{url}|PR #{pr_number}>* CI passed!",
                },
            },
        ],
    }


def build_ci_rerun(result: CIResult, auto: bool, repo: str) -> dict[str, Any]:
    url = result.html_url or _run_url(repo, result.run_id)
    trigger = "Auto-rerun" if auto else "Manual rerun"
    cat = result.failure_category.value if result.failure_category else "unknown"
    return {
        "text": f"{trigger} triggered for PR #{result.pr_number}",
        "blocks": [
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": (
                        f":repeat: *{trigger}* triggered for <{url}|run {result.run_id}>\n"
                        f"PR #{result.pr_number} | Reason: {cat}\n"
                        f"Attempt #{result.auto_rerun_count + 1}"
                    ),
                },
            },
        ],
    }


# ---------------------------------------------------------------------------
# Feature progress messages
# ---------------------------------------------------------------------------

def build_feature_progress(progress: FeatureProgress, repo: str) -> dict[str, Any]:
    issue_url = f"https://github.com/{repo}/issues/{progress.issue_number}"
    bar_len = 20
    filled = int(progress.percent / 100 * bar_len)
    bar = "=" * filled + "-" * (bar_len - filled)

    items_text = ""
    for item in progress.items[:15]:  # Show at most 15 items
        check = ":white_check_mark:" if item["state"] == "completed" else ":white_large_square:"
        items_text += f"{check} {item['title']}\n"
    if len(progress.items) > 15:
        items_text += f"_...and {len(progress.items) - 15} more items_\n"

    return {
        "text": f"Feature Progress: {progress.title}",
        "blocks": [
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": (
                        f":bar_chart: *<{issue_url}|{progress.title}>*\n"
                        f"`[{bar}]` {progress.percent:.0f}% "
                        f"({progress.completed}/{progress.total})"
                    ),
                },
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": items_text or "_No items found_",
                },
            },
        ],
    }


def build_daily_digest(
    open_prs: list[dict[str, Any]],
    progress_list: list[FeatureProgress],
    repo: str,
) -> dict[str, Any]:
    pr_lines = ""
    for pr in open_prs[:10]:
        url = _pr_url(repo, pr["pr_number"])
        pr_lines += f"- <{url}|#{pr['pr_number']}> {pr['title']} (`{pr['author']}`)\n"

    feature_lines = ""
    for p in progress_list:
        feature_lines += f"- {p.title}: {p.percent:.0f}% ({p.completed}/{p.total})\n"

    return {
        "text": "Daily Diffusion Digest",
        "blocks": [
            {
                "type": "header",
                "text": {"type": "plain_text", "text": "Daily Diffusion Digest"},
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Open PRs ({len(open_prs)}):*\n{pr_lines or '_None_'}",
                },
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Feature Progress:*\n{feature_lines or '_None_'}",
                },
            },
        ],
    }


# ---------------------------------------------------------------------------
# AI-related messages
# ---------------------------------------------------------------------------

def build_stall_alert(alert: Any) -> dict[str, Any]:
    """Build a stall/nudge notification message."""
    from sgldhelper.ai.stall_detector import StallAlert

    if alert.alert_type == "feature_stall":
        emoji = ":snail:"
        title = "Feature Stall Detected"
    else:
        emoji = ":eyes:"
        title = "Review Needed"

    pr_text = f" (PR #{alert.pr_number})" if alert.pr_number else ""
    return {
        "text": f"{title}: {alert.title}{pr_text}",
        "blocks": [
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": (
                        f"{emoji} *{title}*\n"
                        f"{alert.details}"
                    ),
                },
            },
        ],
    }


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
