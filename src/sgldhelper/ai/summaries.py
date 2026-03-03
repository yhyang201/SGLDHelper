"""Standup and weekly summary generation using K2.5 instant mode."""

from __future__ import annotations

import json
from typing import Any

import structlog

from sgldhelper.ai.client import KimiClient
from sgldhelper.config import Settings
from sgldhelper.db import queries
from sgldhelper.db.engine import Database

log = structlog.get_logger()

STANDUP_SYSTEM_PROMPT = (
    "You are a standup summary generator for the SGLang Diffusion team. "
    "Given recent activity data (PRs, CI runs, feature items, blockers), "
    "generate a concise daily standup summary.\n\n"
    "Format:\n"
    "- Use Slack mrkdwn formatting (bold with *, code with `)\n"
    "- Group by: PRs, CI, Features, Blockers\n"
    "- Keep it under 500 words\n"
    "- Highlight items needing attention\n"
    "- Respond in the same language as the data (English or Chinese)\n"
)


class SummaryGenerator:
    """Generate standup and periodic summaries from recent activity."""

    def __init__(
        self,
        client: KimiClient,
        db: Database,
        settings: Settings,
    ) -> None:
        self._client = client
        self._db = db
        self._settings = settings

    async def generate_standup(self, since_hours: int = 24) -> str:
        """Generate a daily standup summary from recent activity."""
        activity = await queries.get_recent_activity(self._db.conn, since_hours)
        blockers = await queries.get_confirmed_blockers(self._db.conn, since_hours)

        # Build context for the LLM
        context_parts: list[str] = []

        prs = activity.get("prs", [])
        if prs:
            pr_lines = []
            for pr in prs[:15]:
                pr_lines.append(
                    f"- PR #{pr['pr_number']}: {pr['title']} "
                    f"(by {pr['author']}, state={pr['state']})"
                )
            context_parts.append(f"**Recent PRs ({len(prs)}):**\n" + "\n".join(pr_lines))

        ci_runs = activity.get("ci_runs", [])
        if ci_runs:
            failures = [r for r in ci_runs if r.get("conclusion") == "failure"]
            successes = [r for r in ci_runs if r.get("conclusion") == "success"]
            context_parts.append(
                f"**CI Runs:** {len(ci_runs)} total, "
                f"{len(failures)} failures, {len(successes)} successes"
            )
            if failures:
                fail_lines = []
                for r in failures[:5]:
                    fail_lines.append(
                        f"- Run {r['run_id']} for PR #{r['pr_number']}: "
                        f"{r.get('failure_category', 'unknown')} - "
                        f"{(r.get('failure_summary') or '')[:100]}"
                    )
                context_parts.append("**CI Failures:**\n" + "\n".join(fail_lines))

        open_features = activity.get("open_features", [])
        if open_features:
            context_parts.append(f"**Open Feature Items:** {len(open_features)}")

        if blockers:
            blocker_lines = []
            for b in blockers:
                data = json.loads(b.get("extracted_data", "{}"))
                blocker_lines.append(
                    f"- {data.get('summary', 'Unknown blocker')} "
                    f"(reported by <@{b['user_id']}>)"
                )
            context_parts.append("**Active Blockers:**\n" + "\n".join(blocker_lines))

        if not context_parts:
            return "No significant activity in the last 24 hours. All quiet!"

        context = "\n\n".join(context_parts)

        messages = [
            {"role": "system", "content": STANDUP_SYSTEM_PROMPT},
            {"role": "user", "content": f"Generate a standup summary from this data:\n\n{context}"},
        ]

        response = await self._client.chat(messages, thinking=False)
        tokens_in, tokens_out = self._client.extract_usage(response)

        await queries.log_llm_usage(
            self._db.conn, "standup_summary", self._settings.kimi_model,
            tokens_in, tokens_out,
        )

        return response.choices[0].message.content or "Unable to generate summary."
