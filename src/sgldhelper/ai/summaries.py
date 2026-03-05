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

_SLACK_FORMAT_RULES = (
    "IMPORTANT — you are outputting to Slack mrkdwn, NOT standard Markdown:\n"
    "- Bold: *text* (single asterisk, NOT double **)\n"
    "- Italic: _text_\n"
    "- Inline code: `code`\n"
    "- Code block: ```code```\n"
    "- Bullet list: use • or - at line start\n"
    "- Links: <https://url|display text> (NOT [text](url))\n"
    "- DO NOT use Markdown tables (| col |) — Slack cannot render them\n"
    "- DO NOT use Markdown headings (# ##) — use *bold text* on its own line\n"
)

DIFFUSION_SUMMARY_SYSTEM_PROMPT = (
    "You are a summary generator for the SGLang Diffusion (multimodal_gen) team. "
    "Given recent PR activity data, generate a concise status update.\n\n"
    f"{_SLACK_FORMAT_RULES}\n"
    "Other rules:\n"
    "- Each PR has a status tag: [NEW] = newly opened, [MERGED] = merged, [CLOSED] = closed\n"
    "- You MUST preserve each PR's status tag clearly in the output "
    "(e.g. use emoji: :new: for NEW, :merged: for MERGED, :no_entry_sign: for CLOSED)\n"
    "- Group by status: New PRs, Merged, Closed\n"
    "- Keep it under 300 words\n"
    "- Add a brief one-line comment for each PR explaining its significance\n"
    "- Respond in the same language as the data (English or Chinese)\n"
)

STANDUP_SYSTEM_PROMPT = (
    "You are a standup summary generator for the SGLang Diffusion team. "
    "Given recent activity data (PRs, blockers), "
    "generate a concise daily standup summary.\n\n"
    f"{_SLACK_FORMAT_RULES}\n"
    "Other rules:\n"
    "- Group by: PRs, Blockers\n"
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

    async def generate_diffusion_summary(self, since_hours: int = 2) -> str | None:
        """Generate a periodic diffusion PR summary. Returns None if no activity."""
        data = await queries.get_diffusion_pr_summary(self._db.conn, since_hours)

        opened = data.get("opened", [])
        merged = data.get("merged", [])
        closed = data.get("closed", [])

        if not opened and not merged and not closed:
            return None  # no activity — silently skip

        # Tag each PR with its status change so the LLM preserves it
        pr_lines: list[str] = []
        for pr in opened[:10]:
            pr_lines.append(
                f"- [NEW] PR #{pr['pr_number']}: {pr['title']} (by {pr['author']})"
            )
        for pr in merged[:10]:
            pr_lines.append(
                f"- [MERGED] PR #{pr['pr_number']}: {pr['title']} (by {pr['author']})"
            )
        for pr in closed[:10]:
            pr_lines.append(
                f"- [CLOSED] PR #{pr['pr_number']}: {pr['title']} (by {pr['author']})"
            )

        context = "\n".join(pr_lines)

        messages = [
            {"role": "system", "content": DIFFUSION_SUMMARY_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Generate a diffusion PR status update from the last {since_hours} hours:\n\n"
                    f"{context}"
                ),
            },
        ]

        response = await self._client.chat(messages, thinking=False)
        tokens_in, tokens_out = self._client.extract_usage(response)

        await queries.log_llm_usage(
            self._db.conn, "diffusion_summary", self._settings.kimi_model,
            tokens_in, tokens_out,
        )

        return response.choices[0].message.content or None
