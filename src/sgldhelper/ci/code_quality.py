"""Daily code quality report for merged diffusion PRs."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Awaitable

import structlog

from sgldhelper.ai.client import KimiClient
from sgldhelper.config import Settings
from sgldhelper.db import queries
from sgldhelper.db.engine import Database
from sgldhelper.github.client import GitHubClient

log = structlog.get_logger()

_MAX_DIFF_CHARS = 12000  # truncate large diffs to stay within token limits

_CODE_QUALITY_SYSTEM_PROMPT = (
    "You are a senior code quality reviewer for the SGLang Diffusion (multimodal_gen) "
    "framework. You review merged PR diffs and assess their impact on overall code quality.\n\n"
    "IMPORTANT — you are outputting to Slack mrkdwn, NOT standard Markdown:\n"
    "- Bold: *text* (single asterisk, NOT double **)\n"
    "- Italic: _text_\n"
    "- Inline code: `code`\n"
    "- Code block: ```code```\n"
    "- Bullet list: use • or - at line start\n"
    "- Links: <https://url|display text> (NOT [text](url))\n"
    "- DO NOT use Markdown tables (| col |) — Slack cannot render them\n"
    "- DO NOT use Markdown headings (# ##) — use *bold text* on its own line\n\n"
    "For each PR, provide:\n"
    "1. A one-line summary of what it does\n"
    "2. Code quality observations (code smells, design issues, good practices)\n"
    "   Focus on: naming, duplication, complexity, error handling, test coverage gaps, "
    "type safety, magic numbers, dead code, tight coupling\n"
    "3. A score from 0-10 where:\n"
    "   - 0-3: Harmful to code quality (introduces tech debt, bad patterns)\n"
    "   - 4-5: Neutral or minor issues\n"
    "   - 6-7: Acceptable, minor improvements possible\n"
    "   - 8-9: Good quality, clean code\n"
    "   - 10: Exemplary, improves the codebase\n\n"
    "After individual PR reviews, provide:\n"
    "- *Daily Overall Score*: weighted average (0-10) of all PR scores\n"
    "- *Key Takeaways*: 1-3 bullet points on today's overall code quality trend\n\n"
    "Be concise. Each PR review should be 3-5 lines max.\n"
    "Use Chinese for commentary, English for technical terms.\n\n"
    "CRITICAL: At the very end of your response, on its own line, output a machine-readable "
    "JSON object (and NOTHING else on that line) in this exact format:\n"
    '<!--SCORES:{"overall":N,"prs":[{"pr":PR_NUMBER,"score":N,"reason":"one-line reason if score<=3"},...]}-->\n'
    "This line will be stripped before display. It MUST be the last line.\n"
)


@dataclass
class QualityReport:
    """Parsed result of a code quality report."""

    display_text: str
    overall_score: float | None = None
    pr_scores: list[dict[str, Any]] = field(default_factory=list)
    alert_prs: list[dict[str, Any]] = field(default_factory=list)


def _parse_report(raw: str, threshold: int) -> QualityReport:
    """Parse LLM output, extracting the SCORES JSON and the display text."""
    # Look for the <!--SCORES:{...}--> line
    match = re.search(r"<!--SCORES:(\{.*?\})-->", raw)
    if not match:
        return QualityReport(display_text=raw.strip())

    display_text = raw[:match.start()].rstrip("\n ")
    try:
        data = json.loads(match.group(1))
    except json.JSONDecodeError:
        log.warning("code_quality.scores_json_parse_failed", raw=match.group(1)[:200])
        return QualityReport(display_text=display_text)

    overall = data.get("overall")
    pr_scores = data.get("prs", [])
    alert_prs = [p for p in pr_scores if isinstance(p.get("score"), (int, float)) and p["score"] <= threshold]

    return QualityReport(
        display_text=display_text,
        overall_score=overall,
        pr_scores=pr_scores,
        alert_prs=alert_prs,
    )


class CodeQualityReporter:
    """Generate daily code quality reports for merged diffusion PRs."""

    def __init__(
        self,
        kimi: KimiClient,
        gh: GitHubClient,
        db: Database,
        settings: Settings,
    ) -> None:
        self._kimi = kimi
        self._gh = gh
        self._db = db
        self._settings = settings
        self._on_report: Callable[..., Awaitable[Any]] | None = None
        self._last_report_date: str | None = None

    def set_callback(self, on_report: Callable[..., Awaitable[Any]]) -> None:
        self._on_report = on_report

    async def poll(self) -> None:
        """Check if it's time to generate the daily report.

        Runs once per day — skips if a report was already generated today.
        """
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if self._last_report_date == today:
            log.debug("code_quality.already_reported", date=today)
            return

        merged_prs = await queries.get_merged_diffusion_prs_today(self._db.conn)
        if not merged_prs:
            log.debug("code_quality.no_merged_prs", date=today)
            self._last_report_date = today
            return

        try:
            report = await self._generate_report(merged_prs)
        except Exception:
            log.exception("code_quality.generation_failed")
            return

        if report and self._on_report:
            await self._on_report(
                report.display_text, len(merged_prs), report.alert_prs,
            )

        self._last_report_date = today
        log.info(
            "code_quality.report_posted",
            date=today,
            pr_count=len(merged_prs),
            overall_score=report.overall_score if report else None,
            alerts=len(report.alert_prs) if report else 0,
        )

    async def _generate_report(
        self, merged_prs: list[dict[str, Any]]
    ) -> QualityReport | None:
        """Fetch diffs and generate a quality report via LLM."""
        pr_contexts: list[str] = []

        for pr in merged_prs:
            pr_number = pr["pr_number"]
            try:
                diff = await self._gh.get_pull_diff(pr_number)
            except Exception:
                log.warning("code_quality.diff_fetch_failed", pr=pr_number)
                diff = "(diff unavailable)"

            # Truncate large diffs
            if len(diff) > _MAX_DIFF_CHARS:
                diff = diff[:_MAX_DIFF_CHARS] + "\n... (truncated)"

            pr_contexts.append(
                f"--- PR #{pr_number}: {pr['title']} (by {pr['author']}) ---\n"
                f"Files changed: {pr.get('changed_files', '?')}\n"
                f"Diff:\n```\n{diff}\n```"
            )

        context = "\n\n".join(pr_contexts)

        messages = [
            {"role": "system", "content": _CODE_QUALITY_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Review these {len(merged_prs)} diffusion PRs merged today "
                    f"and generate a code quality report:\n\n{context}"
                ),
            },
        ]

        response = await self._kimi.chat(messages, thinking=True)
        tokens_in, tokens_out = self._kimi.extract_usage(response)

        await queries.log_llm_usage(
            self._db.conn, "code_quality_report", self._settings.kimi_model,
            tokens_in, tokens_out,
        )

        raw = response.choices[0].message.content
        if not raw:
            return None
        return _parse_report(raw, self._settings.code_quality_alert_threshold)
