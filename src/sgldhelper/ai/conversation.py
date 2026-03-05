"""Per-thread conversation manager with function calling loop."""

from __future__ import annotations

import contextvars
import json
from typing import Any

import structlog

from sgldhelper.ai.client import KimiClient
from sgldhelper.ai.tools import ToolRegistry
from sgldhelper.config import Settings
from sgldhelper.db import queries
from sgldhelper.db.engine import Database

log = structlog.get_logger()

_current_user: contextvars.ContextVar[str] = contextvars.ContextVar("current_user")

MAX_TOOL_ROUNDS = 10

SYSTEM_PROMPT = (
    "You are SGLDHelper, an AI assistant for the SGLang Diffusion team. "
    "You help the team track PRs, feature progress, and project health.\n\n"
    "Guidelines:\n"
    "- Be concise and direct.\n"
    "- When asked about PRs or features, use the available tools to fetch real data.\n"
    "- If you don't have enough information, say so instead of guessing.\n"
    "- Respond in the same language the user uses (Chinese or English).\n\n"
    "## Slack mrkdwn formatting (MUST follow)\n"
    "You are outputting to Slack, NOT standard Markdown. The rules are different:\n"
    "- Bold: *text* (single asterisk, NOT double **)\n"
    "- Italic: _text_ (underscore)\n"
    "- Strikethrough: ~text~\n"
    "- Inline code: `code`\n"
    "- Code block: ```code```\n"
    "- Bullet list: use • or - at line start\n"
    "- Links: <https://url|display text> (angle brackets, NOT [text](url))\n"
    "- DO NOT use Markdown tables (| col | col |) — Slack does not render them. "
    "Use bullet lists or plain text instead.\n"
    "- DO NOT use Markdown headings (# ## ###) — use *bold text* on its own line instead.\n"
    "- Emoji: :emoji_name: (e.g. :white_check_mark:, :fire:, :warning:)\n\n"
    "## Code Review\n"
    "When a user asks you to review a PR's code, call `review_pr_code` with the PR number "
    "to fetch its diff. Then provide a structured review covering:\n"
    "1. *Changes summary* — what the PR does at a high level\n"
    "2. *Potential bugs* — logic errors, edge cases, off-by-one, null handling\n"
    "3. *Code style* — naming, readability, consistency with the codebase\n"
    "4. *Performance considerations* — unnecessary allocations, N+1 queries, hot paths\n"
    "If the diff was truncated, mention that you only reviewed a partial diff.\n\n"
    "## PR Tracking\n"
    "When a user asks to track a PR:\n"
    "1. Call `update_tracked_prs` to add it\n"
    "2. Call `get_pr_details` to show the PR overview\n"
    "3. Call `get_ci_status` to show CI status\n"
    "4. Call `get_pr_reviews` to check review state\n"
    "5. If there's no `run-ci` label, proactively ask the user if they want to trigger CI.\n\n"
    "## CI Management\n"
    "- Use `get_ci_status` to check CI (Nvidia + AMD workflows)\n"
    "- Use `trigger_ci` (requires confirmation) to trigger CI via comment\n"
    "- Use `cancel_auto_merge` (requires confirmation) to cancel pending auto-merges\n"
    "- CI has two workflows: Nvidia (GPU tests) and AMD (GPU tests), controlled by `run-ci` label\n\n"
    "## Merge\n"
    "- Use `merge_pr` (requires confirmation) to merge a PR when a user asks to merge.\n"
    "- Default merge method is squash. Before merging, check CI and review status first.\n"
    "- When the user says 'merge吧', 'merge it', '合并' etc., treat it as an explicit merge request.\n"
    "- Do NOT tell the user to go to GitHub to merge manually — you have the ability to merge directly.\n"
    "- When the user asks to 'merge可以merge的PR' or 'merge all ready PRs', "
    "call `get_merge_ready_prs` first (one tool call), then merge each one. "
    "Do NOT loop through PRs individually with get_ci_status + get_pr_reviews — that is too slow.\n"
)


class ConversationManager:
    """Manage per-thread conversations with Kimi K2.5 and function calling."""

    def __init__(
        self,
        client: KimiClient,
        tools: ToolRegistry,
        db: Database,
        settings: Settings,
    ) -> None:
        self._client = client
        self._tools = tools
        self._db = db
        self._settings = settings
        # Pending confirmations: {thread_ts: {tool_name, arguments}}
        self._pending_confirmations: dict[str, dict[str, Any]] = {}

    @staticmethod
    def _sanitise_tool_calls(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Patch up history so every tool_call id has a matching tool response.

        If an assistant message contains tool_calls whose ids never appear in a
        subsequent ``role=tool`` message, we inject a synthetic tool response so
        the API doesn't reject the request.  This handles cases like:
        - Bot crashed mid-tool-execution
        - Confirmation timed out / was never answered
        - Old bug where batch tool_calls were partially handled
        """
        # Collect all tool response ids present in the conversation
        responded_ids: set[str] = set()
        for msg in messages:
            if msg.get("role") == "tool" and msg.get("tool_call_id"):
                responded_ids.add(msg["tool_call_id"])

        # Walk through and patch orphaned tool_calls
        patched: list[dict[str, Any]] = []
        for msg in messages:
            patched.append(msg)
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                for tc in msg["tool_calls"]:
                    tc_id = tc.get("id", "")
                    if tc_id and tc_id not in responded_ids:
                        fn_name = tc.get("function", {}).get("name", "unknown")
                        patched.append({
                            "role": "tool",
                            "tool_call_id": tc_id,
                            "name": fn_name,
                            "content": json.dumps({
                                "error": "Tool execution was interrupted. "
                                "Please retry if needed."
                            }),
                        })
                        responded_ids.add(tc_id)
        return patched

    async def handle_mention(
        self,
        text: str,
        thread_ts: str,
        channel_id: str,
        user_id: str,
    ) -> str:
        """Process an @mention message and return the bot's response text.

        Runs the full function-calling loop: send message → execute tools → reply.
        """
        # Check for pending confirmation responses
        if thread_ts in self._pending_confirmations:
            return await self._handle_confirmation(text, thread_ts, channel_id)

        # Load conversation history from DB
        messages = await self._build_messages(thread_ts, text, user_id)

        # Run the function-calling loop with user context
        token = _current_user.set(user_id)
        try:
            response_text = await self._run_tool_loop(messages, thread_ts, channel_id)
        finally:
            _current_user.reset(token)

        return response_text

    async def _build_messages(
        self, thread_ts: str, new_text: str, user_id: str
    ) -> list[dict[str, Any]]:
        """Build message list from system prompt + DB history + new user message."""
        # Load user memory and inject into system prompt
        system_content = SYSTEM_PROMPT
        mem = await queries.get_user_memory(self._db.conn, user_id)
        if mem:
            parts: list[str] = []
            tracked = json.loads(mem["tracked_prs"])
            if tracked:
                parts.append(f"- Tracking PRs: {', '.join(f'#{p}' for p in tracked)}")
            focus = json.loads(mem["focus_areas"])
            if focus:
                parts.append(f"- Focus areas: {', '.join(focus)}")
            if mem["notes"]:
                parts.append(f"- Notes: {mem['notes']}")
            prefs = json.loads(mem["preferences"])
            if prefs:
                parts.append(f"- Preferences: {json.dumps(prefs, ensure_ascii=False)}")
            if parts:
                user_section = "\n".join(parts)
                system_content += (
                    f"\n## About this user (Slack ID: {user_id}):\n{user_section}\n"
                )

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_content}
        ]

        # Load prior conversation turns from DB
        history = await queries.get_conversation_history(self._db.conn, thread_ts)
        for row in history:
            msg: dict[str, Any] = {"role": row["role"]}
            if row["content"]:
                msg["content"] = row["content"]
            if row.get("reasoning_content"):
                msg["reasoning_content"] = row["reasoning_content"]
            if row["tool_calls"]:
                msg["tool_calls"] = json.loads(row["tool_calls"])
            if row["tool_call_id"]:
                msg["tool_call_id"] = row["tool_call_id"]
            if row["name"]:
                msg["name"] = row["name"]
            messages.append(msg)

        # Sanitise: ensure every assistant tool_call has a matching tool response.
        # Orphaned tool_calls (e.g. from a crash or confirmation timeout) would
        # cause the API to reject the request with a 400 error.
        messages = self._sanitise_tool_calls(messages)

        # Add new user message
        messages.append({"role": "user", "content": new_text})

        return messages

    async def _run_tool_loop(
        self,
        messages: list[dict[str, Any]],
        thread_ts: str,
        channel_id: str,
    ) -> str:
        """Execute the function-calling loop, up to MAX_TOOL_ROUNDS."""
        tool_schemas = self._tools.get_schemas()
        total_in, total_out = 0, 0

        for _round in range(MAX_TOOL_ROUNDS):
            response = await self._client.chat(
                messages=messages,
                tools=tool_schemas if tool_schemas else None,
                thinking=True,
            )

            tokens_in, tokens_out = self._client.extract_usage(response)
            total_in += tokens_in
            total_out += tokens_out

            choice = response.choices[0]
            message = choice.message

            if choice.finish_reason == "tool_calls" or message.tool_calls:
                # Process tool calls
                # Capture reasoning_content from thinking mode (Kimi K2.5 requires
                # it to be replayed in subsequent requests)
                rc = getattr(message, "reasoning_content", None)
                reasoning_content = rc if isinstance(rc, str) else None

                tool_calls_json = json.dumps([
                    {"id": tc.id, "type": tc.type,
                     "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                    for tc in message.tool_calls
                ])
                await queries.save_conversation_message(
                    self._db.conn, thread_ts, channel_id,
                    role="assistant", content=message.content,
                    reasoning_content=reasoning_content,
                    tool_calls=tool_calls_json,
                )

                # Add assistant message to context (include reasoning_content
                # so the next API call won't reject the message)
                assistant_msg: dict[str, Any] = {"role": "assistant"}
                if message.content:
                    assistant_msg["content"] = message.content
                if reasoning_content:
                    assistant_msg["reasoning_content"] = reasoning_content
                assistant_msg["tool_calls"] = [
                    {"id": tc.id, "type": tc.type,
                     "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                    for tc in message.tool_calls
                ]
                messages.append(assistant_msg)

                # Separate tool calls into auto-execute and confirmation-needed
                auto_calls = []
                confirm_calls = []
                for tc in message.tool_calls:
                    if self._tools.needs_confirmation(tc.function.name):
                        confirm_calls.append(tc)
                    else:
                        auto_calls.append(tc)

                # Execute all auto-execute tools first
                for tc in auto_calls:
                    tool_name = tc.function.name
                    log.info("tool.executing", tool=tool_name, round=_round)
                    result = await self._tools.execute(tool_name, tc.function.arguments)

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result,
                    })
                    await queries.save_conversation_message(
                        self._db.conn, thread_ts, channel_id,
                        role="tool", content=result,
                        tool_call_id=tc.id, name=tool_name,
                    )

                # If there are confirmation-needed tools, pause and ask
                if confirm_calls:
                    self._pending_confirmations[thread_ts] = {
                        "pending_tools": [
                            {"tool_call_id": tc.id, "tool_name": tc.function.name,
                             "arguments": tc.function.arguments}
                            for tc in confirm_calls
                        ],
                        "messages": messages,
                    }
                    # Save the user message before returning
                    for msg in reversed(messages):
                        if msg.get("role") == "user" and msg.get("content"):
                            await queries.save_conversation_message(
                                self._db.conn, thread_ts, channel_id,
                                role="user", content=msg["content"],
                            )
                            break

                    # Build confirmation prompt listing all pending tools
                    lines = []
                    for tc in confirm_calls:
                        lines.append(f"• *{tc.function.name}*: `{tc.function.arguments}`")
                    confirmation_text = (
                        "I need to run the following:\n"
                        + "\n".join(lines)
                        + "\n\nReply *yes* to confirm or *no* to cancel."
                    )
                    await queries.log_llm_usage(
                        self._db.conn, "conversation", self._settings.kimi_model,
                        total_in, total_out,
                    )
                    return confirmation_text

                # Continue loop to get LLM response after tool results
                continue

            # finish_reason == "stop" → final response
            reply = message.content or "I wasn't able to generate a response."

            # Save user message + assistant reply
            # Find the user message (last user message before tools)
            for msg in reversed(messages):
                if msg["role"] == "user" and msg.get("content"):
                    await queries.save_conversation_message(
                        self._db.conn, thread_ts, channel_id,
                        role="user", content=msg["content"],
                    )
                    break

            await queries.save_conversation_message(
                self._db.conn, thread_ts, channel_id,
                role="assistant", content=reply,
                tokens_in=total_in, tokens_out=total_out,
            )

            # Track LLM usage
            await queries.log_llm_usage(
                self._db.conn, "conversation", self._settings.kimi_model,
                total_in, total_out,
            )

            return reply

        # Exceeded max rounds
        return "I've reached the maximum number of tool calls. Please try a simpler question."

    async def _handle_confirmation(
        self, text: str, thread_ts: str, channel_id: str
    ) -> str:
        """Handle a user's yes/no response to a tool confirmation request."""
        pending = self._pending_confirmations.pop(thread_ts)
        normalized = text.strip().lower()

        if normalized in ("yes", "y", "确认", "是"):
            messages = pending["messages"]

            # Execute all pending confirmation tools
            for tool_info in pending["pending_tools"]:
                result = await self._tools.execute(
                    tool_info["tool_name"], tool_info["arguments"]
                )
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_info["tool_call_id"],
                    "content": result,
                })
                await queries.save_conversation_message(
                    self._db.conn, thread_ts, channel_id,
                    role="tool", content=result,
                    tool_call_id=tool_info["tool_call_id"],
                    name=tool_info["tool_name"],
                )

            # Continue the conversation loop
            return await self._run_tool_loop(messages, thread_ts, channel_id)

        # User cancelled — still need to send tool responses so the
        # conversation history stays valid for the next API call.
        messages = pending["messages"]
        for tool_info in pending["pending_tools"]:
            cancelled_result = json.dumps({"cancelled": True, "reason": "User declined"})
            messages.append({
                "role": "tool",
                "tool_call_id": tool_info["tool_call_id"],
                "content": cancelled_result,
            })
            await queries.save_conversation_message(
                self._db.conn, thread_ts, channel_id,
                role="tool", content=cancelled_result,
                tool_call_id=tool_info["tool_call_id"],
                name=tool_info["tool_name"],
            )

        return "Cancelled. The operation was not executed."
