"""Per-thread conversation manager with function calling loop."""

from __future__ import annotations

import json
from typing import Any

import structlog

from sgldhelper.ai.client import KimiClient
from sgldhelper.ai.tools import ToolRegistry
from sgldhelper.config import Settings
from sgldhelper.db import queries
from sgldhelper.db.engine import Database

log = structlog.get_logger()

MAX_TOOL_ROUNDS = 10

SYSTEM_PROMPT = (
    "You are SGLDHelper, an AI assistant for the SGLang Diffusion team. "
    "You help the team track PRs, CI status, feature progress, and project health.\n\n"
    "Guidelines:\n"
    "- Be concise and direct. Use Slack-friendly formatting (bold, code blocks).\n"
    "- When asked about PRs, CI, or features, use the available tools to fetch real data.\n"
    "- For write operations (like rerunning CI), always confirm with the user first.\n"
    "- If you don't have enough information, say so instead of guessing.\n"
    "- Respond in the same language the user uses (Chinese or English).\n"
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

        # Run the function-calling loop
        response_text = await self._run_tool_loop(messages, thread_ts, channel_id)

        return response_text

    async def _build_messages(
        self, thread_ts: str, new_text: str, user_id: str
    ) -> list[dict[str, Any]]:
        """Build message list from system prompt + DB history + new user message."""
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]

        # Load prior conversation turns from DB
        history = await queries.get_conversation_history(self._db.conn, thread_ts)
        for row in history:
            msg: dict[str, Any] = {"role": row["role"]}
            if row["content"]:
                msg["content"] = row["content"]
            if row["tool_calls"]:
                msg["tool_calls"] = json.loads(row["tool_calls"])
            if row["tool_call_id"]:
                msg["tool_call_id"] = row["tool_call_id"]
            if row["name"]:
                msg["name"] = row["name"]
            messages.append(msg)

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
                # Save assistant message with tool_calls
                tool_calls_json = json.dumps([
                    {"id": tc.id, "type": tc.type,
                     "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                    for tc in message.tool_calls
                ])
                await queries.save_conversation_message(
                    self._db.conn, thread_ts, channel_id,
                    role="assistant", content=message.content,
                    tool_calls=tool_calls_json,
                )

                # Add assistant message to context
                assistant_msg: dict[str, Any] = {"role": "assistant"}
                if message.content:
                    assistant_msg["content"] = message.content
                assistant_msg["tool_calls"] = [
                    {"id": tc.id, "type": tc.type,
                     "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                    for tc in message.tool_calls
                ]
                messages.append(assistant_msg)

                # Execute each tool call
                for tc in message.tool_calls:
                    tool_name = tc.function.name
                    arguments = tc.function.arguments

                    # Check if tool requires confirmation
                    if self._tools.needs_confirmation(tool_name):
                        self._pending_confirmations[thread_ts] = {
                            "tool_call_id": tc.id,
                            "tool_name": tool_name,
                            "arguments": arguments,
                            "messages": messages,
                        }
                        # Save the user message before returning
                        user_msg = messages[-2] if len(messages) >= 2 else messages[-1]
                        if user_msg.get("role") == "user":
                            await queries.save_conversation_message(
                                self._db.conn, thread_ts, channel_id,
                                role="user", content=user_msg["content"],
                            )
                        confirmation_text = (
                            f"I need to run *{tool_name}* "
                            f"with arguments: `{arguments}`\n\n"
                            "Reply *yes* to confirm or *no* to cancel."
                        )
                        await queries.log_llm_usage(
                            self._db.conn, "conversation", self._settings.kimi_model,
                            total_in, total_out,
                        )
                        return confirmation_text

                    # Execute tool
                    log.info("tool.executing", tool=tool_name, round=_round)
                    result = await self._tools.execute(tool_name, arguments)

                    # Add tool result to messages
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result,
                    })

                    # Save tool result to DB
                    await queries.save_conversation_message(
                        self._db.conn, thread_ts, channel_id,
                        role="tool", content=result,
                        tool_call_id=tc.id, name=tool_name,
                    )

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
            # Execute the pending tool
            result = await self._tools.execute(
                pending["tool_name"], pending["arguments"]
            )

            # Rebuild messages and add tool result
            messages = pending["messages"]
            messages.append({
                "role": "tool",
                "tool_call_id": pending["tool_call_id"],
                "content": result,
            })

            await queries.save_conversation_message(
                self._db.conn, thread_ts, channel_id,
                role="tool", content=result,
                tool_call_id=pending["tool_call_id"],
                name=pending["tool_name"],
            )

            # Continue the conversation loop
            return await self._run_tool_loop(messages, thread_ts, channel_id)

        return "Cancelled. The operation was not executed."
