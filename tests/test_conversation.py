"""Tests for conversation manager with function calling loop."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from sgldhelper.ai.conversation import ConversationManager, SYSTEM_PROMPT
from sgldhelper.ai.client import KimiClient
from sgldhelper.ai.tools import ToolRegistry
from sgldhelper.db import queries


def _make_completion(content="Hello!", tool_calls=None, finish_reason="stop"):
    """Create a mock ChatCompletion response."""
    message = MagicMock()
    message.content = content
    message.tool_calls = tool_calls

    choice = MagicMock()
    choice.message = message
    choice.finish_reason = finish_reason

    usage = MagicMock()
    usage.prompt_tokens = 100
    usage.completion_tokens = 50

    response = MagicMock()
    response.choices = [choice]
    response.usage = usage
    return response


def _make_tool_call(id="call_1", name="get_open_prs", arguments="{}"):
    tc = MagicMock()
    tc.id = id
    tc.type = "function"
    tc.function = MagicMock()
    tc.function.name = name
    tc.function.arguments = arguments
    return tc


@pytest.fixture
def mock_kimi():
    client = AsyncMock(spec=KimiClient)
    client.extract_usage = MagicMock(return_value=(100, 50))
    return client


@pytest.fixture
def mock_tools():
    tools = AsyncMock(spec=ToolRegistry)
    tools.get_schemas = MagicMock(return_value=[
        {"type": "function", "function": {"name": "get_open_prs", "description": "...", "parameters": {}}},
    ])
    tools.needs_confirmation = MagicMock(return_value=False)
    tools.execute = AsyncMock(return_value='[{"pr_number": 1234, "title": "Test PR"}]')
    return tools


@pytest.fixture
async def manager(mock_kimi, mock_tools, db, settings):
    return ConversationManager(mock_kimi, mock_tools, db, settings)


class TestSimpleConversation:
    @pytest.mark.asyncio
    async def test_simple_reply(self, manager, mock_kimi):
        """Bot returns a simple text reply without tool calls."""
        mock_kimi.chat = AsyncMock(return_value=_make_completion("There are 3 open PRs."))

        reply = await manager.handle_mention(
            text="How many PRs are open?",
            thread_ts="thread_001",
            channel_id="C_TEST",
            user_id="U_TEST",
        )

        assert reply == "There are 3 open PRs."
        mock_kimi.chat.assert_called_once()

    @pytest.mark.asyncio
    async def test_conversation_saved_to_db(self, manager, mock_kimi, db):
        mock_kimi.chat = AsyncMock(return_value=_make_completion("Hello!"))

        await manager.handle_mention(
            text="Hi",
            thread_ts="thread_002",
            channel_id="C_TEST",
            user_id="U_TEST",
        )

        history = await queries.get_conversation_history(db.conn, "thread_002")
        assert len(history) == 2  # user + assistant
        assert history[0]["role"] == "user"
        assert history[0]["content"] == "Hi"
        assert history[1]["role"] == "assistant"
        assert history[1]["content"] == "Hello!"


class TestFunctionCalling:
    @pytest.mark.asyncio
    async def test_single_tool_call(self, manager, mock_kimi, mock_tools, db):
        """Bot makes one tool call then responds."""
        tool_call = _make_tool_call()

        # First call returns tool_calls, second returns final response
        mock_kimi.chat = AsyncMock(side_effect=[
            _make_completion(content=None, tool_calls=[tool_call], finish_reason="tool_calls"),
            _make_completion("Here are the open PRs: PR #1234 Test PR"),
        ])

        reply = await manager.handle_mention(
            text="Show me open PRs",
            thread_ts="thread_003",
            channel_id="C_TEST",
            user_id="U_TEST",
        )

        assert "open PRs" in reply
        assert mock_kimi.chat.call_count == 2
        mock_tools.execute.assert_called_once_with("get_open_prs", "{}")

    @pytest.mark.asyncio
    async def test_multi_round_tool_calls(self, manager, mock_kimi, mock_tools, db):
        """Bot makes two rounds of tool calls."""
        tc1 = _make_tool_call(id="call_1", name="get_open_prs", arguments="{}")
        tc2 = _make_tool_call(id="call_2", name="get_ci_status", arguments='{"pr_number": 1234}')

        mock_kimi.chat = AsyncMock(side_effect=[
            _make_completion(content=None, tool_calls=[tc1], finish_reason="tool_calls"),
            _make_completion(content=None, tool_calls=[tc2], finish_reason="tool_calls"),
            _make_completion("PR #1234 has 2 failing CI runs."),
        ])
        mock_tools.execute = AsyncMock(side_effect=[
            '[{"pr_number": 1234}]',
            '[{"run_id": 1, "conclusion": "failure"}]',
        ])

        reply = await manager.handle_mention(
            text="What's the CI status of open PRs?",
            thread_ts="thread_004",
            channel_id="C_TEST",
            user_id="U_TEST",
        )

        assert "failing CI" in reply
        assert mock_kimi.chat.call_count == 3
        assert mock_tools.execute.call_count == 2

    @pytest.mark.asyncio
    async def test_tool_results_saved_to_db(self, manager, mock_kimi, mock_tools, db):
        tool_call = _make_tool_call()
        mock_kimi.chat = AsyncMock(side_effect=[
            _make_completion(content=None, tool_calls=[tool_call], finish_reason="tool_calls"),
            _make_completion("Done!"),
        ])

        await manager.handle_mention(
            text="Show PRs",
            thread_ts="thread_005",
            channel_id="C_TEST",
            user_id="U_TEST",
        )

        history = await queries.get_conversation_history(db.conn, "thread_005")
        roles = [h["role"] for h in history]
        # Should have: assistant (with tool_calls), tool (result), user, assistant (final)
        assert "tool" in roles
        assert "assistant" in roles


class TestConfirmation:
    @pytest.mark.asyncio
    async def test_write_tool_asks_confirmation(self, manager, mock_kimi, mock_tools, db):
        """Write tools should trigger a confirmation prompt."""
        tc = _make_tool_call(id="call_rerun", name="rerun_ci", arguments='{"pr_number": 1234}')
        mock_tools.needs_confirmation = MagicMock(return_value=True)

        mock_kimi.chat = AsyncMock(return_value=_make_completion(
            content=None, tool_calls=[tc], finish_reason="tool_calls"
        ))

        reply = await manager.handle_mention(
            text="Rerun CI for PR 1234",
            thread_ts="thread_006",
            channel_id="C_TEST",
            user_id="U_TEST",
        )

        assert "confirm" in reply.lower() or "yes" in reply.lower()
        mock_tools.execute.assert_not_called()  # Should NOT execute yet

    @pytest.mark.asyncio
    async def test_confirm_yes_executes_tool(self, manager, mock_kimi, mock_tools, db):
        """Confirming 'yes' should execute the pending tool."""
        tc = _make_tool_call(id="call_rerun", name="rerun_ci", arguments='{"pr_number": 1234}')
        mock_tools.needs_confirmation = MagicMock(return_value=True)

        # First call triggers confirmation
        mock_kimi.chat = AsyncMock(return_value=_make_completion(
            content=None, tool_calls=[tc], finish_reason="tool_calls"
        ))
        await manager.handle_mention(
            text="Rerun CI for PR 1234",
            thread_ts="thread_007",
            channel_id="C_TEST",
            user_id="U_TEST",
        )

        # Now confirm with "yes"
        mock_tools.needs_confirmation = MagicMock(return_value=False)
        mock_kimi.chat = AsyncMock(return_value=_make_completion("CI rerun triggered!"))
        mock_tools.execute = AsyncMock(return_value='{"triggered": true}')

        reply = await manager.handle_mention(
            text="yes",
            thread_ts="thread_007",
            channel_id="C_TEST",
            user_id="U_TEST",
        )

        mock_tools.execute.assert_called_once_with("rerun_ci", '{"pr_number": 1234}')

    @pytest.mark.asyncio
    async def test_confirm_no_cancels(self, manager, mock_kimi, mock_tools, db):
        """Confirming 'no' should cancel the pending operation."""
        tc = _make_tool_call(id="call_rerun", name="rerun_ci", arguments='{"pr_number": 1234}')
        mock_tools.needs_confirmation = MagicMock(return_value=True)

        mock_kimi.chat = AsyncMock(return_value=_make_completion(
            content=None, tool_calls=[tc], finish_reason="tool_calls"
        ))
        await manager.handle_mention(
            text="Rerun CI for 1234",
            thread_ts="thread_008",
            channel_id="C_TEST",
            user_id="U_TEST",
        )

        reply = await manager.handle_mention(
            text="no",
            thread_ts="thread_008",
            channel_id="C_TEST",
            user_id="U_TEST",
        )

        assert "cancel" in reply.lower()
        mock_tools.execute.assert_not_called()


class TestConversationHistory:
    @pytest.mark.asyncio
    async def test_history_loaded_for_follow_up(self, manager, mock_kimi, db):
        """Follow-up messages in the same thread should include history."""
        mock_kimi.chat = AsyncMock(return_value=_make_completion("First reply"))

        await manager.handle_mention(
            text="Hi",
            thread_ts="thread_009",
            channel_id="C_TEST",
            user_id="U_TEST",
        )

        mock_kimi.chat = AsyncMock(return_value=_make_completion("Second reply"))
        await manager.handle_mention(
            text="What about CI?",
            thread_ts="thread_009",
            channel_id="C_TEST",
            user_id="U_TEST",
        )

        # The second call should have history in messages
        call_args = mock_kimi.chat.call_args
        messages = call_args.kwargs.get("messages") or call_args[1].get("messages")
        # Should include system + prior history + new message
        assert len(messages) >= 3
        assert messages[0]["role"] == "system"


class TestLLMUsageTracking:
    @pytest.mark.asyncio
    async def test_usage_logged(self, manager, mock_kimi, db):
        mock_kimi.chat = AsyncMock(return_value=_make_completion("OK"))

        await manager.handle_mention(
            text="Hello",
            thread_ts="thread_010",
            channel_id="C_TEST",
            user_id="U_TEST",
        )

        usage = await queries.get_llm_usage_summary(db.conn, 24)
        assert len(usage) >= 1
        total_calls = sum(u["calls"] for u in usage)
        assert total_calls >= 1


class TestMaxRounds:
    @pytest.mark.asyncio
    async def test_max_tool_rounds(self, manager, mock_kimi, mock_tools, db):
        """Should stop after MAX_TOOL_ROUNDS iterations."""
        tc = _make_tool_call()

        # Always return tool calls to exhaust the loop
        mock_kimi.chat = AsyncMock(return_value=_make_completion(
            content=None, tool_calls=[tc], finish_reason="tool_calls"
        ))

        reply = await manager.handle_mention(
            text="Do something",
            thread_ts="thread_011",
            channel_id="C_TEST",
            user_id="U_TEST",
        )

        assert "maximum" in reply.lower()
