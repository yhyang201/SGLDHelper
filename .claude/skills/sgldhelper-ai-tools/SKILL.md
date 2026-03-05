---
name: sgldhelper-ai-tools
description: "Guide for adding or modifying AI tools (function calling) in SGLDHelper. Use when adding new tools to ai/tools.py, modifying the conversation loop in ai/conversation.py, changing tool confirmation behavior, or updating the system prompt. Also trigger when debugging AI responses, tool execution issues, or conversation context problems."
---

# SGLDHelper AI Tools & Conversation System

## Tool Registry (`ai/tools.py`)

### Adding a New Tool — Checklist

1. **Handler method**: Add `async def _my_tool(self, param1: type, ...) -> dict[str, Any]` to `ToolRegistry`
2. **Registration**: Add `self._register(...)` call in `_register_all()`:
   ```python
   self._register(
       name="my_tool",
       description="What this tool does (shown to the LLM)",
       parameters={
           "type": "object",
           "properties": {
               "param1": {"type": "string", "description": "..."},
           },
           "required": ["param1"],
       },
       handler=self._my_tool,
       requires_confirmation=False,  # True for destructive ops
   )
   ```
3. **Late dependencies**: If the tool needs `ci_monitor`, `auto_merge`, or `health_checker`, access via `self._ci_monitor` etc. (injected via `set_ci_components()`). Guard with `if not self._ci_monitor: return {"error": "..."}`
4. **Tests**: Update `tests/test_ai_tools.py`:
   - Increment count in `test_schemas_are_valid`
   - Add tool name to `expected` set in `test_tool_names`
   - Add execution test if non-trivial

### Tool Return Values

Tools return `dict[str, Any]` which gets JSON-serialized. Conventions:
- Success: return the data dict directly
- Error: return `{"error": "Human-readable message"}`
- The LLM sees the JSON and formulates a natural language response

### Confirmation Flow

Tools with `requires_confirmation=True`:
1. LLM calls the tool → bot pauses, asks user "Reply yes to confirm"
2. User says "yes"/"确认" → tool executes → loop continues
3. User says anything else → cancelled, tool gets `{"cancelled": True}`
4. Pending state stored in `ConversationManager._pending_confirmations[thread_ts]`

Currently confirmed tools: `cancel_auto_merge`, `merge_pr`
Currently unconfirmed but destructive: `trigger_ci` (intentionally fast for CI ops)

## Conversation Manager (`ai/conversation.py`)

### Message Flow

```
handle_mention(text, thread_ts, channel_id, user_id)
  → _build_messages(thread_ts, text, user_id)
    → System prompt + user memory
    → Load conversation history from DB (thread_ts)
    → Sanitize orphaned tool_calls
    → Append new user message
  → _run_tool_loop(messages, thread_ts, channel_id)
    → LLM call with tools
    → If tool_calls: execute auto tools, pause for confirmation tools
    → If stop: save and return reply
    → Max 10 rounds
```

### Thread Context

Bot messages need to be in conversation history for thread replies to have context:
- Notifications use `post_message_with_context(db_conn=...)` to save as `role="assistant"`
- When user replies in a PR notification thread, the AI sees the notification text
- Without this, the AI has no idea which PR the thread is about

### System Prompt

Defined as `SYSTEM_PROMPT` in `conversation.py`. Key sections:
- Slack mrkdwn formatting rules (NOT standard Markdown)
- Tool usage guidelines
- Per-user memory injected dynamically

### Kimi K2.5 Specifics

- Uses `thinking=True` for reasoning mode
- `reasoning_content` must be replayed in subsequent requests (Kimi requirement)
- API is OpenAI-compatible via `openai` SDK with custom `base_url`
- Client in `ai/client.py` wraps the OpenAI client

## Classifier (`ai/classifier.py`)

Runs in parallel with conversation on every message:
- Classifies as `progress_update`, `blocker`, or `none`
- Uses Kimi instant mode (no thinking) for speed
- If detected: posts confirmation button (Confirm/Dismiss)
- Independent of conversation — both run concurrently
