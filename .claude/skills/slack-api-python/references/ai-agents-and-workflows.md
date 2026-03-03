# AI Agents & Custom Workflows — Detailed Guide

## Table of Contents
1. [AI Assistant Class](#ai-assistant-class)
2. [Chat Streaming](#chat-streaming)
3. [Feedback Buttons](#feedback-buttons)
4. [Custom Workflow Steps](#custom-workflow-steps)
5. [Complete AI Agent Example](#complete-ai-agent-example)

---

## AI Assistant Class

The `Assistant` class provides a side-panel experience designed for AI agents and assistants.

### Required Scopes

- `assistant:write`
- `chat:write`
- `im:history`

### Required Event Subscriptions

- `assistant_thread_started`
- `assistant_thread_context_changed`
- `message.im`

### Basic Setup

```python
import os
import logging
from slack_bolt import App, Assistant, Say, BoltContext
from slack_bolt.adapter.socket_mode import SocketModeHandler
from slack_sdk import WebClient

app = App(token=os.environ["SLACK_BOT_TOKEN"])
assistant = Assistant()

@assistant.thread_started
def start_thread(
    say: Say,
    set_suggested_prompts,
    get_thread_context,
    logger: logging.Logger,
):
    try:
        say("How can I help you?")

        prompts = [
            {
                "title": "Summarize a channel",
                "message": "Can you generate a brief summary of the referred channel?",
            },
            {
                "title": "Suggest app names",
                "message": "Can you suggest a few names for my Slack app?",
            },
        ]

        # Add context-aware prompts
        thread_context = get_thread_context()
        if thread_context and thread_context.channel_id:
            prompts.append({
                "title": "Summarize this channel",
                "message": "Summarize the recent activity in this channel.",
            })

        set_suggested_prompts(prompts=prompts)
    except Exception as e:
        logger.exception(f"Failed: {e}")
        say(f":warning: Something went wrong! ({e})")

@assistant.user_message
def handle_user_message(
    client: WebClient,
    context: BoltContext,
    logger: logging.Logger,
    payload: dict,
    say: Say,
    set_status,
):
    try:
        set_status(
            status="thinking...",
            loading_messages=[
                "Analyzing your request...",
                "Gathering information...",
            ],
        )

        # Get conversation history
        replies = client.conversations_replies(
            channel=context.channel_id,
            ts=context.thread_ts,
            oldest=context.thread_ts,
            limit=10,
        )
        messages = []
        for msg in replies["messages"]:
            role = "user" if msg.get("bot_id") is None else "assistant"
            messages.append({"role": role, "content": msg["text"]})

        # Call your LLM
        response = call_llm(messages)
        say(text=response)

    except Exception as e:
        logger.exception(f"Failed: {e}")
        say(f":warning: Sorry, something went wrong ({e})")

app.use(assistant)

if __name__ == "__main__":
    SocketModeHandler(app, os.environ["SLACK_APP_TOKEN"]).start()
```

### Assistant Event Handlers

| Handler | Event | Description |
|---------|-------|-------------|
| `@assistant.thread_started` | `assistant_thread_started` | User opens assistant thread |
| `@assistant.user_message` | `message.im` | User sends a message |
| `@assistant.thread_context_changed` | `assistant_thread_context_changed` | Thread context changes (e.g., user navigates to different channel) |
| `@assistant.bot_message` | (bot's own messages) | React to bot's own messages (requires `ignoring_self_assistant_message_events_enabled=False`) |

### Handler Arguments

| Argument | Description |
|----------|-------------|
| `say` | Send message in thread |
| `set_status` | Set thinking status with loading messages |
| `set_suggested_prompts` | Show suggested prompt buttons |
| `get_thread_context` | Get thread context (channel, etc.) |
| `client` | WebClient |
| `context` | BoltContext with `channel_id`, `thread_ts` |
| `payload` | Message payload with `text`, `user`, `thread_ts` |
| `logger` | Logger |

### Custom Thread Context Storage

Default uses message metadata. For custom:

```python
from slack_bolt import FileAssistantThreadContextStore

assistant = Assistant(
    thread_context_store=FileAssistantThreadContextStore()
)
```

### Block Kit in Assistant Threads

Enable handling bot's own messages:

```python
app = App(
    token=os.environ["SLACK_BOT_TOKEN"],
    ignoring_self_assistant_message_events_enabled=False,
)

@assistant.thread_started
def start_thread(say: Say):
    say(
        text="Hi! What can I help with?",
        blocks=[
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text": ":wave: Hi! What can I help with?"},
            },
            {
                "type": "actions",
                "elements": [
                    {
                        "type": "button",
                        "action_id": "generate_report",
                        "text": {"type": "plain_text", "text": "Generate Report"},
                        "value": "report",
                    },
                ],
            },
        ],
    )

@app.action("generate_report")
def handle_report(ack, body, client):
    ack()
    client.views_open(
        trigger_id=body["trigger_id"],
        view={...},
    )
```

---

## Chat Streaming

Stream text to a Slack message in real-time (like typing effect). Uses three APIs: `chat.startStream`, `chat.appendStream`, `chat.stopStream`.

### Using the SDK Utility

```python
@assistant.user_message
def handle_message(client, context, payload, say, set_status, logger):
    try:
        channel_id = payload["channel"]
        team_id = payload["team"]
        user_id = payload["user"]
        thread_ts = payload["thread_ts"]

        set_status(status="thinking...")

        # Get thread history for LLM
        replies = client.conversations_replies(
            channel=context.channel_id,
            ts=context.thread_ts,
            limit=10,
        )
        messages = [
            {"role": "user" if m.get("bot_id") is None else "assistant", "content": m["text"]}
            for m in replies["messages"]
        ]

        # Call LLM with streaming
        llm_stream = call_llm_streaming(messages)

        # Start Slack stream
        streamer = client.chat_stream(
            channel=channel_id,
            recipient_team_id=team_id,
            recipient_user_id=user_id,
            thread_ts=thread_ts,
        )

        # Append chunks as they arrive
        for event in llm_stream:
            if event.type == "response.output_text.delta":
                streamer.append(markdown_text=f"{event.delta}")

        # Stop streaming (finalizes the message)
        streamer.stop()

    except Exception as e:
        logger.exception(f"Streaming failed: {e}")
        say(f":warning: Something went wrong ({e})")
```

### Stop with Feedback Buttons

```python
feedback_block = create_feedback_block()
streamer.stop(blocks=feedback_block)
```

### Manual Stream Control (Without Utility)

```python
# Start stream
start_response = client.chat_startStream(
    channel=channel_id,
    thread_ts=thread_ts,
)
stream_id = start_response["stream_id"]

# Append text
client.chat_appendStream(
    stream_id=stream_id,
    markdown_text="Hello, ",
)
client.chat_appendStream(
    stream_id=stream_id,
    markdown_text="world!",
)

# Stop stream
client.chat_stopStream(stream_id=stream_id)
```

---

## Feedback Buttons

Add thumbs-up/down feedback buttons to AI responses.

```python
from slack_sdk.models.blocks import (
    Block,
    ContextActionsBlock,
    FeedbackButtonsElement,
    FeedbackButtonObject,
)

def create_feedback_block() -> list:
    return [
        ContextActionsBlock(
            elements=[
                FeedbackButtonsElement(
                    action_id="feedback",
                    positive_button=FeedbackButtonObject(
                        text="Good Response",
                        accessibility_label="Submit positive feedback",
                        value="good-feedback",
                    ),
                    negative_button=FeedbackButtonObject(
                        text="Bad Response",
                        accessibility_label="Submit negative feedback",
                        value="bad-feedback",
                    ),
                )
            ]
        )
    ]

# Handle feedback
@app.action("feedback")
def handle_feedback(ack, body, client, logger):
    ack()
    message_ts = body["message"]["ts"]
    channel_id = body["channel"]["id"]
    feedback_type = body["actions"][0]["value"]
    is_positive = feedback_type == "good-feedback"

    if is_positive:
        client.chat_postEphemeral(
            channel=channel_id,
            user=body["user"]["id"],
            thread_ts=message_ts,
            text="Thanks! Glad it was helpful.",
        )
    else:
        client.chat_postEphemeral(
            channel=channel_id,
            user=body["user"]["id"],
            thread_ts=message_ts,
            text="Sorry about that. Starting a new thread may help.",
        )
```

---

## Custom Workflow Steps

Create custom steps for Slack's Workflow Builder.

### Basic Step

```python
@app.function("sample_custom_step")
def sample_step(inputs: dict, fail, complete):
    try:
        message = inputs["message"]
        complete(outputs={"message": f":wave: You submitted: \n\n>{message}"})
    except Exception as e:
        fail(f"Failed: {e}")
        raise e
```

### App Manifest for Custom Steps

```json
{
    "functions": {
        "sample_custom_step": {
            "title": "Sample custom step",
            "description": "Run a sample custom step",
            "input_parameters": {
                "message": {
                    "type": "string",
                    "title": "Message",
                    "description": "A message to be formatted",
                    "is_required": true
                }
            },
            "output_parameters": {
                "message": {
                    "type": "string",
                    "title": "Message",
                    "description": "A formatted message",
                    "is_required": true
                }
            }
        }
    }
}
```

### Interactive Custom Step (with Button)

Don't call `complete()`/`fail()` in the function handler — defer to the interaction handler.

```python
@app.function("custom_step_button")
def step_with_button(inputs, say, fail):
    try:
        say(
            channel=inputs["user_id"],
            text="Click to complete",
            blocks=[
                {
                    "type": "section",
                    "text": {"type": "mrkdwn", "text": "Click the button to complete"},
                    "accessory": {
                        "type": "button",
                        "text": {"type": "plain_text", "text": "Complete"},
                        "action_id": "step_complete_click",
                    },
                }
            ],
        )
    except Exception as e:
        fail(f"Failed: {e}")

@app.action("step_complete_click")
def handle_step_click(ack, body, context, client, complete, fail):
    ack()
    try:
        client.chat_update(
            channel=context.channel_id,
            ts=body["message"]["ts"],
            text="Step completed!",
        )
        complete({"user_id": context.actor_user_id})
    except Exception as e:
        fail(f"Failed: {e}")
```

### Dynamic Options for Steps

```python
@app.function("get-projects", auto_acknowledge=False)
def handle_get_projects(ack, complete):
    try:
        complete(
            outputs={
                "options": [
                    {"text": {"type": "plain_text", "text": "Project A"}, "value": "p1"},
                    {"text": {"type": "plain_text", "text": "Project B"}, "value": "p2"},
                ]
            }
        )
    finally:
        ack()
```

Manifest with dynamic options:
```json
{
    "functions": {
        "create-issue": {
            "title": "Create Issue",
            "input_parameters": {
                "project": {
                    "type": "string",
                    "title": "Project",
                    "is_required": true,
                    "dynamic_options": {
                        "function": "#/functions/get-projects",
                        "inputs": {}
                    }
                }
            },
            "output_parameters": {}
        }
    }
}
```

---

## Complete AI Agent Example

Full example combining Assistant, streaming, feedback, and LLM integration:

```python
import os
import logging
from typing import List, Dict

from slack_bolt import App, Assistant, Say, BoltContext
from slack_bolt.adapter.socket_mode import SocketModeHandler
from slack_sdk import WebClient
from slack_sdk.models.blocks import (
    Block, ContextActionsBlock, FeedbackButtonsElement, FeedbackButtonObject,
)

# Your LLM client (e.g., OpenAI)
import openai

app = App(token=os.environ["SLACK_BOT_TOKEN"])
assistant = Assistant()

SYSTEM_PROMPT = """
You're an AI assistant in a Slack workspace.
Respond professionally. Convert markdown to Slack-compatible format.
Keep Slack syntax like <@USER_ID> or <#CHANNEL_ID> as-is.
"""

def call_llm(messages: List[Dict[str, str]]):
    """Call LLM with streaming."""
    client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    return client.responses.create(
        model="gpt-4o-mini",
        input=[{"role": "system", "content": SYSTEM_PROMPT}] + messages,
        stream=True,
    )

def create_feedback_block() -> List[Block]:
    return [
        ContextActionsBlock(elements=[
            FeedbackButtonsElement(
                action_id="ai_feedback",
                positive_button=FeedbackButtonObject(
                    text="Good", value="good",
                    accessibility_label="Good response",
                ),
                negative_button=FeedbackButtonObject(
                    text="Bad", value="bad",
                    accessibility_label="Bad response",
                ),
            )
        ])
    ]

@assistant.thread_started
def on_thread_started(say: Say, set_suggested_prompts, logger):
    try:
        say("Hi! How can I help?")
        set_suggested_prompts(prompts=[
            {"title": "Summarize channel", "message": "Summarize the referred channel"},
            {"title": "Draft a message", "message": "Help me draft a professional message"},
        ])
    except Exception as e:
        logger.exception(e)
        say(f":warning: Error: {e}")

@assistant.user_message
def on_user_message(
    client: WebClient,
    context: BoltContext,
    payload: dict,
    say: Say,
    set_status,
    logger,
):
    try:
        set_status(status="thinking...")

        # Build conversation history
        replies = client.conversations_replies(
            channel=context.channel_id,
            ts=context.thread_ts,
            limit=20,
        )
        messages = [
            {
                "role": "user" if m.get("bot_id") is None else "assistant",
                "content": m["text"],
            }
            for m in replies["messages"]
        ]

        # Stream response
        llm_stream = call_llm(messages)
        streamer = client.chat_stream(
            channel=payload["channel"],
            recipient_team_id=payload["team"],
            recipient_user_id=payload["user"],
            thread_ts=payload["thread_ts"],
        )

        for event in llm_stream:
            if event.type == "response.output_text.delta":
                streamer.append(markdown_text=event.delta)

        streamer.stop(blocks=create_feedback_block())

    except Exception as e:
        logger.exception(e)
        say(f":warning: Error: {e}")

@app.action("ai_feedback")
def handle_feedback(ack, body, client):
    ack()
    value = body["actions"][0]["value"]
    client.chat_postEphemeral(
        channel=body["channel"]["id"],
        user=body["user"]["id"],
        thread_ts=body["message"]["ts"],
        text="Thanks for the feedback!" if value == "good" else "Sorry, I'll try to do better.",
    )

app.use(assistant)

if __name__ == "__main__":
    SocketModeHandler(app, os.environ["SLACK_APP_TOKEN"]).start()
```
