---
name: slack-api-python
description: "Guide for building Slack apps and integrations with Python. Use this skill whenever the user wants to build a Slack bot, send messages to Slack, listen to Slack events, handle slash commands, create interactive modals, use Block Kit, implement OAuth for Slack, build AI agents/assistants for Slack, create custom workflow steps, or integrate with the Slack platform. Also trigger when code imports `slack_bolt`, `slack_sdk`, or `slack_sdk.web`, or when the user mentions 'Slack API', 'Slack bot', 'Slack app', 'Bolt for Python', or 'slack_sdk' in a development context."
---

# Slack API Python Development Guide

Build Slack apps and integrations using Python. Two main libraries: **Bolt for Python** (`slack_bolt`) as the high-level app framework, and **Python Slack SDK** (`slack_sdk`) as the low-level API client.

## Quick Reference

| Item | Value |
|------|-------|
| Framework Package | `slack_bolt` (Bolt for Python) |
| SDK Package | `slack_sdk` (Python Slack SDK) |
| Python Version | 3.7+ (latest stable recommended) |
| Bot Token Prefix | `xoxb-` |
| User Token Prefix | `xoxp-` |
| App-Level Token Prefix | `xapp-` |
| Socket Mode Scope | `connections:write` |
| Official Docs | https://docs.slack.dev/ |
| Bolt Docs | https://docs.slack.dev/tools/bolt-python/ |
| SDK Docs | https://docs.slack.dev/tools/python-slack-sdk/ |

## Reference Files

Read these for detailed examples when implementing specific features:

| File | Contents |
|------|----------|
| `references/bolt-framework.md` | Bolt app setup, messages, events, commands, actions, shortcuts, modals, views, middleware, context, App Home, select menus, acknowledgement patterns |
| `references/sdk-clients.md` | WebClient, WebhookClient, SocketModeClient, async clients, pagination, file uploads, message formatting, rate limiting, error handling |
| `references/ai-agents-and-workflows.md` | AI Assistant class, chat streaming, feedback buttons, custom workflow steps, LLM integration patterns |
| `references/auth-and-deployment.md` | OAuth setup, authorization, token management, Socket Mode, serverless (AWS Lambda), Flask/Django adapters, lazy listeners |

---

## When to Use What

| Goal | Library | Key Import |
|------|---------|------------|
| Build a full Slack app (events, commands, interactions) | `slack_bolt` | `from slack_bolt import App` |
| Just call a few Slack APIs (send message, list channels) | `slack_sdk` | `from slack_sdk import WebClient` |
| Send messages via webhook (no bot needed) | `slack_sdk` | `from slack_sdk.webhook import WebhookClient` |
| Real-time WebSocket connection | `slack_sdk` | `from slack_sdk.socket_mode import SocketModeClient` |
| Multi-workspace distribution with OAuth | `slack_bolt` | `from slack_bolt.oauth.oauth_settings import OAuthSettings` |
| Build AI assistant in Slack side panel | `slack_bolt` | `from slack_bolt import App, Assistant` |

---

## Installation

```bash
# Bolt framework (includes slack_sdk as dependency)
pip install slack_bolt

# SDK only (if you don't need the framework)
pip install slack_sdk
```

---

## Setup & Authentication

### Token Types

| Token | Prefix | Purpose |
|-------|--------|---------|
| Bot Token | `xoxb-` | Call API methods as the bot user; granted once per workspace install |
| User Token | `xoxp-` | Call API methods on behalf of users after OAuth |
| App-Level Token | `xapp-` | Used for Socket Mode; represents app across organizations |

Never hardcode tokens. Use environment variables:

```python
import os
SLACK_BOT_TOKEN = os.environ["SLACK_BOT_TOKEN"]
SLACK_APP_TOKEN = os.environ["SLACK_APP_TOKEN"]  # for Socket Mode
```

---

## Bolt App — Quick Start (Socket Mode)

```python
import os
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

app = App(token=os.environ.get("SLACK_BOT_TOKEN"))

@app.message("hello")
def message_hello(message, say):
    say(f"Hey there <@{message['user']}>!")

@app.command("/echo")
def handle_echo(ack, respond, command):
    ack()
    respond(f"{command['text']}")

@app.event("app_mention")
def handle_mention(event, say):
    say(f"Hi <@{event['user']}>! You mentioned me.")

if __name__ == "__main__":
    SocketModeHandler(app, os.environ["SLACK_APP_TOKEN"]).start()
```

### Bolt App — HTTP Mode

```python
import os
from slack_bolt import App

app = App(
    token=os.environ.get("SLACK_BOT_TOKEN"),
    signing_secret=os.environ.get("SLACK_SIGNING_SECRET"),
)

# ... add handlers ...

if __name__ == "__main__":
    app.start(port=int(os.environ.get("PORT", 3000)))
```

---

## SDK — Direct API Calls

```python
import os
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

client = WebClient(token=os.environ["SLACK_BOT_TOKEN"])

# Send a message
try:
    result = client.chat_postMessage(channel="#general", text="Hello from Python!")
    print(f"Message sent: {result['ts']}")
except SlackApiError as e:
    print(f"Error: {e.response['error']}")

# List channels
result = client.conversations_list()
for channel in result["channels"]:
    print(channel["name"])

# Upload a file
result = client.files_upload_v2(
    channels=["C12345"],
    file="./report.csv",
    title="Daily Report",
    initial_comment="Here's today's report",
)
```

### Webhook — Send Messages Without a Bot

```python
from slack_sdk.webhook import WebhookClient

webhook = WebhookClient(url=os.environ["SLACK_WEBHOOK_URL"])
response = webhook.send(text="Hello from a webhook!")
```

---

## Core Bolt Concepts

### Listener Decorator Reference

| Decorator | Purpose | Must `ack()`? |
|-----------|---------|---------------|
| `@app.message(pattern)` | Listen to messages matching str/regex | No |
| `@app.event(event_type)` | Listen to Events API events | No |
| `@app.command("/name")` | Listen to slash commands | **Yes** |
| `@app.action(action_id)` | Listen to button clicks, menu selects | **Yes** |
| `@app.shortcut(callback_id)` | Listen to global/message shortcuts | **Yes** |
| `@app.view(callback_id)` | Listen to modal submissions | **Yes** |
| `@app.view_closed(callback_id)` | Listen to modal close | **Yes** |
| `@app.options(action_id)` | Handle dynamic select menus | **Yes** (with options) |
| `@app.function(callback_id)` | Handle custom workflow steps | No (use complete/fail) |

### Common Listener Arguments

| Argument | Description |
|----------|-------------|
| `ack` | Acknowledge request (3-second deadline) |
| `say` | Send message to originating conversation |
| `respond` | Reply via `response_url` |
| `client` | Slack Web API client (`WebClient`) |
| `body` | Full request body |
| `event` / `message` / `command` / `action` / `shortcut` / `view` | Type-specific payload |
| `context` | Mutable execution context dict |
| `logger` | Application logger |
| `complete` / `fail` | For custom workflow steps |

### Acknowledgement Rule

Actions, commands, shortcuts, options, and view submissions **must always** be acknowledged with `ack()` within **3 seconds**. Call `ack()` immediately before any time-consuming work.

---

## Block Kit — Building Rich UIs

Block Kit provides reusable, stackable, interactive components for messages, modals, and App Home.

### Message with Button

```python
@app.message("hello")
def message_hello(message, say):
    say(
        blocks=[
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text": f"Hey <@{message['user']}>!"},
                "accessory": {
                    "type": "button",
                    "text": {"type": "plain_text", "text": "Click Me"},
                    "action_id": "button_click",
                },
            }
        ],
        text=f"Hey <@{message['user']}>!",  # fallback for notifications
    )

@app.action("button_click")
def handle_click(body, ack, say):
    ack()
    say(f"<@{body['user']['id']}> clicked the button")
```

### Common Block Types

| Block | Purpose |
|-------|---------|
| `section` | Text with optional accessory (button, menu, image) |
| `actions` | Container for interactive elements |
| `input` | Collects user input (modals only) |
| `header` | Large text heading |
| `divider` | Visual separator |
| `image` | Displays an image |
| `context` | Small contextual text/images |
| `rich_text` | Rich text formatted content |

### Common Interactive Elements

`button`, `static_select`, `external_select`, `multi_static_select`, `datepicker`, `timepicker`, `checkboxes`, `radio_buttons`, `plain_text_input`, `number_input`, `email_text_input`, `url_text_input`, `overflow`, `file_input`

---

## Opening Modals

```python
@app.command("/feedback")
def open_feedback_modal(ack, body, client):
    ack()
    client.views_open(
        trigger_id=body["trigger_id"],
        view={
            "type": "modal",
            "callback_id": "feedback_modal",
            "title": {"type": "plain_text", "text": "Feedback"},
            "submit": {"type": "plain_text", "text": "Submit"},
            "blocks": [
                {
                    "type": "input",
                    "block_id": "feedback_input",
                    "label": {"type": "plain_text", "text": "Your feedback"},
                    "element": {
                        "type": "plain_text_input",
                        "action_id": "feedback_text",
                        "multiline": True,
                    },
                }
            ],
        },
    )

@app.view("feedback_modal")
def handle_feedback(ack, body, client, view):
    feedback = view["state"]["values"]["feedback_input"]["feedback_text"]["value"]
    user = body["user"]["id"]
    # Validate
    errors = {}
    if len(feedback) < 10:
        errors["feedback_input"] = "Please write at least 10 characters"
    if errors:
        ack(response_action="errors", errors=errors)
        return
    ack()
    client.chat_postMessage(channel=user, text=f"Thanks for your feedback: {feedback}")
```

**Key modal operations:**
- `client.views_open(trigger_id, view)` — Open a new modal (trigger_id valid for 3 seconds)
- `client.views_update(view_id, hash, view)` — Update an existing modal
- `client.views_push(trigger_id, view)` — Push a new view onto the modal stack (max 2 pushes)

Access submitted values: `view["state"]["values"][block_id][action_id]`

---

## Event Subscriptions

Subscribe to events in your app configuration. Common events:

| Event | Description | Required Scope |
|-------|-------------|---------------|
| `message.channels` | Messages in public channels | `channels:history` |
| `message.groups` | Messages in private channels | `groups:history` |
| `message.im` | Direct messages | `im:history` |
| `app_mention` | When your app is @mentioned | `app_mentions:read` |
| `app_home_opened` | User opens App Home tab | — |
| `team_join` | New user joins workspace | `users:read` |
| `reaction_added` | Reaction added to a message | `reactions:read` |

### Event Subtypes

```python
@app.event({"type": "message", "subtype": "message_changed"})
def log_edit(logger, event):
    logger.info(f"Message edited: {event['text']}")
```

---

## Common Scopes

| Scope | Grants |
|-------|--------|
| `chat:write` | Send messages as the app |
| `channels:history` | Read public channel messages |
| `channels:read` | Read basic channel info |
| `groups:history` | Read private channel messages |
| `im:history` | Read DM messages |
| `commands` | Register slash commands |
| `users:read` | Read user info |
| `reactions:read` / `reactions:write` | Read/add reactions |
| `files:read` / `files:write` | Access/upload files |
| `assistant:write` | AI Assistant features |
| `connections:write` | Socket Mode |

---

## Best Practices

1. **Always `ack()` first** — Call `ack()` immediately in commands/actions/shortcuts before doing work
2. **Use Socket Mode for development** — No public URL needed, simpler setup
3. **Use environment variables for tokens** — Never hardcode `xoxb-`, `xoxp-`, or `xapp-` tokens
4. **Provide `text` fallback** — When using `blocks`, always include a `text` parameter for notification previews
5. **Handle errors** — Wrap API calls in try/except for `SlackApiError`
6. **Use Block Kit Builder** — Design interactive layouts at https://app.slack.com/block-kit-builder
7. **Request minimal scopes** — Only request the permissions your app actually needs
8. **Use lazy listeners for FaaS** — Set `process_before_response=True` and use `lazy=[]` for serverless
