# Bolt for Python Framework — Detailed Guide

## Table of Contents
1. [App Setup](#app-setup)
2. [Listening to Messages](#listening-to-messages)
3. [Listening to Events](#listening-to-events)
4. [Slash Commands](#slash-commands)
5. [Actions & Interactivity](#actions--interactivity)
6. [Shortcuts](#shortcuts)
7. [Opening Modals](#opening-modals)
8. [Updating & Pushing Views](#updating--pushing-views)
9. [View Submissions](#view-submissions)
10. [App Home](#app-home)
11. [Select Menu Options](#select-menu-options)
12. [Middleware](#middleware)
13. [Context](#context)
14. [Sending Messages](#sending-messages)

---

## App Setup

### Socket Mode (Recommended for Development)

```python
import os
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

app = App(token=os.environ.get("SLACK_BOT_TOKEN"))

if __name__ == "__main__":
    SocketModeHandler(app, os.environ["SLACK_APP_TOKEN"]).start()
```

Requires an App-Level Token with `connections:write` scope.

### HTTP Mode

```python
import os
from slack_bolt import App

app = App(
    token=os.environ.get("SLACK_BOT_TOKEN"),
    signing_secret=os.environ.get("SLACK_SIGNING_SECRET"),
)

if __name__ == "__main__":
    app.start(port=int(os.environ.get("PORT", 3000)))
```

### Async App

```python
from slack_bolt.app.async_app import AsyncApp
from slack_bolt.adapter.socket_mode.async_handler import AsyncSocketModeHandler

app = AsyncApp(token=os.environ["SLACK_BOT_TOKEN"])

@app.message("hello")
async def message_hello(message, say):
    await say(f"Hey <@{message['user']}>!")

async def main():
    handler = AsyncSocketModeHandler(app, os.environ["SLACK_APP_TOKEN"])
    await handler.start_async()
```

---

## Listening to Messages

The `message()` method filters events of type `message`. Accepts `str` or `re.Pattern`.

### String Matching

```python
@app.message(":wave:")
def say_hello(message, say):
    user = message['user']
    say(f"Hi there, <@{user}>!")
```

### Regular Expression Matching

```python
import re

@app.message(re.compile("(hi|hello|hey)"))
def say_hello_regex(say, context):
    greeting = context['matches'][0]
    say(f"{greeting}, how are you?")
```

Regex matches are accessible via `context['matches']`.

---

## Listening to Events

The `event()` method listens to Events API events. Requires `eventType` as `str`.

### Basic Event

```python
@app.event("team_join")
def ask_for_introduction(event, say):
    welcome_channel_id = "C12345"
    user_id = event["user"]
    say(text=f"Welcome to the team, <@{user_id}>!", channel=welcome_channel_id)
```

### Message Subtypes

```python
@app.event({"type": "message", "subtype": "message_changed"})
def log_message_change(logger, event):
    logger.info(f"User {event['user']} changed message to {event['text']}")
```

Setting `subtype` to `None` filters for events without a subtype.

### Required Event Subscriptions for Messages

- `message.channels` — messages in public channels
- `message.groups` — messages in private channels
- `message.im` — direct messages
- `message.mpim` — multi-person direct messages

---

## Slash Commands

The `command()` method listens to slash commands. **Must call `ack()`.**

```python
@app.command("/echo")
def repeat_text(ack, respond, command):
    ack()
    respond(f"{command['text']}")
```

### Command Payload Fields

| Field | Description |
|-------|-------------|
| `command['text']` | Text after the command |
| `command['user_id']` | User who invoked |
| `command['channel_id']` | Channel where invoked |
| `command['trigger_id']` | For opening modals (valid 3s) |

When configuring in app settings, append `/slack/events` to the request URL.

---

## Actions & Interactivity

The `action()` method handles button clicks, menu selects, etc. Accepts `action_id` as `str`, `re.Pattern`, or constraint dict. **Must call `ack()`.**

### Basic Action

```python
@app.action("approve_button")
def handle_approve(ack, say):
    ack()
    say("Request approved")
```

### Constraint Object (match block_id + action_id)

```python
@app.action({"block_id": "assign_ticket", "action_id": "select_user"})
def update_message(ack, body, client):
    ack()
    client.reactions_add(
        name="white_check_mark",
        channel=body["channel"]["id"],
        timestamp=body["container"]["message_ts"],
    )
```

### Using `respond()`

```python
@app.action("user_select")
def select_user(ack, action, respond):
    ack()
    respond(f"You selected <@{action['selected_user']}>")
```

`respond()` accepts: `response_type` ("in_channel"/"ephemeral"), `replace_original`, `delete_original`, `unfurl_links`, `unfurl_media`.

### Interactive Message Example

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
        text=f"Hey <@{message['user']}>!",
    )

@app.action("button_click")
def action_button_click(body, ack, say):
    ack()
    say(f"<@{body['user']['id']}> clicked the button")
```

---

## Shortcuts

The `shortcut()` method handles global and message shortcuts. **Must call `ack()`.**

- **Global shortcuts**: Available from search and composer — NO channel ID
- **Message shortcuts**: Available from message context menus — HAS channel ID

```python
@app.shortcut("open_modal")
def open_modal(ack, shortcut, client):
    ack()
    client.views_open(
        trigger_id=shortcut["trigger_id"],
        view={
            "type": "modal",
            "title": {"type": "plain_text", "text": "My App"},
            "close": {"type": "plain_text", "text": "Close"},
            "blocks": [
                {
                    "type": "section",
                    "text": {"type": "mrkdwn", "text": "A simple modal."},
                }
            ],
        },
    )
```

### Filter by Type

```python
@app.shortcut({"callback_id": "open_modal", "type": "message_action"})
def open_modal(ack, shortcut, client):
    ack()
    client.views_open(trigger_id=shortcut["trigger_id"], view={...})
```

---

## Opening Modals

Open modals with `client.views_open()` using a valid `trigger_id` (expires in 3 seconds).

```python
@app.shortcut("open_modal")
def open_modal(ack, body, client):
    ack()
    client.views_open(
        trigger_id=body["trigger_id"],
        view={
            "type": "modal",
            "callback_id": "view_1",
            "title": {"type": "plain_text", "text": "My App"},
            "submit": {"type": "plain_text", "text": "Submit"},
            "blocks": [
                {
                    "type": "section",
                    "text": {"type": "mrkdwn", "text": "Welcome to a modal with _blocks_"},
                    "accessory": {
                        "type": "button",
                        "text": {"type": "plain_text", "text": "Click me!"},
                        "action_id": "button_abc",
                    },
                },
                {
                    "type": "input",
                    "block_id": "input_c",
                    "label": {"type": "plain_text", "text": "Your hopes and dreams?"},
                    "element": {
                        "type": "plain_text_input",
                        "action_id": "dreamy_input",
                        "multiline": True,
                    },
                },
            ],
        },
    )
```

---

## Updating & Pushing Views

### Update an Existing View

```python
@app.action("button_abc")
def update_modal(ack, body, client):
    ack()
    client.views_update(
        view_id=body["view"]["id"],
        hash=body["view"]["hash"],  # race condition protection
        view={
            "type": "modal",
            "callback_id": "view_1",
            "title": {"type": "plain_text", "text": "Updated modal"},
            "blocks": [
                {
                    "type": "section",
                    "text": {"type": "plain_text", "text": "You updated the modal!"},
                },
            ],
        },
    )
```

### Push a New View onto the Stack

After opening a modal, you may push up to **2 additional views** onto the stack.

```python
client.views_push(
    trigger_id=body["trigger_id"],
    view={
        "type": "modal",
        "title": {"type": "plain_text", "text": "Step 2"},
        "blocks": [...],
    },
)
```

---

## View Submissions

The `view()` method handles `view_submission` events. Access input values via `view["state"]["values"][block_id][action_id]`.

```python
@app.view("view_1")
def handle_submission(ack, body, client, view, logger):
    value = view["state"]["values"]["input_c"]["dreamy_input"]["value"]
    user = body["user"]["id"]

    errors = {}
    if value and len(value) <= 5:
        errors["input_c"] = "Must be longer than 5 characters"
    if errors:
        ack(response_action="errors", errors=errors)
        return

    ack()
    client.chat_postMessage(channel=user, text=f"Your submission: {value}")
```

### Update View on Submission

```python
ack(response_action="update", view=build_new_view(body))
```

### Handle View Close

Set `notify_on_close: True` during view creation, then:

```python
@app.view_closed("modal-id")
def handle_close(ack, body, logger):
    ack()
    logger.info(f"User {body['user']['id']} closed the modal")
```

---

## App Home

Publish a Home tab view when users open your app.

```python
@app.event("app_home_opened")
def update_home_tab(client, event, logger):
    try:
        client.views_publish(
            user_id=event["user"],
            view={
                "type": "home",
                "blocks": [
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": f"*Welcome, <@{event['user']}>!* :house:",
                        },
                    },
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": "Here's what you can do...",
                        },
                    },
                ],
            },
        )
    except Exception as e:
        logger.error(f"Error publishing home tab: {e}")
```

---

## Select Menu Options

Handle dynamic options for external select menus with `options()`. **Must `ack()` with options.**

```python
@app.options("external_action")
def show_options(ack, payload):
    options = [
        {"text": {"type": "plain_text", "text": "Option 1"}, "value": "1-1"},
        {"text": {"type": "plain_text", "text": "Option 2"}, "value": "1-2"},
    ]
    keyword = payload.get("value")
    if keyword:
        options = [o for o in options if keyword in o["text"]["text"]]
    ack(options=options)
```

---

## Middleware

### Global Middleware

Runs for all incoming requests. Must call `next()`.

```python
@app.use
def auth_middleware(client, context, logger, payload, next):
    slack_user_id = payload["user"]
    try:
        user = db.lookup_by_id(slack_user_id)
        context["user"] = user
    except Exception:
        client.chat_postEphemeral(
            channel=payload["channel"],
            user=slack_user_id,
            text=f"Sorry <@{slack_user_id}>, you aren't authorized.",
        )
        return  # don't call next() to block the request
    next()
```

### Listener Middleware

Only runs for the listener it is attached to.

```python
def no_bot_messages(message, next):
    if "bot_id" not in message:
        next()

@app.event(event="message", middleware=[no_bot_messages])
def log_message(logger, event):
    logger.info(f"User: {event['user']}, Message: {event['text']}")
```

### Listener Matchers (Simplified)

Return `bool` instead of calling `next()`:

```python
def no_bot_messages(message) -> bool:
    return "bot_id" not in message

@app.event(event="message", matchers=[no_bot_messages])
def log_message(logger, event):
    logger.info(f"User: {event['user']}, Message: {event['text']}")
```

---

## Context

All listeners have access to a `context` dictionary. Bolt attaches: `user_id`, `team_id`, `channel_id`, `enterprise_id`.

### Context Through Middleware Chain

```python
def fetch_tasks(context, event, next):
    user = event["user"]
    context["tasks"] = db.get_tasks(user) or []
    next()

def create_sections(context, next):
    context["blocks"] = [
        {
            "type": "section",
            "text": {"type": "mrkdwn", "text": f"*{t['title']}*\n{t['body']}"},
            "accessory": {
                "type": "button",
                "text": {"type": "plain_text", "text": "See task"},
                "url": t["url"],
            },
        }
        for t in context["tasks"]
    ]
    next()

@app.event(event="app_home_opened", middleware=[fetch_tasks, create_sections])
def show_tasks(event, client, context):
    client.views_publish(
        user_id=event["user"],
        view={"type": "home", "blocks": context["blocks"]},
    )
```

---

## Sending Messages

### Simple Text

```python
@app.message("knock knock")
def ask_who(message, say):
    say("_Who's there?_")
```

### Rich Message with Blocks

```python
@app.event("reaction_added")
def show_datepicker(event, say):
    if event["reaction"] == "calendar":
        say(
            blocks=[
                {
                    "type": "section",
                    "text": {"type": "mrkdwn", "text": "Pick a date"},
                    "accessory": {
                        "type": "datepicker",
                        "action_id": "datepicker_remind",
                        "initial_date": "2024-01-01",
                        "placeholder": {"type": "plain_text", "text": "Select a date"},
                    },
                }
            ],
            text="Pick a date",
        )
```

### Streaming Messages (AI Chatbot Style)

Uses `chat.startStream`, `chat.appendStream`, `chat.stopStream`:

```python
streamer = client.chat_stream(
    channel=channel_id,
    recipient_team_id=team_id,
    recipient_user_id=user_id,
    thread_ts=thread_ts,
)

for event in llm_response:
    if event.type == "response.output_text.delta":
        streamer.append(markdown_text=f"{event.delta}")

streamer.stop()
```
