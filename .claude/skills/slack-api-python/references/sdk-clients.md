# Python Slack SDK Clients — Detailed Guide

## Table of Contents
1. [Installation](#installation)
2. [WebClient](#webclient)
3. [AsyncWebClient](#asyncwebclient)
4. [WebhookClient](#webhookclient)
5. [SocketModeClient](#socketmodeclient)
6. [RTM Client](#rtm-client)
7. [Audit Logs Client](#audit-logs-client)
8. [SCIM Client](#scim-client)
9. [Common Patterns](#common-patterns)

---

## Installation

```bash
pip install slack-sdk
```

Verify:

```python
import sys, logging
logging.basicConfig(level=logging.DEBUG)
from slack_sdk import WebClient
client = WebClient()
api_response = client.api_test()
```

---

## WebClient

The `WebClient` provides access to 200+ Slack Web API methods.

### Basic Usage

```python
import os
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

client = WebClient(token=os.environ["SLACK_BOT_TOKEN"])
```

### Send a Message

```python
try:
    response = client.chat_postMessage(
        channel="C0XXXXXX",
        text="Hello from Python!",
    )
    print(f"Message ts: {response['ts']}")
except SlackApiError as e:
    print(f"Error: {e.response['error']}")
```

### Send with Blocks

```python
response = client.chat_postMessage(
    channel="C0XXXXXX",
    text="Fallback text",
    blocks=[
        {
            "type": "section",
            "text": {"type": "mrkdwn", "text": "*Hello* from Block Kit!"},
        },
        {
            "type": "actions",
            "elements": [
                {
                    "type": "button",
                    "text": {"type": "plain_text", "text": "Click Me"},
                    "action_id": "button_click",
                    "value": "clicked",
                }
            ],
        },
    ],
)
```

### Send Ephemeral Message (Visible to One User)

```python
client.chat_postEphemeral(
    channel="C0XXXXXX",
    user="U0XXXXXX",
    text="Only you can see this",
)
```

### Schedule a Message

```python
import datetime

scheduled_time = datetime.datetime(2024, 6, 1, 9, 0, 0)
response = client.chat_scheduleMessage(
    channel="C0XXXXXX",
    text="Good morning!",
    post_at=int(scheduled_time.timestamp()),
)
```

### Update a Message

```python
client.chat_update(
    channel="C0XXXXXX",
    ts="1234567890.123456",
    text="Updated message content",
)
```

### Delete a Message

```python
client.chat_delete(
    channel="C0XXXXXX",
    ts="1234567890.123456",
)
```

### List Conversations (Channels)

```python
response = client.conversations_list()
for channel in response["channels"]:
    print(f"{channel['name']} ({channel['id']})")
```

### Paginated Results

```python
response = client.conversations_list(limit=100)
channels = response["channels"]

while response["response_metadata"].get("next_cursor"):
    response = client.conversations_list(
        limit=100,
        cursor=response["response_metadata"]["next_cursor"],
    )
    channels.extend(response["channels"])
```

### Conversation History

```python
response = client.conversations_history(channel="C0XXXXXX", limit=50)
for message in response["messages"]:
    print(f"{message.get('user', 'bot')}: {message['text']}")
```

### Thread Replies

```python
response = client.conversations_replies(
    channel="C0XXXXXX",
    ts="1234567890.123456",  # parent message ts
    limit=50,
)
for reply in response["messages"]:
    print(reply["text"])
```

### Reply in Thread

```python
client.chat_postMessage(
    channel="C0XXXXXX",
    thread_ts="1234567890.123456",
    text="This is a threaded reply",
)
```

### User Info

```python
response = client.users_info(user="U0XXXXXX")
user = response["user"]
print(f"Name: {user['real_name']}, Email: {user['profile'].get('email')}")
```

### List Users

```python
response = client.users_list()
for member in response["members"]:
    if not member["deleted"]:
        print(f"{member['name']} ({member['id']})")
```

### Upload a File

```python
response = client.files_upload_v2(
    channels=["C0XXXXXX"],
    file="./report.csv",
    title="Daily Report",
    initial_comment="Here's today's report",
)
```

### Upload File Content from String

```python
response = client.files_upload_v2(
    channels=["C0XXXXXX"],
    content="Hello, this is file content",
    filename="hello.txt",
    title="Hello File",
)
```

### Add Reaction

```python
client.reactions_add(
    channel="C0XXXXXX",
    timestamp="1234567890.123456",
    name="thumbsup",
)
```

### Open/Update/Push Views

```python
# Open a modal
client.views_open(trigger_id="xxx", view={...})

# Update a modal
client.views_update(view_id="xxx", hash="xxx", view={...})

# Push onto modal stack
client.views_push(trigger_id="xxx", view={...})

# Publish App Home
client.views_publish(user_id="U0XXXXXX", view={"type": "home", "blocks": [...]})
```

### Pin/Unpin Messages

```python
client.pins_add(channel="C0XXXXXX", timestamp="1234567890.123456")
client.pins_remove(channel="C0XXXXXX", timestamp="1234567890.123456")
```

### Set Channel Topic/Purpose

```python
client.conversations_setTopic(channel="C0XXXXXX", topic="New topic")
client.conversations_setPurpose(channel="C0XXXXXX", purpose="New purpose")
```

### Create a Channel

```python
response = client.conversations_create(name="new-channel", is_private=False)
channel_id = response["channel"]["id"]
```

### Invite Users to Channel

```python
client.conversations_invite(channel="C0XXXXXX", users="U0XXXXXX,U1XXXXXX")
```

---

## AsyncWebClient

For async/await usage:

```python
import asyncio
from slack_sdk.web.async_client import AsyncWebClient

client = AsyncWebClient(token=os.environ["SLACK_BOT_TOKEN"])

async def send_message():
    response = await client.chat_postMessage(
        channel="C0XXXXXX",
        text="Hello async!",
    )
    print(response["ts"])

asyncio.run(send_message())
```

---

## WebhookClient

Send messages via Incoming Webhooks (no bot token needed).

```python
from slack_sdk.webhook import WebhookClient

webhook = WebhookClient(url=os.environ["SLACK_WEBHOOK_URL"])

# Simple text
response = webhook.send(text="Hello from webhook!")
print(f"Status: {response.status_code}")

# With blocks
response = webhook.send(
    text="Fallback text",
    blocks=[
        {
            "type": "section",
            "text": {"type": "mrkdwn", "text": "*Hello* from webhook!"},
        }
    ],
)
```

### Async Webhook

```python
from slack_sdk.webhook.async_client import AsyncWebhookClient

async_webhook = AsyncWebhookClient(url=os.environ["SLACK_WEBHOOK_URL"])
response = await async_webhook.send(text="Hello async webhook!")
```

### response_url Client

Used to respond to interactions (same interface as WebhookClient):

```python
from slack_sdk.webhook import WebhookClient

url = body["response_url"]
webhook = WebhookClient(url=url)
webhook.send(
    text="Response via response_url",
    response_type="ephemeral",  # or "in_channel"
    replace_original=True,
)
```

---

## SocketModeClient

Real-time bidirectional communication via WebSocket. Requires an App-Level Token with `connections:write` scope.

### Built-in Client

```python
import os
from slack_sdk.socket_mode import SocketModeClient
from slack_sdk.socket_mode.request import SocketModeRequest
from slack_sdk.socket_mode.response import SocketModeResponse

client = SocketModeClient(
    app_token=os.environ["SLACK_APP_TOKEN"],
    web_client=WebClient(token=os.environ["SLACK_BOT_TOKEN"]),
)

def process(client: SocketModeClient, req: SocketModeRequest):
    if req.type == "events_api":
        # Acknowledge the event
        response = SocketModeResponse(envelope_id=req.envelope_id)
        client.send_socket_mode_response(response)

        event = req.payload["event"]
        if event["type"] == "message":
            client.web_client.chat_postMessage(
                channel=event["channel"],
                text=f"Echo: {event['text']}",
            )

client.socket_mode_request_listeners.append(process)
client.connect()

# Keep alive
from threading import Event
Event().wait()
```

### With Bolt (Recommended)

```python
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

app = App(token=os.environ["SLACK_BOT_TOKEN"])
handler = SocketModeHandler(app, os.environ["SLACK_APP_TOKEN"])
handler.start()
```

### Async Socket Mode

```python
from slack_sdk.socket_mode.aiohttp import SocketModeClient as AsyncSocketModeClient

client = AsyncSocketModeClient(
    app_token=os.environ["SLACK_APP_TOKEN"],
    web_client=AsyncWebClient(token=os.environ["SLACK_BOT_TOKEN"]),
)
```

---

## RTM Client

Real-time messaging via WebSocket (legacy, use Socket Mode for new apps).

```python
import os
from slack_sdk.rtm_v2 import RTMClient

rtm = RTMClient(token=os.environ["SLACK_BOT_TOKEN"])

@rtm.on("message")
def handle(client: RTMClient, event: dict):
    if "Hello" in event.get("text", ""):
        channel_id = event["channel"]
        client.web_client.chat_postMessage(channel=channel_id, text="Hi there!")

rtm.start()
```

---

## Audit Logs Client

For Enterprise Grid organizations.

```python
import os
from slack_sdk.audit_logs import AuditLogsClient

client = AuditLogsClient(token=os.environ["SLACK_ORG_ADMIN_USER_TOKEN"])

# Get schemas
response = client.schemas()

# Get actions
response = client.actions()

# Get logs with filters
response = client.logs(
    action="user_login",
    limit=10,
)
for entry in response.typed_body.get("entries", []):
    print(entry)
```

### Async Audit Logs

```python
from slack_sdk.audit_logs.v1.async_client import AsyncAuditLogsClient

client = AsyncAuditLogsClient(token=os.environ["SLACK_ORG_ADMIN_USER_TOKEN"])
response = await client.logs(action="user_login", limit=10)
```

---

## SCIM Client

User/group provisioning for Enterprise Grid.

```python
import os
from slack_sdk.scim import SCIMClient

client = SCIMClient(token=os.environ["SLACK_ORG_ADMIN_USER_TOKEN"])

# Search users
response = client.search_users(filter='userName eq "john.doe"', count=1)
for user in response.users:
    print(f"{user.user_name} ({user.id})")

# Read a user
response = client.read_user(id="U0XXXXXX")

# Create a user
from slack_sdk.scim.v1.user import User, UserName, UserEmail
response = client.create_user(
    user=User(
        user_name="new.user",
        name=UserName(given_name="New", family_name="User"),
        emails=[UserEmail(value="new.user@example.com")],
    )
)

# Search groups
response = client.search_groups(filter='displayName eq "Engineering"', count=1)
```

---

## Common Patterns

### Error Handling

```python
from slack_sdk.errors import SlackApiError

try:
    response = client.chat_postMessage(channel="C0XXXXXX", text="Hello")
except SlackApiError as e:
    error_code = e.response["error"]
    if error_code == "channel_not_found":
        print("Channel not found")
    elif error_code == "not_in_channel":
        print("Bot is not in the channel")
    elif error_code == "ratelimited":
        retry_after = int(e.response.headers.get("Retry-After", 1))
        print(f"Rate limited. Retry after {retry_after}s")
    else:
        print(f"Slack API error: {error_code}")
```

### Rate Limiting

The SDK has built-in rate limit handling. For manual retry:

```python
import time
from slack_sdk.errors import SlackApiError

def send_with_retry(client, channel, text, max_retries=3):
    for attempt in range(max_retries):
        try:
            return client.chat_postMessage(channel=channel, text=text)
        except SlackApiError as e:
            if e.response["error"] == "ratelimited":
                delay = int(e.response.headers.get("Retry-After", 1))
                time.sleep(delay)
            else:
                raise
    raise Exception("Max retries exceeded")
```

### Request Signature Verification

For HTTP mode, verify requests are from Slack:

```python
from slack_sdk.signature import SignatureVerifier

verifier = SignatureVerifier(signing_secret=os.environ["SLACK_SIGNING_SECRET"])

# In your HTTP handler:
is_valid = verifier.is_valid_request(request_body, request_headers)
```

### Message Formatting

```python
# User mention
text = f"Hello <@{user_id}>!"

# Channel link
text = f"Check out <#{channel_id}>!"

# URL with custom text
text = "Visit <https://example.com|our website>"

# Bold, italic, strikethrough
text = "*bold* _italic_ ~strikethrough~"

# Code
text = "Use `inline code` or ```code block```"

# Blockquote
text = "> This is a quote"

# Ordered/unordered list
text = "1. First\n2. Second\n• Bullet\n• Another"
```

### Token Types Reference

| Token | Prefix | Environment Variable | Purpose |
|-------|--------|---------------------|---------|
| Bot Token | `xoxb-` | `SLACK_BOT_TOKEN` | API calls as bot |
| User Token | `xoxp-` | `SLACK_USER_TOKEN` | API calls as user |
| App-Level Token | `xapp-` | `SLACK_APP_TOKEN` | Socket Mode connection |
| Signing Secret | (hex string) | `SLACK_SIGNING_SECRET` | HTTP request verification |
| Webhook URL | `https://hooks.slack.com/...` | `SLACK_WEBHOOK_URL` | Incoming webhooks |
