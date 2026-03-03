# Authentication & Deployment — Detailed Guide

## Table of Contents
1. [Token Types](#token-types)
2. [Single Workspace Setup](#single-workspace-setup)
3. [OAuth for Multi-Workspace](#oauth-for-multi-workspace)
4. [Custom Authorization](#custom-authorization)
5. [Socket Mode](#socket-mode)
6. [HTTP Mode with Signing Secret](#http-mode-with-signing-secret)
7. [Flask Adapter](#flask-adapter)
8. [Django Adapter](#django-adapter)
9. [AWS Lambda (Serverless)](#aws-lambda-serverless)
10. [Lazy Listeners for FaaS](#lazy-listeners-for-faas)

---

## Token Types

| Token | Prefix | How to Get | Usage |
|-------|--------|------------|-------|
| Bot Token | `xoxb-` | Install app to workspace → OAuth & Permissions | API calls as bot user |
| User Token | `xoxp-` | OAuth flow with user consent | API calls on behalf of user |
| App-Level Token | `xapp-` | App settings → Basic Information → App-Level Tokens | Socket Mode (with `connections:write` scope) |
| Signing Secret | hex string | App settings → Basic Information | Verify HTTP requests from Slack |

**Security:** Never hardcode tokens. Always use environment variables.

```bash
export SLACK_BOT_TOKEN=xoxb-your-bot-token
export SLACK_APP_TOKEN=xapp-your-app-token
export SLACK_SIGNING_SECRET=your-signing-secret
```

```python
import os
token = os.environ["SLACK_BOT_TOKEN"]
```

---

## Single Workspace Setup

For apps used in only one workspace:

1. Go to https://api.slack.com/apps → Create New App
2. Add Bot Token Scopes under OAuth & Permissions
3. Install to Workspace
4. Copy the Bot User OAuth Token (`xoxb-...`)

```python
import os
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

app = App(token=os.environ["SLACK_BOT_TOKEN"])

# Add handlers...

SocketModeHandler(app, os.environ["SLACK_APP_TOKEN"]).start()
```

---

## OAuth for Multi-Workspace

For apps distributed to multiple workspaces.

### Bolt OAuth Setup

```python
import os
from slack_bolt import App
from slack_bolt.oauth.oauth_settings import OAuthSettings
from slack_sdk.oauth.installation_store import FileInstallationStore
from slack_sdk.oauth.state_store import FileOAuthStateStore

oauth_settings = OAuthSettings(
    client_id=os.environ["SLACK_CLIENT_ID"],
    client_secret=os.environ["SLACK_CLIENT_SECRET"],
    scopes=["channels:read", "groups:read", "chat:write", "commands"],
    installation_store=FileInstallationStore(base_dir="./data/installations"),
    state_store=FileOAuthStateStore(expiration_seconds=600, base_dir="./data/states"),
)

app = App(
    signing_secret=os.environ["SLACK_SIGNING_SECRET"],
    oauth_settings=oauth_settings,
)
```

Bolt auto-creates two routes:
- **`/slack/install`** — Shows "Add to Slack" button
- **`/slack/oauth_redirect`** — Handles OAuth callback

### Custom OAuth Callbacks

```python
from slack_bolt.oauth.callback_options import CallbackOptions, SuccessArgs, FailureArgs
from slack_bolt.response import BoltResponse

def success(args: SuccessArgs) -> BoltResponse:
    return BoltResponse(status=200, body="App installed successfully!")

def failure(args: FailureArgs) -> BoltResponse:
    return BoltResponse(
        status=args.suggested_status_code,
        body=f"Installation failed: {args.reason}",
    )

oauth_settings = OAuthSettings(
    client_id=os.environ["SLACK_CLIENT_ID"],
    client_secret=os.environ["SLACK_CLIENT_SECRET"],
    scopes=["chat:write", "commands"],
    installation_store=FileInstallationStore(base_dir="./data/installations"),
    state_store=FileOAuthStateStore(expiration_seconds=600, base_dir="./data/states"),
    callback_options=CallbackOptions(success=success, failure=failure),
    install_path="/slack/install",
    redirect_uri_path="/slack/oauth_redirect",
)
```

### SDK-Level OAuth (Without Bolt)

```python
import os
from slack_sdk import WebClient
from flask import Flask, request

client_id = os.environ["SLACK_CLIENT_ID"]
client_secret = os.environ["SLACK_CLIENT_SECRET"]
oauth_scope = os.environ["SLACK_SCOPES"]

flask_app = Flask(__name__)

@flask_app.route("/slack/install", methods=["GET"])
def pre_install():
    state = "randomly-generated-one-time-value"
    return (
        f'<a href="https://slack.com/oauth/v2/authorize?'
        f'scope={oauth_scope}&client_id={client_id}&state={state}">'
        f'Add to Slack</a>'
    )

@flask_app.route("/slack/oauth_redirect", methods=["GET"])
def post_install():
    code = request.args["code"]
    client = WebClient()
    response = client.oauth_v2_access(
        client_id=client_id,
        client_secret=client_secret,
        code=code,
    )
    # Store response["access_token"], response["team"]["id"], etc.
    return "Installation successful!"
```

### Installation Stores

| Store | Package | Usage |
|-------|---------|-------|
| `FileInstallationStore` | `slack_sdk` | File-based (development) |
| `SQLAlchemyInstallationStore` | `slack_bolt[sqlalchemy]` | SQLAlchemy databases |
| `AmazonS3InstallationStore` | `slack_bolt[s3]` | AWS S3 |
| Custom | Implement `InstallationStore` | Any backend |

---

## Custom Authorization

For multi-workspace apps without full OAuth:

```python
from slack_bolt import App
from slack_bolt.authorization import AuthorizeResult

installations = [
    {
        "enterprise_id": "E1234",
        "team_id": "T12345",
        "bot_token": "xoxb-123abc",
        "bot_id": "B1251",
        "bot_user_id": "U12385",
    },
]

def authorize(enterprise_id, team_id, logger):
    for team in installations:
        is_valid = (
            team.get("enterprise_id", enterprise_id) == enterprise_id
            and team["team_id"] == team_id
        )
        if is_valid:
            return AuthorizeResult(
                enterprise_id=enterprise_id,
                team_id=team_id,
                bot_token=team["bot_token"],
                bot_id=team["bot_id"],
                bot_user_id=team["bot_user_id"],
            )
    logger.error("No authorization found")

app = App(
    signing_secret=os.environ["SLACK_SIGNING_SECRET"],
    authorize=authorize,
)
```

---

## Socket Mode

No public URL needed. Uses WebSocket connection.

### Requirements
- App-Level Token with `connections:write` scope
- Enable Socket Mode in app settings

### Sync

```python
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

app = App(token=os.environ["SLACK_BOT_TOKEN"])
handler = SocketModeHandler(app, os.environ["SLACK_APP_TOKEN"])
handler.start()
```

### Async

```python
from slack_bolt.app.async_app import AsyncApp
from slack_bolt.adapter.socket_mode.async_handler import AsyncSocketModeHandler

app = AsyncApp(token=os.environ["SLACK_BOT_TOKEN"])
handler = AsyncSocketModeHandler(app, os.environ["SLACK_APP_TOKEN"])
await handler.start_async()
```

### Available Socket Mode Adapters

| Library | Adapter |
|---------|---------|
| `slack_sdk` (built-in) | `slack_bolt.adapter.socket_mode` |
| `websocket-client` | `slack_bolt.adapter.socket_mode.websocket_client` |
| `aiohttp` (async) | `slack_bolt.adapter.socket_mode.aiohttp` |
| `websockets` (async) | `slack_bolt.adapter.socket_mode.websockets` |

---

## HTTP Mode with Signing Secret

For apps behind a public URL.

```python
app = App(
    token=os.environ["SLACK_BOT_TOKEN"],
    signing_secret=os.environ["SLACK_SIGNING_SECRET"],
)

# Bolt automatically verifies request signatures
app.start(port=3000)
```

Request URL format: `https://your-domain.com/slack/events`

### Manual Signature Verification (SDK only)

```python
from slack_sdk.signature import SignatureVerifier

verifier = SignatureVerifier(signing_secret=os.environ["SLACK_SIGNING_SECRET"])
is_valid = verifier.is_valid_request(request_body, request_headers)
```

---

## Flask Adapter

```python
import os
from slack_bolt import App
from slack_bolt.adapter.flask import SlackRequestHandler
from flask import Flask, request

bolt_app = App(
    signing_secret=os.environ["SLACK_SIGNING_SECRET"],
    token=os.environ["SLACK_BOT_TOKEN"],
)

@bolt_app.command("/hello")
def hello(body, ack):
    ack(f"Hi <@{body['user_id']}>!")

flask_app = Flask(__name__)
handler = SlackRequestHandler(bolt_app)

@flask_app.route("/slack/events", methods=["POST"])
def slack_events():
    return handler.handle(request)

@flask_app.route("/slack/install", methods=["GET"])
def install():
    return handler.handle(request)

@flask_app.route("/slack/oauth_redirect", methods=["GET"])
def oauth_redirect():
    return handler.handle(request)
```

---

## Django Adapter

```python
# views.py
import os
from slack_bolt import App
from slack_bolt.adapter.django import SlackRequestHandler

app = App(
    signing_secret=os.environ["SLACK_SIGNING_SECRET"],
    token=os.environ["SLACK_BOT_TOKEN"],
)

@app.command("/hello")
def hello(body, ack):
    ack(f"Hi <@{body['user_id']}>!")

handler = SlackRequestHandler(app)

from django.http import HttpRequest
def slack_events(request: HttpRequest):
    return handler.handle(request)

# urls.py
from django.urls import path
from . import views

urlpatterns = [
    path("slack/events", views.slack_events, name="slack_events"),
]
```

---

## AWS Lambda (Serverless)

### Key Requirement

Set `process_before_response=True` — in FaaS, you cannot run threads after returning the HTTP response.

```python
import os
from slack_bolt import App
from slack_bolt.adapter.aws_lambda import SlackRequestHandler

app = App(
    signing_secret=os.environ["SLACK_SIGNING_SECRET"],
    token=os.environ["SLACK_BOT_TOKEN"],
    process_before_response=True,
)

@app.command("/hello")
def hello(ack, body):
    ack(f"Hi <@{body['user_id']}>!")

def handler(event, context):
    slack_handler = SlackRequestHandler(app=app)
    return slack_handler.handle(event, context)
```

### IAM Permissions Required

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": ["lambda:InvokeFunction", "lambda:GetFunction"],
            "Resource": "*"
        }
    ]
}
```

---

## Lazy Listeners for FaaS

Only available in Bolt for Python. Split `ack()` and long-running work into separate functions.

```python
app = App(
    token=os.environ["SLACK_BOT_TOKEN"],
    signing_secret=os.environ["SLACK_SIGNING_SECRET"],
    process_before_response=True,
)

def respond_to_slack_within_3_seconds(body, ack):
    text = body.get("text")
    if not text:
        ack(":x: Usage: /process (description)")
    else:
        ack(f"Accepted! Processing: {body['text']}")

import time
def run_long_process(respond, body):
    time.sleep(5)  # Simulate long work
    respond(f"Completed! Result for: {body['text']}")

# Register with ack function and lazy functions
app.command("/process")(
    ack=respond_to_slack_within_3_seconds,
    lazy=[run_long_process],
)
```

### How Lazy Listeners Work

1. **`ack` function** — Runs first, must respond within 3 seconds
2. **`lazy` functions** — Run asynchronously after `ack` completes
3. In Lambda: lazy functions are invoked as separate Lambda invocations
4. In non-FaaS: lazy functions run in background threads

### Lambda with Lazy Listeners

```python
from slack_bolt import App
from slack_bolt.adapter.aws_lambda import SlackRequestHandler

app = App(process_before_response=True)

def ack_command(body, ack):
    ack(f"Processing your request...")

def process_command(respond, body, client):
    # Long-running work here
    result = do_expensive_computation(body["text"])
    respond(f"Done! Result: {result}")

app.command("/compute")(ack=ack_command, lazy=[process_command])

def handler(event, context):
    slack_handler = SlackRequestHandler(app=app)
    return slack_handler.handle(event, context)
```
