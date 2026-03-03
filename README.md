# SGLDHelper

Automated operations assistant for the SGLang Diffusion subsystem, integrating Slack and GitHub with two layers of functionality:

**Core Layer (enabled by default)** — Notification bot:
- Polls GitHub to monitor Diffusion-related PRs, CI runs, and feature roadmap
- Pushes notifications to Slack channels
- Automatically classifies CI failures (flaky / infra / code) and auto-reruns
- 3 slash commands + interactive buttons

**AI Layer (opt-in)** — Conversational assistant (Kimi K2.5):
- `@bot` natural language queries answered via Kimi K2.5 + function calling
- Passively listens to messages, detects progress updates and blockers
- Alerts on stalled features and PRs missing review
- `/diffusion-standup` daily summary generation

## Prerequisites

You need three sets of credentials:

| Credential | Where to get it |
|------------|-----------------|
| **GitHub Token** | GitHub Settings → Developer settings → Personal access tokens (needs `repo` scope) |
| **Slack Bot Token** (`xoxb-`) | Slack API → Create App → OAuth & Permissions → Install to Workspace |
| **Slack App Token** (`xapp-`) | Slack API → App → Basic Information → App-Level Tokens (scope: `connections:write`) |

Required Slack App permissions and features:
- **Socket Mode** — no public URL needed
- **Event Subscriptions** — `app_mention`, `message.channels` (required for AI layer)
- **Slash Commands** — `/diffusion-status`, `/diffusion-rerun`, `/diffusion-features`, `/diffusion-standup`
- **Bot Scopes** — `chat:write`, `commands`, `reactions:write`, `channels:history`

For the AI layer, you also need a **Moonshot API Key** ([platform.moonshot.cn](https://platform.moonshot.cn)).

## Quick Start

### 1. Configure Environment Variables

```bash
cp .env.example .env
# Edit .env and fill in your real tokens and channel IDs
```

Required fields:

```
GITHUB_TOKEN=ghp_your_token
SLACK_BOT_TOKEN=xoxb-your_token
SLACK_APP_TOKEN=xapp-your_token
SLACK_PR_CHANNEL=C0123456789       # Channel ID for PR notifications
SLACK_CI_CHANNEL=C0123456789       # Channel ID for CI notifications
SLACK_FEATURES_CHANNEL=C0123456789 # Channel ID for feature notifications
```

### 2. Run (pick one)

```bash
# Option A: Docker Compose (recommended for production)
docker compose up -d

# Option B: Docker build directly
docker build -t sgldhelper .
docker run --env-file .env -v sgldhelper-data:/app/data sgldhelper

# Option C: Local development
pip install -e ".[dev]"
python -m sgldhelper
```

On startup you should see:

```
starting repo=sgl-project/sglang
slack.connected
ai.disabled            # if AI is not enabled
```

## Core Features

Three pollers run automatically on startup:

| Poller | Interval | What it does |
|--------|----------|--------------|
| PR Poller | 60s | Detects diffusion PR open/update/merge/close events, pushes to `SLACK_PR_CHANNEL` |
| CI Poller | 120s | Analyzes CI results, classifies failures, pushes to `SLACK_CI_CHANNEL`, auto-reruns flaky/infra failures |
| Feature Poller | 1h | Parses roadmap issue checkboxes, pushes progress to `SLACK_FEATURES_CHANNEL` |

### Slash Commands

```
/diffusion-status          # List all open diffusion PRs
/diffusion-rerun 1234      # Manually rerun failed CI for PR #1234
/diffusion-features        # Show feature roadmap progress
```

### Interactive Buttons

- CI failure notifications include a **"Rerun CI"** button
- PR notifications include a **"CI Status"** button

## AI Layer (Optional)

Set the following in `.env`:

```
AI_ENABLED=true
MOONSHOT_API_KEY=sk-your_moonshot_key
```

After restarting you should see:

```
ai.enabled model=kimi-k2.5 base_url=https://api.moonshot.ai/v1
```

### Natural Language Chat (@mention)

Mention the bot in any channel:

```
@SGLDHelper What's wrong with CI on PR 1234?
@SGLDHelper Which PRs are currently open?
@SGLDHelper How is the feature roadmap progressing?
@SGLDHelper Rerun CI for PR 5678      # asks for confirmation first
```

The bot uses function calling to query internal tools for real data, then replies in natural language. Multi-turn conversations are supported within the same thread.

### Passive Message Classification

Messages in the 3 monitored channels are automatically classified:

```
"ControlNet PR is done, already merged"       → detected as progress_update → confirmation button
"I'm blocked on SDXL, getting GPU OOM"        → detected as blocker → confirmation button
```

Users click **Confirm** to record the update, or **Dismiss** to ignore (avoids false positives).

### Stall Alerts

Runs automatically every 12 hours:

- Feature items with linked PRs that haven't been updated in 3+ days → `:snail: Feature Stall Detected`
- Open PRs without review approval for 2+ days → `:eyes: Review Needed`

Thresholds are configurable via `STALL_DAYS_THRESHOLD` and `REVIEW_NUDGE_DAYS`.

### Standup Summary

```
/diffusion-standup
```

Collects the last 24 hours of PR, CI, and feature activity plus confirmed blockers, then generates a summary using K2.5.

## Configuration Reference

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `AI_RATE_LIMIT_RPM` | 10 | Global AI API rate limit (requests/min) |
| `AI_USER_COOLDOWN_MAX` | 5 | Max @mentions per user within 60s |
| `STALL_DAYS_THRESHOLD` | 3 | Days before a feature is considered stalled |
| `REVIEW_NUDGE_DAYS` | 2 | Days before a PR review nudge is sent |
| `STALL_CHECK_INTERVAL` | 43200 | Stall check interval in seconds (default 12h) |
| `KIMI_MODEL` | kimi-k2.5 | Moonshot model name (can be swapped) |
| `MAX_AUTO_RERUNS` | 2 | Max auto-rerun attempts per CI run |

## Development & Testing

```bash
pip install -e ".[dev]"
pytest tests/ -v              # Run all 75 tests
pytest tests/test_ai_tools.py # Run AI tool tests only
```

All AI tests mock the OpenAI client — no real API key is needed to run them.

## Project Structure

```
src/sgldhelper/
├── __main__.py              # Entry point: starts all pollers + Slack
├── config.py                # pydantic-settings env var configuration
├── ai/                      # AI conversational layer (optional)
│   ├── client.py            #   Kimi K2.5 OpenAI-compatible client
│   ├── tools.py             #   10 function calling tool definitions
│   ├── conversation.py      #   Per-thread conversation + tool loop
│   ├── classifier.py        #   Instant-mode message classification
│   ├── stall_detector.py    #   Stall detection (pure logic, no LLM)
│   └── summaries.py         #   Standup summary generation
├── db/
│   ├── engine.py            #   aiosqlite connection management
│   ├── models.py            #   9-table DDL schema
│   └── queries.py           #   All DB query functions
├── github/
│   ├── client.py            #   GitHub REST API (ETag caching + rate limiting)
│   ├── pr_tracker.py        #   PR polling + diffusion file detection
│   ├── ci_analyzer.py       #   CI failure classification
│   ├── ci_rerunner.py       #   Auto/manual CI rerun
│   ├── issue_tracker.py     #   Feature roadmap checkbox parsing
│   └── poller.py            #   Generic async polling scheduler
├── slack/
│   ├── app.py               #   Slack Bolt + Socket Mode wrapper
│   ├── channels.py          #   Channel routing
│   ├── handlers.py          #   Slash commands + buttons + AI handlers
│   └── messages.py          #   Block Kit message templates
├── notifications/
│   ├── dispatcher.py        #   Event routing
│   ├── pr_events.py         #   PR lifecycle notifications
│   ├── ci_events.py         #   CI notifications
│   └── feature_events.py    #   Feature notifications
└── utils/
    ├── logging_setup.py     #   structlog configuration
    └── rate_limiter.py      #   Token-bucket rate limiter
```
