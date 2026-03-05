# SGLDHelper

Automated operations assistant for the SGLang Diffusion subsystem, integrating Slack, GitHub, and AI (Kimi K2.5).

## What It Does

- **PR Monitoring** — Polls GitHub for Diffusion-related PRs, pushes open/update/merge/close notifications to Slack
- **CI Monitoring** — Tracks Nvidia + AMD CI workflows, auto-retries failed jobs, supports high-priority PR escalation
- **Auto Merge** — CI passed + approved + tracked PR → countdown → squash merge (cancellable via Slack)
- **Health Check** — Periodic report of all open Diffusion PRs: merge-ready, needs-review, CI-stalled
- **AI Chat** — Natural language queries via Kimi K2.5 + 16 function-calling tools
- **Passive Classification** — Detects progress updates and blockers from channel messages
- **Tracked PR Summaries** — Per-user PR tracking with periodic AI-generated status updates
- **Standup Generation** — `/diffusion-standup` slash command for daily summaries

## Prerequisites

| Credential | Where to get it |
|------------|-----------------|
| **GitHub Token** | GitHub Settings > Developer settings > Personal access tokens (`repo` scope) |
| **Slack Bot Token** (`xoxb-`) | Slack API > Create App > OAuth & Permissions > Install to Workspace |
| **Slack App Token** (`xapp-`) | Slack API > App > Basic Information > App-Level Tokens (`connections:write`) |
| **Moonshot API Key** | [platform.moonshot.cn](https://platform.moonshot.cn) |

Required Slack App permissions:
- **Socket Mode** enabled
- **Event Subscriptions**: `app_mention`, `message.channels`
- **Slash Commands**: `/diffusion-standup`
- **Bot Scopes**: `chat:write`, `commands`, `reactions:write`, `channels:history`
- **Interactivity**: enabled (for confirmation buttons)

## Quick Start

### 1. Configure

```bash
cp .env.example .env
# Edit .env with your real tokens and channel IDs
```

### 2. Run

```bash
# Docker Compose (recommended)
docker compose up -d

# Or local development
pip install -e ".[dev]"
python -m sgldhelper
```

## Pollers

Six pollers run automatically on startup:

| Poller | Default Interval | What it does |
|--------|-----------------|--------------|
| PR | 60s | Detects diffusion PR lifecycle events, posts to `SLACK_PR_CHANNEL` |
| CI | 300s | Checks CI status for tracked PRs, auto-retries failures, posts to `SLACK_CI_CHANNEL` |
| Tracked PR Summary | 12h | AI-generated status updates for each user's tracked PRs |
| Diffusion Summary | 2h | Periodic summary of diffusion PR activity |
| Health Check | 2h | Batch report: merge-ready / needs-review / CI-stalled |
| Auto Merge | event-driven | Countdown timer on CI-passed + approved tracked PRs |

## AI Chat

The bot responds to all messages in monitored channels and to @mentions in any channel. It uses Kimi K2.5 with function calling:

```
这个PR是干嘛的？                    → (uses thread context from PR notification)
CI怎么样了 #19876？                  → get_ci_status
帮我重跑一下CI                       → trigger_ci (asks confirmation)
跑一下健康检查                        → run_health_check
哪些PR可以merge了？                  → get_merge_ready_prs
merge吧                             → merge_pr (asks confirmation)
```

**16 tools**: `get_open_prs`, `get_pr_details`, `get_pr_reviews`, `search_prs`, `search_github_prs`, `get_recent_activity`, `get_my_preferences`, `update_tracked_prs`, `save_user_note`, `review_pr_code`, `get_ci_status`, `trigger_ci`, `cancel_auto_merge`, `merge_pr`, `get_merge_ready_prs`, `run_health_check`

Destructive tools (`trigger_ci`, `cancel_auto_merge`, `merge_pr`) require user confirmation before execution.

## CI Features

### Auto Retry
- Failed CI jobs are automatically retried up to **3 times** (normal) or **10 times** (high-priority PRs with `high-priority` label)
- Handles `failure`, `cancelled`, and `timed_out` conclusions

### High-Priority PR Support
PRs with the `high-priority` label get:
- Extended retry limit (10 instead of 3)
- GitHub @mention notification when Nvidia CI passes + PR is approved

### Auto Merge
When a tracked PR has CI passed + approved:
1. Bot announces countdown in Slack
2. After delay (default 5 min), squash merges
3. Cancellable via keywords in Slack ("取消merge", "cancel merge", etc.)

## Configuration Reference

All settings via environment variables (or `.env` file):

| Variable | Default | Description |
|----------|---------|-------------|
| `GITHUB_TOKEN` | — | GitHub PAT with `repo` scope |
| `SLACK_BOT_TOKEN` | — | Slack bot token (`xoxb-`) |
| `SLACK_APP_TOKEN` | — | Slack app token for Socket Mode (`xapp-`) |
| `SLACK_PR_CHANNEL` | — | Channel ID for PR notifications |
| `SLACK_CI_CHANNEL` | — | Channel ID for CI notifications |
| `MOONSHOT_API_KEY` | — | Moonshot / Kimi API key |
| `PR_POLL_INTERVAL` | 60 | PR polling interval (seconds) |
| `CI_POLL_INTERVAL` | 300 | CI polling interval (seconds) |
| `CI_MAX_RETRIES` | 3 | Max CI retry attempts per job |
| `CI_HIGH_PRIORITY_MAX_RETRIES` | 10 | Max retries for high-priority PRs |
| `CI_HIGH_PRIORITY_LABEL` | high-priority | Label name for high-priority PRs |
| `CI_HIGH_PRIORITY_PING_USER` | mickqian | GitHub user to ping when HP nvidia CI passes |
| `CI_NVIDIA_WORKFLOW_ID` | 115218617 | GitHub Actions workflow ID for Nvidia CI |
| `CI_AMD_WORKFLOW_ID` | 119055250 | GitHub Actions workflow ID for AMD CI |
| `AUTO_MERGE_ENABLED` | true | Enable auto-merge for tracked PRs |
| `AUTO_MERGE_DELAY_SECONDS` | 300 | Countdown before auto-merge |
| `TRACKED_PR_SUMMARY_INTERVAL` | 43200 | Tracked PR summary interval (seconds) |
| `DIFFUSION_SUMMARY_INTERVAL` | 7200 | Diffusion summary interval (seconds) |
| `PR_HEALTH_CHECK_INTERVAL` | 7200 | Health check interval (seconds) |
| `AI_RATE_LIMIT_RPM` | 10 | Global AI API rate limit (requests/min) |
| `AI_USER_COOLDOWN_MAX` | 5 | Max messages per user within cooldown window |
| `AI_USER_COOLDOWN_SECONDS` | 60 | User cooldown window (seconds) |
| `KIMI_MODEL` | kimi-k2.5 | Moonshot model name |
| `COLD_START_MAX_PRS` | 500 | Max PRs to fetch on first run |
| `LOG_LEVEL` | INFO | Logging level |

## Development

```bash
pip install -e ".[dev]"
pytest tests/ -v          # Run all 127 tests
pytest tests/test_ci_monitor.py -v  # Run specific test file
```

All tests mock external APIs — no real tokens needed.

## Project Structure

```
src/sgldhelper/
├── __main__.py              # Entry point: pollers + Slack + wiring
├── config.py                # pydantic-settings configuration
├── ai/
│   ├── client.py            # Kimi K2.5 OpenAI-compatible client
│   ├── tools.py             # 16 function-calling tools
│   ├── conversation.py      # Per-thread conversation + tool loop
│   ├── classifier.py        # Message classification (progress/blocker)
│   └── summaries.py         # Standup + diffusion summary generation
├── ci/
│   ├── monitor.py           # CI status checking, retry logic, HP support
│   ├── auto_merge.py        # Countdown + squash merge
│   ├── health_check.py      # Periodic batch health report
│   └── tracked_pr_summary.py # Per-user tracked PR summaries
├── db/
│   ├── engine.py            # aiosqlite connection management
│   ├── models.py            # 10-table DDL schema
│   └── queries.py           # All DB query functions
├── github/
│   ├── client.py            # GitHub REST API (ETag caching + rate limiting)
│   ├── pr_tracker.py        # PR polling + diffusion file detection
│   └── poller.py            # Generic async polling scheduler
├── slack/
│   ├── app.py               # Slack Bolt + Socket Mode wrapper
│   ├── channels.py          # Channel routing
│   ├── handlers.py          # Message + mention + action handlers
│   └── messages.py          # Block Kit message templates
├── notifications/
│   ├── dispatcher.py        # Event routing facade
│   ├── pr_events.py         # PR lifecycle notifications
│   └── ci_events.py         # CI status notifications
└── utils/
    ├── logging_setup.py     # structlog JSON configuration
    └── rate_limiter.py      # Token-bucket rate limiter
```
