---
name: sgldhelper-architecture
description: "Architecture guide for the SGLDHelper project. Use this skill when modifying core components: adding new pollers, CI features, notification handlers, callback wiring, or changing the startup flow in __main__.py. Also trigger when working with the database schema, CI monitor, auto-merge, or health check systems."
---

# SGLDHelper Architecture

## Component Lifecycle

All components are created and wired in `src/sgldhelper/__main__.py`:

```
Database → GitHubClient → PRTracker → SlackApp → ChannelRouter → NotificationDispatcher
    → CIMonitor → AutoMergeManager → PRHealthChecker
    → ToolRegistry (set_ci_components) → ConversationManager
    → Pollers (PR, CI, summaries, health check)
```

**Order matters**: Components that are injected into others must be created first. Example: `health_checker` must exist before `tool_registry.set_ci_components()`.

## Callback Wiring Pattern

Components use a callback pattern instead of direct coupling:

```python
# In component class:
def set_callbacks(self, *, on_x: Any = None, on_y: Any = None) -> None:
    self._on_x = on_x
    self._on_y = on_y

# In __main__.py:
def _make_x_handler(dep1, dep2):
    async def handler(pr_number, user_ids, ...):
        await dep1.do_something(pr_number)
    return handler

component.set_callbacks(on_x=_make_x_handler(dep1, dep2))
```

When adding new callbacks:
1. Add the slot to `__init__` (default `None`) and `set_callbacks`
2. Create factory function in `__main__.py`
3. Pass in the `set_callbacks()` call

## Database

- SQLite via `aiosqlite` (single file, no server)
- Schema: `db/models.py` — 10 tables, all `CREATE TABLE IF NOT EXISTS`
- Queries: `db/queries.py` — all functions take `conn: aiosqlite.Connection` as first arg
- Auto-creates on first connect via `Database.connect()`
- Key tables:
  - `pull_requests` — PR metadata + `slack_thread_ts` for thread context
  - `ci_snapshots` — per PR+SHA CI state, `snapshot_data` JSON for extensible metadata
  - `ci_retry_state` — per PR+SHA+job retry count
  - `conversations` — per-thread chat history (user + assistant + tool messages)
  - `tracked_pr_summaries` — AI-generated summaries
  - `user_memories` — per-user preferences, tracked PRs, notes

## Notification Context

**Critical rule**: Any bot message that users might reply to MUST use `post_message_with_context(db_conn=db.conn)`. This saves the message to `conversations` table so the AI has thread context.

- `ci_events.py` — uses `post_message_with_context` via `_post()` helper
- `pr_events.py` — uses `post_message_with_context` for all PR notifications
- `health_check.py` — uses `post_message_with_context`
- `__main__.py` diffusion summary — uses plain `post_message` (fire-and-forget OK here)

## CI Monitor Internals

`ci/monitor.py` is the most complex module:

- `CIStatus` dataclass — snapshot of CI state for a PR at a SHA
- `check_pr_ci()` — fetches GitHub workflow runs + jobs, classifies overall status
- `should_retry()` — checks retry count against limit (3 normal, 10 high-priority)
- `_nvidia_ci_passed()` — checks nvidia jobs only (for high-priority ping)
- `_poll_single_pr()` — the main loop body:
  1. Fetch PR + CI status
  2. Get previous snapshot for change detection
  3. Get review state + commit count
  4. Save snapshot (upsert)
  5. High-priority nvidia ping (once per SHA, tracked via `snapshot_data.hp_nvidia_pinged`)
  6. Merge-ready check (every poll when CI passed)
  7. Status change notifications (only on transitions)
  8. Retry logic (when failed + all runs completed)

### Preventing Duplicate Actions

- **Duplicate pings**: Use `ci_snapshots.snapshot_data` JSON field. Set flag, upsert snapshot.
- **Status change notifications**: Compare `prev_overall` with current `ci_status.overall.value`
- **Retry counting**: `ci_retry_state` table tracks per-job retry count, resets on new SHA

## Poller System

`github/poller.py` — generic async poller:

```python
Poller(name: str, interval: int, callback: Callable)
```

- Runs `callback()` every `interval` seconds
- Catches exceptions (logs, continues)
- `stop()` for graceful shutdown
- All pollers run in `asyncio.TaskGroup` with signal handlers for SIGINT/SIGTERM
