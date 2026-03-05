# CLAUDE.md — SGLDHelper

## Project Overview

SGLDHelper is an async Python Slack bot for the SGLang Diffusion team. It monitors GitHub PRs/CI, provides AI chat (Kimi K2.5), and auto-merges approved PRs. Everything runs under a single `asyncio.run()` in `__main__.py`.

## Quick Commands

```bash
pip install -e ".[dev]"    # Install with dev dependencies
python -m sgldhelper        # Run the bot
pytest tests/ -v            # Run all tests (127 tests, all mocked)
```

## Architecture

### Entry Point & Wiring

`__main__.py` is the central wiring file. It creates all components, connects callbacks, and starts pollers. The pattern:

1. Create component (e.g., `CIMonitor`)
2. Wire callbacks via `set_callbacks(on_x=handler)` — callbacks are factory functions defined at module level
3. Register in a `Poller` for periodic execution

### Key Patterns

**Callback wiring**: Components expose `set_callbacks()` instead of direct dependencies. `__main__.py` creates factory functions (`_make_merge_ready_handler`, `_make_hp_nvidia_handler`) that close over dependencies.

**`post_message` vs `post_message_with_context`**: Use `post_message_with_context(db_conn=db.conn)` for ANY bot message that users might reply to in a thread. This stores the message in conversation history so the AI has context. Plain `post_message` is only for fire-and-forget messages.

**Tool registration**: Tools are defined in `ai/tools.py` via `_register()` calls in `_register_all()`. Each tool needs: name, description, parameters (JSON Schema), handler (async method), and optional `requires_confirmation=True`. Late-injected dependencies (ci_monitor, auto_merge, health_checker) use `set_ci_components()`.

**Poller pattern**: `github/poller.py` defines a generic `Poller(name, interval, callback)` that runs `callback` every `interval` seconds. All pollers are collected in `__main__.py` and run under `asyncio.TaskGroup`.

**Database**: SQLite via aiosqlite. Schema in `db/models.py`, all queries in `db/queries.py`. No ORM — raw SQL with `aiosqlite.Connection`.

**Config**: All config via `pydantic-settings` `Settings` class in `config.py`. Reads from `.env` + environment variables. Add new config fields there with defaults.

### CI Monitor Flow

```
poll_all_tracked_prs()
  → _poll_single_pr(pr_number, user_ids)
    → check_pr_ci() — fetches workflow runs + jobs, returns CIStatus
    → upsert_ci_snapshot() — save to DB
    → [high-priority ping logic] — check nvidia passed + approved + snapshot_data
    → [merge-ready check] — callback to auto_merge
    → [status change notifications] — CI passed / failed retrying / failed permanent
    → should_retry(is_high_priority=) — check retry count vs limit
```

### Adding a New AI Tool

1. Add handler method `async def _my_tool(self, ...) -> dict` in `ToolRegistry`
2. Add `self._register(name="my_tool", ...)` call in `_register_all()`
3. If it needs late-injected deps, add to `set_ci_components()` signature
4. Update tool count in `tests/test_ai_tools.py` (`test_schemas_are_valid` count + `test_tool_names` set)
5. If destructive, set `requires_confirmation=True`

### Adding a New Poller

1. Define `async def poll_xxx()` in `__main__.py`
2. Create `Poller("name", settings.xxx_interval, poll_xxx)`
3. Add to `pollers` list
4. Add interval config to `Settings` in `config.py`

### Adding a New Notification

1. If it's a CI notification, add method to `CIEventHandler` in `notifications/ci_events.py`
2. Use `self._post(msg)` which calls `post_message_with_context` (stores for thread context)
3. Wire as callback in `__main__.py`

## Testing Conventions

- All external APIs are mocked (GitHub, Slack, Kimi) — no real tokens needed
- Fixtures in `tests/conftest.py`: `settings`, `db`, plus per-file `mock_gh`, `ci_monitor`
- Use `@pytest.mark.asyncio` for async tests (auto mode enabled)
- Test classes group related tests: `TestCheckPrCI`, `TestRetryLogic`, `TestHighPriority`
- When adding new tools/features: update `test_ai_tools.py` tool count and name set

## File Ownership

| Area | Key Files |
|------|-----------|
| Config | `config.py`, `.env.example` |
| CI logic | `ci/monitor.py`, `ci/auto_merge.py`, `ci/health_check.py` |
| AI layer | `ai/tools.py`, `ai/conversation.py`, `ai/client.py` |
| Slack | `slack/handlers.py`, `slack/app.py`, `slack/messages.py` |
| Wiring | `__main__.py` |
| DB | `db/models.py`, `db/queries.py` |
| Notifications | `notifications/pr_events.py`, `notifications/ci_events.py` |
