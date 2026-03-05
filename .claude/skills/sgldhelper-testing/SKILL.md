---
name: sgldhelper-testing
description: "Guide for writing tests in SGLDHelper. Use when adding new test files, writing test cases for new features, debugging test failures, or setting up test fixtures. Also trigger when working with pytest, mock patterns, or the test database setup."
---

# SGLDHelper Testing Guide

## Setup

```bash
pip install -e ".[dev]"
pytest tests/ -v            # All tests
pytest tests/test_ci_monitor.py -v  # Single file
pytest tests/ -k "high_priority"    # By keyword
```

Config: `pyproject.toml` sets `asyncio_mode = "auto"` and `testpaths = ["tests"]`.

## Shared Fixtures (`tests/conftest.py`)

```python
@pytest.fixture
def settings(tmp_path):
    """Minimal Settings with fake tokens, temp DB path."""
    return Settings(
        github_token="ghp_test_token",
        slack_bot_token="xoxb-test",
        slack_app_token="xapp-test",
        slack_pr_channel="C_TEST_PR",
        slack_ci_channel="C_TEST_CI",
        moonshot_api_key="sk-test",
        db_path=str(tmp_path / "test.db"),
    )

@pytest_asyncio.fixture
async def db(settings):
    """Real SQLite database (temp file), auto-creates schema."""
    database = Database(settings.db_path)
    await database.connect()
    yield database
    await database.close()
```

## Mock Patterns

### GitHub Client
```python
@pytest.fixture
def mock_gh():
    gh = AsyncMock()
    gh.get_pull = AsyncMock(return_value={
        "number": 19876, "state": "open",
        "head": {"sha": "abc123def456"},
        "labels": [{"name": "run-ci"}],
        "user": {"login": "alice"},
        "title": "Test PR",
        "updated_at": "2025-03-01T00:00:00Z",
    })
    gh.get_pull_reviews = AsyncMock(return_value=[])
    gh.get_pull_commits = AsyncMock(return_value=[{"sha": "abc123"}])
    gh.get_workflow_runs_for_ref = AsyncMock(return_value=[])
    gh.get_workflow_run_jobs = AsyncMock(return_value=[])
    gh.rerun_failed_jobs = AsyncMock()
    gh.create_issue_comment = AsyncMock()
    return gh
```

### CI Monitor with Real DB
```python
@pytest.fixture
def ci_monitor(mock_gh, db, settings):
    return CIMonitor(mock_gh, db, settings)
```

### Workflow Run Data
```python
# Nvidia workflow run (use settings.ci_nvidia_workflow_id):
{"id": 1, "workflow_id": settings.ci_nvidia_workflow_id,
 "status": "completed", "conclusion": "success"}

# Job data:
{"name": "build", "status": "completed", "conclusion": "success", "id": 10}
```

## Test Organization

Group tests in classes by feature area:

```python
class TestHighPriority:
    @pytest.mark.asyncio
    async def test_label_detected(self, ci_monitor, mock_gh, settings):
        ...

    @pytest.mark.asyncio
    async def test_retry_limit(self, ci_monitor, db, settings):
        ...
```

## Common Test Patterns

### Testing Callbacks
```python
callback = AsyncMock()
ci_monitor.set_callbacks(on_high_priority_nvidia_passed=callback)
await ci_monitor._poll_single_pr(19876, ["U123"])
callback.assert_called_once_with(19876, ["U123"], "approved")
```

### Testing Deduplication (Second Poll Shouldn't Trigger)
```python
await ci_monitor._poll_single_pr(19876, ["U123"])
assert callback.call_count == 1
await ci_monitor._poll_single_pr(19876, ["U123"])
assert callback.call_count == 1  # Still 1, not 2
```

### Testing Retry Exhaustion
```python
from sgldhelper.db import queries
for _ in range(settings.ci_max_retries):
    await queries.increment_ci_retry(db.conn, pr, sha, job_name)
assert await ci_monitor.should_retry(pr, sha, job) is False
```

### Testing AI Tools
```python
# Tool count and names must stay in sync:
def test_schemas_are_valid(self, registry):
    schemas = registry.get_schemas()
    assert len(schemas) == 16  # Update when adding tools

def test_tool_names(self, registry):
    names = {s["function"]["name"] for s in schemas}
    expected = {"get_open_prs", "trigger_ci", ...}  # Update set
    assert names == expected
```

## What to Test When Adding Features

| Change | Tests Needed |
|--------|-------------|
| New config field | Usually none (pydantic handles it) |
| New CI logic | Test the logic method + integration via `_poll_single_pr` |
| New AI tool | Update count + name set in `test_ai_tools.py`, add execution test |
| New callback | Test it fires with correct args, test dedup if applicable |
| New DB query | Test via the feature that uses it (no separate query tests) |
| New notification | Usually tested indirectly via callback tests |
