"""Database query functions."""

from __future__ import annotations

import json
from typing import Any

import aiosqlite


# ---------------------------------------------------------------------------
# Poll state
# ---------------------------------------------------------------------------

async def get_poll_state(conn: aiosqlite.Connection, key: str) -> str | None:
    cur = await conn.execute("SELECT value FROM poll_state WHERE key = ?", (key,))
    row = await cur.fetchone()
    return row["value"] if row else None


async def set_poll_state(conn: aiosqlite.Connection, key: str, value: str) -> None:
    await conn.execute(
        "INSERT INTO poll_state (key, value) VALUES (?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (key, value),
    )
    await conn.commit()


# ---------------------------------------------------------------------------
# Pull requests
# ---------------------------------------------------------------------------

async def upsert_pr(conn: aiosqlite.Connection, pr: dict[str, Any]) -> dict[str, Any] | None:
    """Insert or update a PR. Returns the previous row if it existed."""
    cur = await conn.execute(
        "SELECT * FROM pull_requests WHERE pr_number = ?", (pr["pr_number"],)
    )
    old = await cur.fetchone()
    old_dict = dict(old) if old else None

    await conn.execute(
        """INSERT INTO pull_requests
            (pr_number, title, author, state, head_sha, updated_at, changed_files, labels)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(pr_number) DO UPDATE SET
            title = excluded.title,
            author = excluded.author,
            state = excluded.state,
            head_sha = excluded.head_sha,
            updated_at = excluded.updated_at,
            changed_files = excluded.changed_files,
            labels = excluded.labels
        """,
        (
            pr["pr_number"],
            pr["title"],
            pr["author"],
            pr["state"],
            pr["head_sha"],
            pr["updated_at"],
            pr["changed_files"],
            json.dumps(pr.get("labels", [])),
        ),
    )
    await conn.commit()
    return old_dict


async def get_pr(conn: aiosqlite.Connection, pr_number: int) -> dict[str, Any] | None:
    cur = await conn.execute(
        "SELECT * FROM pull_requests WHERE pr_number = ?", (pr_number,)
    )
    row = await cur.fetchone()
    return dict(row) if row else None


async def set_pr_slack_thread(
    conn: aiosqlite.Connection, pr_number: int, thread_ts: str
) -> None:
    await conn.execute(
        "UPDATE pull_requests SET slack_thread_ts = ? WHERE pr_number = ?",
        (thread_ts, pr_number),
    )
    await conn.commit()


async def set_pr_notified_state(
    conn: aiosqlite.Connection, pr_number: int, state: str
) -> None:
    await conn.execute(
        "UPDATE pull_requests SET last_notified_state = ? WHERE pr_number = ?",
        (state, pr_number),
    )
    await conn.commit()


async def get_open_prs(conn: aiosqlite.Connection) -> list[dict[str, Any]]:
    cur = await conn.execute(
        "SELECT * FROM pull_requests WHERE state = 'open' ORDER BY pr_number DESC"
    )
    return [dict(r) for r in await cur.fetchall()]


async def get_users_tracking_pr(conn: aiosqlite.Connection, pr_number: int) -> list[str]:
    """Return user IDs whose tracked_prs list contains *pr_number*."""
    cur = await conn.execute("SELECT user_id, tracked_prs FROM user_memories")
    rows = await cur.fetchall()
    result: list[str] = []
    for row in rows:
        tracked = json.loads(row["tracked_prs"])
        if pr_number in tracked:
            result.append(row["user_id"])
    return result


# ---------------------------------------------------------------------------
# Conversations (per-thread AI chat history)
# ---------------------------------------------------------------------------

async def save_conversation_message(
    conn: aiosqlite.Connection,
    thread_ts: str,
    channel_id: str,
    role: str,
    content: str | None = None,
    reasoning_content: str | None = None,
    tool_calls: str | None = None,
    tool_call_id: str | None = None,
    name: str | None = None,
    tokens_in: int = 0,
    tokens_out: int = 0,
) -> None:
    await conn.execute(
        """INSERT INTO conversations
            (thread_ts, channel_id, role, content, reasoning_content,
             tool_calls, tool_call_id, name, tokens_in, tokens_out)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (thread_ts, channel_id, role, content, reasoning_content,
         tool_calls, tool_call_id, name, tokens_in, tokens_out),
    )
    await conn.commit()


async def get_conversation_history(
    conn: aiosqlite.Connection, thread_ts: str, limit: int = 50
) -> list[dict[str, Any]]:
    cur = await conn.execute(
        "SELECT * FROM conversations WHERE thread_ts = ? ORDER BY id ASC LIMIT ?",
        (thread_ts, limit),
    )
    return [dict(r) for r in await cur.fetchall()]


async def clear_conversation(conn: aiosqlite.Connection, thread_ts: str) -> None:
    await conn.execute("DELETE FROM conversations WHERE thread_ts = ?", (thread_ts,))
    await conn.commit()


# ---------------------------------------------------------------------------
# Detected updates (passive message classification)
# ---------------------------------------------------------------------------

async def save_detected_update(
    conn: aiosqlite.Connection,
    channel_id: str,
    message_ts: str,
    user_id: str,
    classification: str,
    extracted_data: str = "{}",
    matched_item_id: str | None = None,
) -> int:
    cur = await conn.execute(
        """INSERT INTO detected_updates
            (channel_id, message_ts, user_id, classification, extracted_data, matched_item_id)
        VALUES (?, ?, ?, ?, ?, ?)""",
        (channel_id, message_ts, user_id, classification, extracted_data, matched_item_id),
    )
    await conn.commit()
    return cur.lastrowid  # type: ignore[return-value]


async def confirm_detected_update(conn: aiosqlite.Connection, update_id: int) -> None:
    await conn.execute(
        "UPDATE detected_updates SET confirmed = 1 WHERE id = ?", (update_id,)
    )
    await conn.commit()


async def get_pending_updates(
    conn: aiosqlite.Connection, classification: str | None = None
) -> list[dict[str, Any]]:
    if classification:
        cur = await conn.execute(
            "SELECT * FROM detected_updates WHERE confirmed = 0 AND classification = ? ORDER BY id DESC",
            (classification,),
        )
    else:
        cur = await conn.execute(
            "SELECT * FROM detected_updates WHERE confirmed = 0 ORDER BY id DESC"
        )
    return [dict(r) for r in await cur.fetchall()]


async def get_confirmed_blockers(
    conn: aiosqlite.Connection, since_hours: int = 24
) -> list[dict[str, Any]]:
    cur = await conn.execute(
        """SELECT * FROM detected_updates
        WHERE confirmed = 1 AND classification = 'blocker'
          AND created_at >= datetime('now', ?)
        ORDER BY created_at DESC""",
        (f"-{since_hours} hours",),
    )
    return [dict(r) for r in await cur.fetchall()]


# ---------------------------------------------------------------------------
# LLM usage tracking
# ---------------------------------------------------------------------------

async def log_llm_usage(
    conn: aiosqlite.Connection,
    task_type: str,
    model: str,
    tokens_in: int,
    tokens_out: int,
) -> None:
    await conn.execute(
        "INSERT INTO llm_usage (task_type, model, tokens_in, tokens_out) VALUES (?, ?, ?, ?)",
        (task_type, model, tokens_in, tokens_out),
    )
    await conn.commit()


async def get_llm_usage_summary(
    conn: aiosqlite.Connection, since_hours: int = 24
) -> list[dict[str, Any]]:
    cur = await conn.execute(
        """SELECT task_type, model,
                  SUM(tokens_in) as total_in, SUM(tokens_out) as total_out,
                  COUNT(*) as calls
        FROM llm_usage
        WHERE created_at >= datetime('now', ?)
        GROUP BY task_type, model""",
        (f"-{since_hours} hours",),
    )
    return [dict(r) for r in await cur.fetchall()]


# ---------------------------------------------------------------------------
# New search / activity queries for AI tools
# ---------------------------------------------------------------------------

async def search_prs(
    conn: aiosqlite.Connection, query: str, limit: int = 10
) -> list[dict[str, Any]]:
    cur = await conn.execute(
        """SELECT * FROM pull_requests
        WHERE title LIKE ? OR author LIKE ?
        ORDER BY pr_number DESC LIMIT ?""",
        (f"%{query}%", f"%{query}%", limit),
    )
    return [dict(r) for r in await cur.fetchall()]


async def get_recent_activity(
    conn: aiosqlite.Connection, since_hours: int = 24
) -> dict[str, Any]:
    """Aggregate recent PR activity for standup summaries."""
    prs_cur = await conn.execute(
        """SELECT * FROM pull_requests
        WHERE updated_at >= datetime('now', ?)
        ORDER BY updated_at DESC""",
        (f"-{since_hours} hours",),
    )
    prs = [dict(r) for r in await prs_cur.fetchall()]

    return {"prs": prs}


# ---------------------------------------------------------------------------
# User memories (per-user preference tracking)
# ---------------------------------------------------------------------------

async def get_user_memory(conn: aiosqlite.Connection, user_id: str) -> dict[str, Any] | None:
    """Get the memory record for a user, or None if not found."""
    cur = await conn.execute(
        "SELECT * FROM user_memories WHERE user_id = ?", (user_id,)
    )
    row = await cur.fetchone()
    return dict(row) if row else None


async def upsert_user_memory(
    conn: aiosqlite.Connection,
    user_id: str,
    *,
    tracked_prs: list[int] | None = None,
    focus_areas: list[str] | None = None,
    preferences: dict[str, Any] | None = None,
    notes: str | None = None,
) -> None:
    """Create or update a user's memory record. Only non-None fields are updated."""
    existing = await get_user_memory(conn, user_id)
    if existing is None:
        await conn.execute(
            """INSERT INTO user_memories (user_id, tracked_prs, focus_areas, preferences, notes)
            VALUES (?, ?, ?, ?, ?)""",
            (
                user_id,
                json.dumps(tracked_prs or []),
                json.dumps(focus_areas or []),
                json.dumps(preferences or {}),
                notes or "",
            ),
        )
    else:
        if tracked_prs is not None:
            await conn.execute(
                "UPDATE user_memories SET tracked_prs = ?, updated_at = datetime('now') WHERE user_id = ?",
                (json.dumps(tracked_prs), user_id),
            )
        if focus_areas is not None:
            await conn.execute(
                "UPDATE user_memories SET focus_areas = ?, updated_at = datetime('now') WHERE user_id = ?",
                (json.dumps(focus_areas), user_id),
            )
        if preferences is not None:
            await conn.execute(
                "UPDATE user_memories SET preferences = ?, updated_at = datetime('now') WHERE user_id = ?",
                (json.dumps(preferences), user_id),
            )
        if notes is not None:
            await conn.execute(
                "UPDATE user_memories SET notes = ?, updated_at = datetime('now') WHERE user_id = ?",
                (notes, user_id),
            )
    await conn.commit()


async def add_tracked_pr(conn: aiosqlite.Connection, user_id: str, pr_number: int) -> list[int]:
    """Add a PR to user's tracked list. Returns the updated list."""
    mem = await get_user_memory(conn, user_id)
    if mem:
        current = json.loads(mem["tracked_prs"])
    else:
        current = []
    if pr_number not in current:
        current.append(pr_number)
    await upsert_user_memory(conn, user_id, tracked_prs=current)
    return current


async def remove_tracked_pr(conn: aiosqlite.Connection, user_id: str, pr_number: int) -> list[int]:
    """Remove a PR from user's tracked list. Returns the updated list."""
    mem = await get_user_memory(conn, user_id)
    if mem:
        current = json.loads(mem["tracked_prs"])
    else:
        current = []
    current = [p for p in current if p != pr_number]
    await upsert_user_memory(conn, user_id, tracked_prs=current)
    return current


async def save_user_note(conn: aiosqlite.Connection, user_id: str, note: str) -> None:
    """Append or replace the AI-written note for a user."""
    await upsert_user_memory(conn, user_id, notes=note)


# ---------------------------------------------------------------------------
# PR classification cache (diffusion detection)
# ---------------------------------------------------------------------------

async def get_diffusion_pr_summary(
    conn: aiosqlite.Connection, since_hours: int = 2
) -> dict[str, list[dict[str, Any]]]:
    """Return diffusion PRs grouped by recent activity: new open, merged, closed."""
    interval = f"-{since_hours} hours"

    # Diffusion PRs are those stored in pull_requests (only diffusion PRs are upserted)
    cur_open = await conn.execute(
        """SELECT * FROM pull_requests
        WHERE state = 'open' AND created_at >= datetime('now', ?)
        ORDER BY pr_number DESC""",
        (interval,),
    )
    newly_opened = [dict(r) for r in await cur_open.fetchall()]

    cur_merged = await conn.execute(
        """SELECT * FROM pull_requests
        WHERE state = 'merged' AND updated_at >= datetime('now', ?)
        ORDER BY pr_number DESC""",
        (interval,),
    )
    merged = [dict(r) for r in await cur_merged.fetchall()]

    cur_closed = await conn.execute(
        """SELECT * FROM pull_requests
        WHERE state = 'closed' AND updated_at >= datetime('now', ?)
        ORDER BY pr_number DESC""",
        (interval,),
    )
    closed = [dict(r) for r in await cur_closed.fetchall()]

    return {"opened": newly_opened, "merged": merged, "closed": closed}


async def get_pr_classifications(conn: aiosqlite.Connection) -> dict[int, tuple[str, bool]]:
    """Return {pr_number: (head_sha, is_diffusion)} for all cached entries."""
    cur = await conn.execute("SELECT pr_number, head_sha, is_diffusion FROM pr_classification")
    rows = await cur.fetchall()
    return {row["pr_number"]: (row["head_sha"], bool(row["is_diffusion"])) for row in rows}


async def upsert_pr_classification(
    conn: aiosqlite.Connection, pr_number: int, head_sha: str, is_diffusion: bool
) -> None:
    """Insert or update the classification cache for a PR."""
    await conn.execute(
        "INSERT INTO pr_classification (pr_number, head_sha, is_diffusion) VALUES (?, ?, ?) "
        "ON CONFLICT(pr_number) DO UPDATE SET head_sha = excluded.head_sha, is_diffusion = excluded.is_diffusion",
        (pr_number, head_sha, int(is_diffusion)),
    )
    await conn.commit()


# ---------------------------------------------------------------------------
# CI retry state
# ---------------------------------------------------------------------------

async def get_ci_retry_count(
    conn: aiosqlite.Connection, pr_number: int, head_sha: str, job_name: str
) -> int:
    cur = await conn.execute(
        "SELECT retry_count FROM ci_retry_state WHERE pr_number = ? AND head_sha = ? AND job_name = ?",
        (pr_number, head_sha, job_name),
    )
    row = await cur.fetchone()
    return row["retry_count"] if row else 0


async def increment_ci_retry(
    conn: aiosqlite.Connection, pr_number: int, head_sha: str, job_name: str
) -> int:
    """Increment retry count and return the new value."""
    await conn.execute(
        """INSERT INTO ci_retry_state (pr_number, head_sha, job_name, retry_count)
        VALUES (?, ?, ?, 1)
        ON CONFLICT(pr_number, head_sha, job_name) DO UPDATE SET
            retry_count = ci_retry_state.retry_count + 1,
            updated_at = datetime('now')""",
        (pr_number, head_sha, job_name),
    )
    await conn.commit()
    return await get_ci_retry_count(conn, pr_number, head_sha, job_name)


async def reset_ci_retries(
    conn: aiosqlite.Connection, pr_number: int, head_sha: str
) -> None:
    await conn.execute(
        "DELETE FROM ci_retry_state WHERE pr_number = ? AND head_sha = ?",
        (pr_number, head_sha),
    )
    await conn.commit()


# ---------------------------------------------------------------------------
# CI snapshots
# ---------------------------------------------------------------------------

async def upsert_ci_snapshot(
    conn: aiosqlite.Connection,
    pr_number: int,
    head_sha: str,
    *,
    overall_status: str = "unknown",
    has_run_ci_label: bool = False,
    failed_jobs: str = "[]",
    review_state: str = "none",
    commit_count: int = 0,
    snapshot_data: str = "{}",
) -> None:
    await conn.execute(
        """INSERT INTO ci_snapshots
            (pr_number, head_sha, overall_status, has_run_ci_label,
             failed_jobs, review_state, commit_count, snapshot_data)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(pr_number, head_sha) DO UPDATE SET
            overall_status = excluded.overall_status,
            has_run_ci_label = excluded.has_run_ci_label,
            failed_jobs = excluded.failed_jobs,
            review_state = excluded.review_state,
            commit_count = excluded.commit_count,
            snapshot_data = excluded.snapshot_data,
            updated_at = datetime('now')""",
        (pr_number, head_sha, overall_status, int(has_run_ci_label),
         failed_jobs, review_state, commit_count, snapshot_data),
    )
    await conn.commit()


async def get_ci_snapshot(
    conn: aiosqlite.Connection, pr_number: int, head_sha: str
) -> dict[str, Any] | None:
    cur = await conn.execute(
        "SELECT * FROM ci_snapshots WHERE pr_number = ? AND head_sha = ?",
        (pr_number, head_sha),
    )
    row = await cur.fetchone()
    return dict(row) if row else None


async def get_latest_ci_snapshot(
    conn: aiosqlite.Connection, pr_number: int
) -> dict[str, Any] | None:
    """Get the most recent snapshot for a PR regardless of SHA."""
    cur = await conn.execute(
        "SELECT * FROM ci_snapshots WHERE pr_number = ? ORDER BY updated_at DESC LIMIT 1",
        (pr_number,),
    )
    row = await cur.fetchone()
    return dict(row) if row else None


# ---------------------------------------------------------------------------
# Tracked PR summaries
# ---------------------------------------------------------------------------

async def save_tracked_pr_summary(
    conn: aiosqlite.Connection, pr_number: int, summary: str, user_ids: list[str]
) -> None:
    await conn.execute(
        "INSERT INTO tracked_pr_summaries (pr_number, summary, user_ids) VALUES (?, ?, ?)",
        (pr_number, summary, json.dumps(user_ids)),
    )
    await conn.commit()


async def get_last_tracked_pr_summary(
    conn: aiosqlite.Connection, pr_number: int
) -> dict[str, Any] | None:
    cur = await conn.execute(
        "SELECT * FROM tracked_pr_summaries WHERE pr_number = ? ORDER BY id DESC LIMIT 1",
        (pr_number,),
    )
    row = await cur.fetchone()
    return dict(row) if row else None


# ---------------------------------------------------------------------------
# Tracked PR helpers (cross-user)
# ---------------------------------------------------------------------------

async def get_all_tracked_prs(conn: aiosqlite.Connection) -> dict[int, list[str]]:
    """Return {pr_number: [user_ids]} for all tracked PRs across all users."""
    cur = await conn.execute("SELECT user_id, tracked_prs FROM user_memories")
    rows = await cur.fetchall()
    result: dict[int, list[str]] = {}
    for row in rows:
        tracked = json.loads(row["tracked_prs"])
        for pr_num in tracked:
            result.setdefault(pr_num, []).append(row["user_id"])
    return result


async def remove_tracked_pr_all_users(
    conn: aiosqlite.Connection, pr_number: int
) -> list[str]:
    """Remove a PR from all users' tracked lists. Returns affected user IDs."""
    cur = await conn.execute("SELECT user_id, tracked_prs FROM user_memories")
    rows = await cur.fetchall()
    affected: list[str] = []
    for row in rows:
        tracked = json.loads(row["tracked_prs"])
        if pr_number in tracked:
            tracked = [p for p in tracked if p != pr_number]
            await conn.execute(
                "UPDATE user_memories SET tracked_prs = ?, updated_at = datetime('now') WHERE user_id = ?",
                (json.dumps(tracked), row["user_id"]),
            )
            affected.append(row["user_id"])
    if affected:
        await conn.commit()
    return affected
