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


# ---------------------------------------------------------------------------
# CI runs
# ---------------------------------------------------------------------------

async def upsert_ci_run(conn: aiosqlite.Connection, run: dict[str, Any]) -> dict[str, Any] | None:
    cur = await conn.execute(
        "SELECT * FROM ci_runs WHERE run_id = ?", (run["run_id"],)
    )
    old = await cur.fetchone()
    old_dict = dict(old) if old else None

    await conn.execute(
        """INSERT INTO ci_runs
            (run_id, pr_number, job_name, head_sha, status, conclusion,
             failure_category, failure_summary, auto_rerun_count, html_url)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(run_id) DO UPDATE SET
            status = excluded.status,
            conclusion = excluded.conclusion,
            failure_category = excluded.failure_category,
            failure_summary = excluded.failure_summary,
            auto_rerun_count = excluded.auto_rerun_count,
            html_url = excluded.html_url
        """,
        (
            run["run_id"],
            run["pr_number"],
            run["job_name"],
            run["head_sha"],
            run["status"],
            run.get("conclusion"),
            run.get("failure_category"),
            run.get("failure_summary"),
            run.get("auto_rerun_count", 0),
            run.get("html_url"),
        ),
    )
    await conn.commit()
    return old_dict


async def get_ci_run(conn: aiosqlite.Connection, run_id: int) -> dict[str, Any] | None:
    cur = await conn.execute("SELECT * FROM ci_runs WHERE run_id = ?", (run_id,))
    row = await cur.fetchone()
    return dict(row) if row else None


async def get_ci_runs_for_pr(
    conn: aiosqlite.Connection, pr_number: int
) -> list[dict[str, Any]]:
    cur = await conn.execute(
        "SELECT * FROM ci_runs WHERE pr_number = ? ORDER BY created_at DESC",
        (pr_number,),
    )
    return [dict(r) for r in await cur.fetchall()]


async def increment_rerun_count(conn: aiosqlite.Connection, run_id: int) -> None:
    await conn.execute(
        "UPDATE ci_runs SET auto_rerun_count = auto_rerun_count + 1 WHERE run_id = ?",
        (run_id,),
    )
    await conn.commit()


async def log_rerun(
    conn: aiosqlite.Connection,
    run_id: int,
    new_run_id: int | None,
    triggered_by: str,
    reason: str,
) -> None:
    await conn.execute(
        "INSERT INTO ci_rerun_log (run_id, new_run_id, triggered_by, reason) VALUES (?, ?, ?, ?)",
        (run_id, new_run_id, triggered_by, reason),
    )
    await conn.commit()


# ---------------------------------------------------------------------------
# Feature items
# ---------------------------------------------------------------------------

async def upsert_feature_item(conn: aiosqlite.Connection, item: dict[str, Any]) -> dict[str, Any] | None:
    cur = await conn.execute(
        "SELECT * FROM feature_items WHERE item_id = ?", (item["item_id"],)
    )
    old = await cur.fetchone()
    old_dict = dict(old) if old else None

    await conn.execute(
        """INSERT INTO feature_items
            (item_id, parent_issue, title, item_type, state, linked_pr, completed_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(item_id) DO UPDATE SET
            title = excluded.title,
            state = excluded.state,
            linked_pr = excluded.linked_pr,
            completed_at = excluded.completed_at
        """,
        (
            item["item_id"],
            item["parent_issue"],
            item["title"],
            item.get("item_type", "checkbox"),
            item["state"],
            item.get("linked_pr"),
            item.get("completed_at"),
        ),
    )
    await conn.commit()
    return old_dict


async def get_feature_items(
    conn: aiosqlite.Connection, parent_issue: int
) -> list[dict[str, Any]]:
    cur = await conn.execute(
        "SELECT * FROM feature_items WHERE parent_issue = ? ORDER BY item_id",
        (parent_issue,),
    )
    return [dict(r) for r in await cur.fetchall()]


async def get_feature_summary(
    conn: aiosqlite.Connection, parent_issue: int
) -> dict[str, int]:
    cur = await conn.execute(
        "SELECT state, COUNT(*) as cnt FROM feature_items WHERE parent_issue = ? GROUP BY state",
        (parent_issue,),
    )
    rows = await cur.fetchall()
    result: dict[str, int] = {}
    for r in rows:
        result[r["state"]] = r["cnt"]
    return result


async def update_feature_linked_pr(
    conn: aiosqlite.Connection, item_id: str, pr_number: int
) -> bool:
    """Link a PR to a feature item. Returns True if the row was updated."""
    cur = await conn.execute(
        "UPDATE feature_items SET linked_pr = ? WHERE item_id = ?",
        (pr_number, item_id),
    )
    await conn.commit()
    return cur.rowcount > 0


async def get_unlinked_features(conn: aiosqlite.Connection) -> list[dict[str, Any]]:
    """Return open feature items that have no linked PR."""
    cur = await conn.execute(
        "SELECT * FROM feature_items WHERE linked_pr IS NULL AND state = 'open' ORDER BY parent_issue, item_id"
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
# Stall alerts (dedup)
# ---------------------------------------------------------------------------

async def record_stall_alert(
    conn: aiosqlite.Connection,
    alert_type: str,
    ref_id: str,
    days_stalled: int,
) -> bool:
    """Record a stall alert. Returns True if new (not a duplicate)."""
    try:
        await conn.execute(
            "INSERT INTO stall_alerts (alert_type, ref_id, days_stalled) VALUES (?, ?, ?)",
            (alert_type, ref_id, days_stalled),
        )
        await conn.commit()
        return True
    except Exception:
        # UNIQUE constraint violation → duplicate
        return False


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
    """Aggregate recent PR/CI/feature activity for standup summaries."""
    prs_cur = await conn.execute(
        """SELECT * FROM pull_requests
        WHERE updated_at >= datetime('now', ?)
        ORDER BY updated_at DESC""",
        (f"-{since_hours} hours",),
    )
    prs = [dict(r) for r in await prs_cur.fetchall()]

    ci_cur = await conn.execute(
        """SELECT * FROM ci_runs
        WHERE created_at >= datetime('now', ?)
        ORDER BY created_at DESC""",
        (f"-{since_hours} hours",),
    )
    ci_runs = [dict(r) for r in await ci_cur.fetchall()]

    feature_cur = await conn.execute(
        "SELECT * FROM feature_items WHERE state = 'open' ORDER BY parent_issue, item_id"
    )
    open_features = [dict(r) for r in await feature_cur.fetchall()]

    return {"prs": prs, "ci_runs": ci_runs, "open_features": open_features}


async def get_stalled_features(
    conn: aiosqlite.Connection, stall_days: int = 3
) -> list[dict[str, Any]]:
    """Find open feature items with linked PRs that haven't been updated recently."""
    cur = await conn.execute(
        """SELECT fi.*, pr.updated_at as pr_updated_at, pr.title as pr_title, pr.author as pr_author
        FROM feature_items fi
        JOIN pull_requests pr ON fi.linked_pr = pr.pr_number
        WHERE fi.state = 'open'
          AND pr.state = 'open'
          AND pr.updated_at < datetime('now', ?)
        ORDER BY pr.updated_at ASC""",
        (f"-{stall_days} days",),
    )
    return [dict(r) for r in await cur.fetchall()]


async def get_prs_needing_review(
    conn: aiosqlite.Connection, review_days: int = 2
) -> list[dict[str, Any]]:
    """Find open PRs that haven't been updated in N days (likely need review)."""
    cur = await conn.execute(
        """SELECT * FROM pull_requests
        WHERE state = 'open'
          AND updated_at < datetime('now', ?)
        ORDER BY updated_at ASC""",
        (f"-{review_days} days",),
    )
    return [dict(r) for r in await cur.fetchall()]


async def get_all_feature_items(
    conn: aiosqlite.Connection,
) -> list[dict[str, Any]]:
    cur = await conn.execute(
        "SELECT * FROM feature_items ORDER BY parent_issue, item_id"
    )
    return [dict(r) for r in await cur.fetchall()]


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
