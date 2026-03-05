"""SQLite schema definitions."""

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS pull_requests (
    pr_number       INTEGER PRIMARY KEY,
    title           TEXT NOT NULL,
    author          TEXT NOT NULL,
    state           TEXT NOT NULL DEFAULT 'open',
    head_sha        TEXT NOT NULL,
    updated_at      TEXT NOT NULL,
    changed_files   INTEGER NOT NULL DEFAULT 0,
    labels          TEXT NOT NULL DEFAULT '[]',
    slack_thread_ts TEXT,
    last_notified_state TEXT,
    created_at      TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS poll_state (
    key     TEXT PRIMARY KEY,
    value   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS conversations (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    thread_ts           TEXT NOT NULL,
    channel_id          TEXT NOT NULL,
    role                TEXT NOT NULL,
    content             TEXT,
    reasoning_content   TEXT,
    tool_calls          TEXT,
    tool_call_id        TEXT,
    name                TEXT,
    tokens_in           INTEGER NOT NULL DEFAULT 0,
    tokens_out          INTEGER NOT NULL DEFAULT 0,
    created_at          TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_conversations_thread ON conversations(thread_ts);

CREATE TABLE IF NOT EXISTS detected_updates (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    channel_id      TEXT NOT NULL,
    message_ts      TEXT NOT NULL,
    user_id         TEXT NOT NULL,
    classification  TEXT NOT NULL,
    extracted_data  TEXT NOT NULL DEFAULT '{}',
    matched_item_id TEXT,
    confirmed       INTEGER NOT NULL DEFAULT 0,
    created_at      TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS llm_usage (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    task_type   TEXT NOT NULL,
    model       TEXT NOT NULL,
    tokens_in   INTEGER NOT NULL DEFAULT 0,
    tokens_out  INTEGER NOT NULL DEFAULT 0,
    created_at  TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS user_memories (
    user_id         TEXT PRIMARY KEY,
    tracked_prs     TEXT NOT NULL DEFAULT '[]',
    focus_areas     TEXT NOT NULL DEFAULT '[]',
    preferences     TEXT NOT NULL DEFAULT '{}',
    notes           TEXT NOT NULL DEFAULT '',
    updated_at      TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS pr_classification (
    pr_number    INTEGER PRIMARY KEY,
    head_sha     TEXT NOT NULL,
    is_diffusion INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS ci_retry_state (
    pr_number   INTEGER NOT NULL,
    head_sha    TEXT NOT NULL,
    job_name    TEXT NOT NULL,
    retry_count INTEGER NOT NULL DEFAULT 0,
    updated_at  TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (pr_number, head_sha, job_name)
);

CREATE TABLE IF NOT EXISTS ci_snapshots (
    pr_number       INTEGER NOT NULL,
    head_sha        TEXT NOT NULL,
    overall_status  TEXT NOT NULL DEFAULT 'unknown',
    has_run_ci_label INTEGER NOT NULL DEFAULT 0,
    failed_jobs     TEXT NOT NULL DEFAULT '[]',
    review_state    TEXT NOT NULL DEFAULT 'none',
    commit_count    INTEGER NOT NULL DEFAULT 0,
    snapshot_data   TEXT NOT NULL DEFAULT '{}',
    updated_at      TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (pr_number, head_sha)
);

CREATE TABLE IF NOT EXISTS tracked_pr_summaries (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    pr_number   INTEGER NOT NULL,
    summary     TEXT NOT NULL,
    user_ids    TEXT NOT NULL DEFAULT '[]',
    created_at  TEXT NOT NULL DEFAULT (datetime('now'))
);
"""
