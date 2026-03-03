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

CREATE TABLE IF NOT EXISTS ci_runs (
    run_id              INTEGER PRIMARY KEY,
    pr_number           INTEGER NOT NULL,
    job_name            TEXT NOT NULL,
    head_sha            TEXT NOT NULL,
    status              TEXT NOT NULL,
    conclusion          TEXT,
    failure_category    TEXT,
    failure_summary     TEXT,
    auto_rerun_count    INTEGER NOT NULL DEFAULT 0,
    html_url            TEXT,
    created_at          TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (pr_number) REFERENCES pull_requests(pr_number)
);

CREATE TABLE IF NOT EXISTS ci_rerun_log (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id          INTEGER NOT NULL,
    new_run_id      INTEGER,
    triggered_by    TEXT NOT NULL DEFAULT 'auto',
    reason          TEXT,
    created_at      TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (run_id) REFERENCES ci_runs(run_id)
);

CREATE TABLE IF NOT EXISTS feature_items (
    item_id         TEXT PRIMARY KEY,
    parent_issue    INTEGER NOT NULL,
    title           TEXT NOT NULL,
    item_type       TEXT NOT NULL DEFAULT 'checkbox',
    state           TEXT NOT NULL DEFAULT 'open',
    linked_pr       INTEGER,
    completed_at    TEXT,
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

CREATE TABLE IF NOT EXISTS stall_alerts (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    alert_type      TEXT NOT NULL,
    ref_id          TEXT NOT NULL,
    days_stalled    INTEGER NOT NULL,
    alert_sent_at   TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(alert_type, ref_id, days_stalled)
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
"""
