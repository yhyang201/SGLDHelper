"""Tests for CI rerun logic using PR comment commands."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from sgldhelper.db import queries
from sgldhelper.github.ci_analyzer import CIResult, FailureCategory
from sgldhelper.github.ci_rerunner import CIRerunner


@pytest.fixture
def rerunner(settings, db):
    client = AsyncMock()
    return CIRerunner(client, db, settings)


def _make_ci_result(*, pr_number=1234, run_id=100, category=FailureCategory.FLAKY):
    return CIResult(
        run_id=run_id,
        pr_number=pr_number,
        job_name="multimodal-gen-test-1-gpu",
        head_sha="abc123",
        status="completed",
        conclusion="failure",
        failure_category=category,
        failure_summary="SafetensorError: corrupted",
        html_url=None,
        auto_rerun_count=0,
    )


class TestAutoRerun:
    @pytest.mark.asyncio
    async def test_auto_rerun_comments_rerun_failed_ci(self, rerunner, db):
        """auto_rerun should comment /rerun-failed-ci on the PR."""
        # Seed a PR and CI run for the foreign key.
        await queries.upsert_pr(db.conn, {
            "pr_number": 1234, "title": "Test", "author": "u",
            "state": "open", "head_sha": "abc123",
            "updated_at": "2025-01-01T00:00:00Z", "changed_files": 1, "labels": [],
        })
        await queries.upsert_ci_run(db.conn, {
            "run_id": 100, "pr_number": 1234, "job_name": "test",
            "head_sha": "abc123", "status": "completed", "conclusion": "failure",
        })

        result = _make_ci_result()
        rr = await rerunner.auto_rerun(result)

        assert rr.triggered is True
        rerunner._client.create_issue_comment.assert_awaited_once_with(
            1234, "/rerun-failed-ci"
        )

    @pytest.mark.asyncio
    async def test_auto_rerun_skips_code_failure(self, rerunner):
        """CODE failures should not be auto-rerun."""
        result = _make_ci_result(category=FailureCategory.CODE)
        rr = await rerunner.auto_rerun(result)
        assert rr.triggered is False
        rerunner._client.create_issue_comment.assert_not_awaited()


class TestManualRerun:
    @pytest.mark.asyncio
    async def test_manual_rerun_comments_once(self, rerunner, db):
        """manual_rerun should post a single /rerun-failed-ci comment."""
        await queries.upsert_pr(db.conn, {
            "pr_number": 1234, "title": "Test", "author": "u",
            "state": "open", "head_sha": "abc123",
            "updated_at": "2025-01-01T00:00:00Z", "changed_files": 1, "labels": [],
        })
        # Two failed runs for the same PR.
        for run_id in (100, 101):
            await queries.upsert_ci_run(db.conn, {
                "run_id": run_id, "pr_number": 1234, "job_name": "test",
                "head_sha": "abc123", "status": "completed", "conclusion": "failure",
            })

        results = await rerunner.manual_rerun(1234)
        assert len(results) == 2
        assert all(r.triggered for r in results)
        # Only one comment despite two failed runs.
        rerunner._client.create_issue_comment.assert_awaited_once_with(
            1234, "/rerun-failed-ci"
        )

    @pytest.mark.asyncio
    async def test_manual_rerun_no_failures(self, rerunner, db):
        """manual_rerun with no failed runs should not comment."""
        await queries.upsert_pr(db.conn, {
            "pr_number": 1234, "title": "Test", "author": "u",
            "state": "open", "head_sha": "abc123",
            "updated_at": "2025-01-01T00:00:00Z", "changed_files": 1, "labels": [],
        })
        await queries.upsert_ci_run(db.conn, {
            "run_id": 100, "pr_number": 1234, "job_name": "test",
            "head_sha": "abc123", "status": "completed", "conclusion": "success",
        })

        results = await rerunner.manual_rerun(1234)
        assert results == []
        rerunner._client.create_issue_comment.assert_not_awaited()
