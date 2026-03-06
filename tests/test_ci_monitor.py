"""Tests for CI monitor: job parsing, status aggregation, retry logic."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock

import pytest

from sgldhelper.ci.monitor import CIMonitor, CIJobResult, CIStatus, CIOverallStatus


@pytest.fixture
def mock_gh():
    gh = AsyncMock()
    gh.get_pull = AsyncMock(return_value={
        "number": 19876,
        "state": "open",
        "head": {"sha": "abc123def456"},
        "labels": [{"name": "run-ci"}],
        "user": {"login": "alice"},
        "title": "Test PR",
        "updated_at": "2025-03-01T00:00:00Z",
    })
    gh.get_pull_reviews = AsyncMock(return_value=[])
    gh.get_pull_commits = AsyncMock(return_value=[{"sha": "abc123"}])
    gh.rerun_failed_jobs = AsyncMock()
    gh.create_issue_comment = AsyncMock()
    return gh


@pytest.fixture
def ci_monitor(mock_gh, db, settings):
    return CIMonitor(mock_gh, db, settings)


class TestCheckPrCI:
    @pytest.mark.asyncio
    async def test_no_ci_runs(self, ci_monitor, mock_gh):
        """When no workflow runs exist, return NO_CI."""
        mock_gh.get_workflow_runs_for_ref = AsyncMock(return_value=[])
        status = await ci_monitor.check_pr_ci(19876, "abc123def456")
        assert status.overall == CIOverallStatus.NO_CI
        assert status.all_runs_completed is True

    @pytest.mark.asyncio
    async def test_all_passed(self, ci_monitor, mock_gh, settings):
        """When all jobs succeed, return PASSED."""
        mock_gh.get_workflow_runs_for_ref = AsyncMock(return_value=[
            {
                "id": 1,
                "workflow_id": settings.ci_nvidia_workflow_id,
                "status": "completed",
                "conclusion": "success",
            },
        ])
        mock_gh.get_workflow_run_jobs = AsyncMock(return_value=[
            {"name": "build", "status": "completed", "conclusion": "success", "id": 10},
        ])
        status = await ci_monitor.check_pr_ci(19876, "abc123def456")
        assert status.overall == CIOverallStatus.PASSED
        assert len(status.nvidia_jobs) == 1
        assert status.all_runs_completed is True

    @pytest.mark.asyncio
    async def test_failed_jobs(self, ci_monitor, mock_gh, settings):
        """When a job fails, return FAILED with failed_jobs populated."""
        mock_gh.get_workflow_runs_for_ref = AsyncMock(return_value=[
            {
                "id": 1,
                "workflow_id": settings.ci_nvidia_workflow_id,
                "status": "completed",
                "conclusion": "failure",
            },
        ])
        mock_gh.get_workflow_run_jobs = AsyncMock(return_value=[
            {"name": "test-gpu", "status": "completed", "conclusion": "failure", "id": 10},
            {"name": "lint", "status": "completed", "conclusion": "success", "id": 11},
        ])
        status = await ci_monitor.check_pr_ci(19876, "abc123def456")
        assert status.overall == CIOverallStatus.FAILED
        assert len(status.failed_jobs) == 1
        assert status.failed_jobs[0].job_name == "test-gpu"

    @pytest.mark.asyncio
    async def test_running_status(self, ci_monitor, mock_gh, settings):
        """When runs are still in progress, return RUNNING."""
        mock_gh.get_workflow_runs_for_ref = AsyncMock(return_value=[
            {
                "id": 1,
                "workflow_id": settings.ci_nvidia_workflow_id,
                "status": "in_progress",
                "conclusion": None,
            },
        ])
        mock_gh.get_workflow_run_jobs = AsyncMock(return_value=[
            {"name": "build", "status": "in_progress", "conclusion": None, "id": 10},
        ])
        status = await ci_monitor.check_pr_ci(19876, "abc123def456")
        assert status.overall == CIOverallStatus.RUNNING
        assert status.all_runs_completed is False

    @pytest.mark.asyncio
    async def test_skipped_jobs_filtered(self, ci_monitor, mock_gh, settings):
        """Skipped jobs should be excluded from results."""
        mock_gh.get_workflow_runs_for_ref = AsyncMock(return_value=[
            {
                "id": 1,
                "workflow_id": settings.ci_nvidia_workflow_id,
                "status": "completed",
                "conclusion": "success",
            },
        ])
        mock_gh.get_workflow_run_jobs = AsyncMock(return_value=[
            {"name": "build", "status": "completed", "conclusion": "success", "id": 10},
            {"name": "optional", "status": "completed", "conclusion": "skipped", "id": 11},
        ])
        status = await ci_monitor.check_pr_ci(19876, "abc123def456")
        assert len(status.nvidia_jobs) == 1
        assert status.nvidia_jobs[0].job_name == "build"

    @pytest.mark.asyncio
    async def test_has_run_ci_label(self, ci_monitor, mock_gh):
        """Should detect the run-ci label."""
        mock_gh.get_workflow_runs_for_ref = AsyncMock(return_value=[])
        status = await ci_monitor.check_pr_ci(19876, "abc123def456")
        assert status.has_run_ci_label is True

    @pytest.mark.asyncio
    async def test_no_run_ci_label(self, ci_monitor, mock_gh):
        """Should detect missing run-ci label."""
        mock_gh.get_pull = AsyncMock(return_value={
            "number": 19876,
            "state": "open",
            "head": {"sha": "abc123def456"},
            "labels": [],
            "user": {"login": "alice"},
        })
        mock_gh.get_workflow_runs_for_ref = AsyncMock(return_value=[])
        status = await ci_monitor.check_pr_ci(19876, "abc123def456")
        assert status.has_run_ci_label is False


class TestRetryLogic:
    @pytest.mark.asyncio
    async def test_should_retry_under_limit(self, ci_monitor, mock_gh, settings):
        """Should return True when retry count is under the limit."""
        from sgldhelper.ci.monitor import CIJobResult
        job = CIJobResult(
            job_name="test-gpu", workflow_name="nvidia",
            status="completed", conclusion="failure",
            run_id=1, job_id=10,
        )
        assert await ci_monitor.should_retry(19876, "abc123", job) is True

    @pytest.mark.asyncio
    async def test_should_not_retry_at_limit(self, ci_monitor, db, settings):
        """Should return False when retry count reaches the limit."""
        from sgldhelper.ci.monitor import CIJobResult
        from sgldhelper.db import queries

        job = CIJobResult(
            job_name="test-gpu", workflow_name="nvidia",
            status="completed", conclusion="failure",
            run_id=1, job_id=10,
        )
        # Exhaust retries
        for _ in range(settings.ci_max_retries):
            await queries.increment_ci_retry(db.conn, 19876, "abc123", "test-gpu")

        assert await ci_monitor.should_retry(19876, "abc123", job) is False


class TestTriggerCI:
    @pytest.mark.asyncio
    async def test_trigger_without_label(self, ci_monitor, mock_gh):
        """Should post a comment when run-ci label is missing."""
        await ci_monitor.trigger_ci(19876, has_label=False)
        mock_gh.create_issue_comment.assert_called_once_with(19876, "/tag-and-rerun-ci")

    @pytest.mark.asyncio
    async def test_trigger_with_label(self, ci_monitor, mock_gh, settings):
        """Should comment /rerun-failed-ci when run-ci label exists and CI failed."""
        mock_gh.get_workflow_runs_for_ref = AsyncMock(return_value=[
            {
                "id": 100,
                "workflow_id": settings.ci_nvidia_workflow_id,
                "status": "completed",
                "conclusion": "failure",
            },
        ])
        await ci_monitor.trigger_ci(19876, has_label=True)
        mock_gh.create_issue_comment.assert_called_once_with(19876, "/rerun-failed-ci")


class TestHighPriority:
    @pytest.mark.asyncio
    async def test_high_priority_label_detected(self, ci_monitor, mock_gh, settings):
        """Should detect the high-priority label on a PR."""
        mock_gh.get_pull = AsyncMock(return_value={
            "number": 19876,
            "state": "open",
            "head": {"sha": "abc123def456"},
            "labels": [{"name": "run-ci"}, {"name": "high-priority"}],
            "user": {"login": "alice"},
        })
        mock_gh.get_workflow_runs_for_ref = AsyncMock(return_value=[])
        status = await ci_monitor.check_pr_ci(19876, "abc123def456")
        assert status.has_high_priority_label is True

    @pytest.mark.asyncio
    async def test_should_retry_high_priority_limit(self, ci_monitor, db, settings):
        """High-priority PRs should retry up to 10 times, normal up to 3."""
        from sgldhelper.db import queries

        job = CIJobResult(
            job_name="test-gpu", workflow_name="nvidia",
            status="completed", conclusion="failure",
            run_id=1, job_id=10,
        )
        # After 3 retries: normal=False, high-priority=True
        for _ in range(settings.ci_max_retries):
            await queries.increment_ci_retry(db.conn, 19876, "abc123", "test-gpu")

        assert await ci_monitor.should_retry(19876, "abc123", job, is_high_priority=False) is False
        assert await ci_monitor.should_retry(19876, "abc123", job, is_high_priority=True) is True

        # After 10 retries: both False
        for _ in range(settings.ci_high_priority_max_retries - settings.ci_max_retries):
            await queries.increment_ci_retry(db.conn, 19876, "abc123", "test-gpu")

        assert await ci_monitor.should_retry(19876, "abc123", job, is_high_priority=True) is False

    @pytest.mark.asyncio
    async def test_nvidia_ci_passed_helper(self, ci_monitor):
        """_nvidia_ci_passed returns True when all nvidia jobs pass, even if AMD fails."""
        ci_status = CIStatus(
            pr_number=19876,
            head_sha="abc123",
            overall=CIOverallStatus.FAILED,
            has_run_ci_label=True,
            nvidia_jobs=[
                CIJobResult("build", "nvidia", "completed", "success", 1, 10),
                CIJobResult("test", "nvidia", "completed", "success", 1, 11),
            ],
            amd_jobs=[
                CIJobResult("build", "amd", "completed", "failure", 2, 20),
            ],
            failed_jobs=[
                CIJobResult("build", "amd", "completed", "failure", 2, 20),
            ],
        )
        assert ci_monitor._nvidia_ci_passed(ci_status) is True

    @pytest.mark.asyncio
    async def test_nvidia_passed_github_ping_triggered(self, ci_monitor, mock_gh, db, settings):
        """Should trigger callback when nvidia passed + approved (no high-priority label needed)."""
        mock_gh.get_pull = AsyncMock(return_value={
            "number": 19876,
            "state": "open",
            "head": {"sha": "abc123def456"},
            "labels": [{"name": "run-ci"}],
            "user": {"login": "alice"},
        })
        # Nvidia CI passed
        mock_gh.get_workflow_runs_for_ref = AsyncMock(return_value=[
            {
                "id": 1,
                "workflow_id": settings.ci_nvidia_workflow_id,
                "status": "completed",
                "conclusion": "success",
            },
        ])
        mock_gh.get_workflow_run_jobs = AsyncMock(return_value=[
            {"name": "build", "status": "completed", "conclusion": "success", "id": 10},
        ])
        # Approved
        mock_gh.get_pull_reviews = AsyncMock(return_value=[
            {"state": "APPROVED", "user": {"login": "reviewer"}},
        ])
        mock_gh.get_pull_commits = AsyncMock(return_value=[{"sha": "abc123def456"}])

        hp_callback = AsyncMock()
        ci_monitor.set_callbacks(on_high_priority_nvidia_passed=hp_callback)

        await ci_monitor._poll_single_pr(19876, ["U123"])
        hp_callback.assert_called_once_with(19876, ["U123"], "approved")

    @pytest.mark.asyncio
    async def test_nvidia_passed_no_duplicate_ping(self, ci_monitor, mock_gh, db, settings):
        """Should NOT ping a second time on the next poll for the same SHA."""
        mock_gh.get_pull = AsyncMock(return_value={
            "number": 19876,
            "state": "open",
            "head": {"sha": "abc123def456"},
            "labels": [{"name": "run-ci"}],
            "user": {"login": "alice"},
        })
        mock_gh.get_workflow_runs_for_ref = AsyncMock(return_value=[
            {
                "id": 1,
                "workflow_id": settings.ci_nvidia_workflow_id,
                "status": "completed",
                "conclusion": "success",
            },
        ])
        mock_gh.get_workflow_run_jobs = AsyncMock(return_value=[
            {"name": "build", "status": "completed", "conclusion": "success", "id": 10},
        ])
        mock_gh.get_pull_reviews = AsyncMock(return_value=[
            {"state": "APPROVED", "user": {"login": "reviewer"}},
        ])
        mock_gh.get_pull_commits = AsyncMock(return_value=[{"sha": "abc123def456"}])

        hp_callback = AsyncMock()
        ci_monitor.set_callbacks(on_high_priority_nvidia_passed=hp_callback)

        # First poll — should ping
        await ci_monitor._poll_single_pr(19876, ["U123"])
        assert hp_callback.call_count == 1

        # Second poll — should NOT ping again
        await ci_monitor._poll_single_pr(19876, ["U123"])
        assert hp_callback.call_count == 1


class TestOwnerRerun:
    """Tests for auto-retry when owner (mickqian) commented /tag-and-rerun-ci."""

    def _setup_failed_ci(self, mock_gh, settings, *, with_owner_comment=True):
        """Set up mocks for a PR with failed CI."""
        mock_gh.get_pull = AsyncMock(return_value={
            "number": 19876,
            "state": "open",
            "head": {"sha": "abc123def456"},
            "labels": [{"name": "run-ci"}],
            "user": {"login": "alice"},
        })
        mock_gh.get_workflow_runs_for_ref = AsyncMock(return_value=[
            {
                "id": 1,
                "workflow_id": settings.ci_nvidia_workflow_id,
                "status": "completed",
                "conclusion": "failure",
            },
        ])
        mock_gh.get_workflow_run_jobs = AsyncMock(return_value=[
            {"name": "test-gpu", "status": "completed", "conclusion": "failure", "id": 10},
        ])
        comments = []
        if with_owner_comment:
            comments.append({
                "user": {"login": settings.ci_high_priority_ping_user},
                "body": "/tag-and-rerun-ci",
            })
        mock_gh.get_issue_comments = AsyncMock(return_value=comments)

    @pytest.mark.asyncio
    async def test_owner_rerun_triggers_retry(self, ci_monitor, mock_gh, db, settings):
        """Should comment /rerun-failed-ci when owner commented /tag-and-rerun-ci."""
        self._setup_failed_ci(mock_gh, settings)
        await ci_monitor._check_untracked_pr(19876)
        mock_gh.create_issue_comment.assert_called_once_with(19876, "/rerun-failed-ci")

    @pytest.mark.asyncio
    async def test_owner_rerun_no_comment_no_retry(self, ci_monitor, mock_gh, db, settings):
        """Should NOT comment /rerun-failed-ci when owner has not commented."""
        self._setup_failed_ci(mock_gh, settings, with_owner_comment=False)
        await ci_monitor._check_untracked_pr(19876)
        mock_gh.create_issue_comment.assert_not_called()

    @pytest.mark.asyncio
    async def test_owner_rerun_respects_max_retries(self, ci_monitor, mock_gh, db, settings):
        """Should stop retrying after ci_owner_rerun_max_retries."""
        from sgldhelper.db import queries

        self._setup_failed_ci(mock_gh, settings)

        # Exhaust retries
        for _ in range(settings.ci_owner_rerun_max_retries):
            await queries.increment_ci_retry(db.conn, 19876, "abc123def456", "test-gpu")

        await ci_monitor._check_untracked_pr(19876)
        mock_gh.create_issue_comment.assert_not_called()

    @pytest.mark.asyncio
    async def test_owner_rerun_caches_comment_check(self, ci_monitor, mock_gh, db, settings):
        """Should only fetch comments once per SHA (cached in snapshot_data)."""
        self._setup_failed_ci(mock_gh, settings)

        # First call — fetches comments
        await ci_monitor._check_untracked_pr(19876)
        assert mock_gh.get_issue_comments.call_count == 1

        # Second call — uses cached value
        mock_gh.create_issue_comment.reset_mock()
        await ci_monitor._check_untracked_pr(19876)
        assert mock_gh.get_issue_comments.call_count == 1  # Still 1, cached


class TestApproveAutoCI:
    """Tests for auto-CI on approval by configured users (mickqian/bbuf)."""

    def _setup_pr(self, mock_gh, settings, *, has_label=False, ci_failed=False,
                  approver="mickqian"):
        labels = [{"name": "run-ci"}] if has_label else []
        mock_gh.get_pull = AsyncMock(return_value={
            "number": 19876, "state": "open",
            "head": {"sha": "abc123def456"},
            "labels": labels,
            "user": {"login": "alice"},
            "title": "Test PR",
            "updated_at": "2025-03-01T00:00:00Z",
        })
        mock_gh.get_pull_reviews = AsyncMock(return_value=[
            {"user": {"login": approver}, "state": "APPROVED"},
        ])

        if has_label:
            conclusion = "failure" if ci_failed else "success"
            status = "completed"
            mock_gh.get_workflow_runs_for_ref = AsyncMock(return_value=[
                {"id": 1, "workflow_id": settings.ci_nvidia_workflow_id,
                 "status": status, "conclusion": conclusion},
            ])
            mock_gh.get_workflow_run_jobs = AsyncMock(return_value=[
                {"name": "build", "status": status, "conclusion": conclusion, "id": 10},
            ])
        else:
            mock_gh.get_workflow_runs_for_ref = AsyncMock(return_value=[])
            mock_gh.get_workflow_run_jobs = AsyncMock(return_value=[])

        mock_gh.get_issue_comments = AsyncMock(return_value=[])
        mock_gh.get_pull_commits = AsyncMock(return_value=[{"sha": "abc123"}])

    @pytest.mark.asyncio
    async def test_no_label_starts_ci(self, ci_monitor, mock_gh, db, settings):
        """When approved by configured user and no run-ci label, comment /tag-and-rerun-ci."""
        self._setup_pr(mock_gh, settings, has_label=False)
        await ci_monitor._check_untracked_pr(19876)
        mock_gh.create_issue_comment.assert_called_once_with(19876, "/tag-and-rerun-ci")

    @pytest.mark.asyncio
    async def test_no_label_no_duplicate(self, ci_monitor, mock_gh, db, settings):
        """Should only comment /tag-and-rerun-ci once per SHA."""
        self._setup_pr(mock_gh, settings, has_label=False)
        await ci_monitor._check_untracked_pr(19876)
        mock_gh.create_issue_comment.reset_mock()
        await ci_monitor._check_untracked_pr(19876)
        mock_gh.create_issue_comment.assert_not_called()

    @pytest.mark.asyncio
    async def test_ci_failed_retries(self, ci_monitor, mock_gh, db, settings):
        """When approved and CI failed, comment /rerun-failed-ci."""
        self._setup_pr(mock_gh, settings, has_label=True, ci_failed=True)
        await ci_monitor._check_untracked_pr(19876)
        mock_gh.create_issue_comment.assert_called_once_with(19876, "/rerun-failed-ci")

    @pytest.mark.asyncio
    async def test_ci_failed_respects_max_retries(self, ci_monitor, mock_gh, db, settings):
        """Should stop retrying after ci_approve_auto_ci_max_retries."""
        from sgldhelper.db import queries

        self._setup_pr(mock_gh, settings, has_label=True, ci_failed=True)
        for _ in range(settings.ci_approve_auto_ci_max_retries):
            await queries.increment_ci_retry(db.conn, 19876, "abc123def456", "build")

        await ci_monitor._check_untracked_pr(19876)
        mock_gh.create_issue_comment.assert_not_called()

    @pytest.mark.asyncio
    async def test_non_configured_user_no_action(self, ci_monitor, mock_gh, db, settings):
        """When approved by a user NOT in ci_approve_auto_ci_users, do nothing."""
        self._setup_pr(mock_gh, settings, has_label=False, approver="random_user")
        await ci_monitor._check_untracked_pr(19876)
        mock_gh.create_issue_comment.assert_not_called()

    @pytest.mark.asyncio
    async def test_bbuf_also_triggers(self, ci_monitor, mock_gh, db, settings):
        """bbuf approval should also trigger auto-CI."""
        self._setup_pr(mock_gh, settings, has_label=False, approver="bbuf")
        await ci_monitor._check_untracked_pr(19876)
        mock_gh.create_issue_comment.assert_called_once_with(19876, "/tag-and-rerun-ci")
