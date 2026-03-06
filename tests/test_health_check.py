"""Tests for PR health check: categorisation of open diffusion PRs."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from sgldhelper.ci.health_check import PRHealthChecker
from sgldhelper.ci.monitor import CIJobResult, CIMonitor, CIOverallStatus, CIStatus
from sgldhelper.db import queries


def _make_pr_data(pr_number, *, mergeable=True, labels=None):
    return {
        "number": pr_number,
        "state": "open",
        "mergeable": mergeable,
        "head": {"sha": f"sha{pr_number}"},
        "user": {"login": "alice"},
        "title": f"PR {pr_number}",
        "labels": [{"name": l} for l in (labels or [])],
        "updated_at": "2025-03-01T00:00:00Z",
    }


def _make_job(name: str, workflow: str = "nvidia", conclusion: str = "success") -> CIJobResult:
    return CIJobResult(
        job_name=name, workflow_name=workflow,
        status="completed", conclusion=conclusion, run_id=1, job_id=1,
    )


def _make_ci_status(
    pr_number,
    overall: CIOverallStatus,
    *,
    completed=True,
    both_ci=False,
    nvidia_conclusion: str = "success",
    amd_conclusion: str = "success",
):
    nvidia_jobs = [_make_job("build", "nvidia", nvidia_conclusion)] if both_ci else []
    amd_jobs = [_make_job("build", "amd", amd_conclusion)] if both_ci else []
    failed = [j for j in nvidia_jobs + amd_jobs if j.conclusion == "failure"]
    return CIStatus(
        pr_number=pr_number,
        head_sha=f"sha{pr_number}",
        overall=overall,
        has_run_ci_label=True,
        all_runs_completed=completed,
        nvidia_jobs=nvidia_jobs,
        amd_jobs=amd_jobs,
        failed_jobs=failed,
    )


@pytest.fixture
def mock_gh():
    return AsyncMock()


@pytest.fixture
def mock_slack():
    slack = AsyncMock()
    slack.post_message = AsyncMock(return_value={"ts": "123"})
    slack.post_message_with_context = AsyncMock(return_value={"ts": "123"})
    return slack


@pytest.fixture
def mock_channels():
    from sgldhelper.slack.channels import ChannelRouter
    return ChannelRouter(pr_channel="C_PR", ci_channel="C_CI")


@pytest.fixture
def ci_monitor(mock_gh, db, settings):
    return CIMonitor(mock_gh, db, settings)


@pytest.fixture
def health_checker(mock_gh, db, ci_monitor, mock_slack, mock_channels, settings):
    return PRHealthChecker(mock_gh, db, ci_monitor, mock_slack, mock_channels, settings)


class TestHealthCheck:
    @pytest.mark.asyncio
    async def test_empty_db_posts_all_clear(self, health_checker, mock_slack):
        """Should still post a report when there are no open PRs."""
        await health_checker.poll()
        mock_slack.post_message_with_context.assert_called_once()
        text = mock_slack.post_message_with_context.call_args.kwargs["text"]
        assert "健康检查" in text
        # All three sections should show "无"
        assert text.count("无") == 3

    @pytest.mark.asyncio
    async def test_merge_ready(self, health_checker, db, mock_gh, ci_monitor):
        """PR with CI passed + approved + mergeable → merge_ready."""
        await queries.upsert_pr(db.conn, {
            "pr_number": 100, "title": "Ready", "author": "alice",
            "state": "open", "head_sha": "sha100", "updated_at": "2025-03-01",
            "changed_files": 1,
        })
        mock_gh.get_pull = AsyncMock(return_value=_make_pr_data(100))
        mock_gh.get_pull_reviews = AsyncMock(return_value=[
            {"user": {"login": "bob"}, "state": "APPROVED"},
        ])
        mock_gh.get_workflow_runs_for_ref = AsyncMock(return_value=[])
        # Override ci_monitor to return PASSED with both workflows
        ci_monitor.check_pr_ci = AsyncMock(
            return_value=_make_ci_status(100, CIOverallStatus.PASSED, both_ci=True)
        )

        await health_checker.poll()
        text = health_checker._slack.post_message_with_context.call_args.kwargs["text"]
        assert "#100" in text
        assert "可以 Merge (1)" in text

    @pytest.mark.asyncio
    async def test_needs_review(self, health_checker, db, mock_gh, ci_monitor):
        """PR with CI passed but no approval → needs_review."""
        await queries.upsert_pr(db.conn, {
            "pr_number": 200, "title": "Needs Review", "author": "bob",
            "state": "open", "head_sha": "sha200", "updated_at": "2025-03-01",
            "changed_files": 2,
        })
        mock_gh.get_pull = AsyncMock(return_value=_make_pr_data(200))
        mock_gh.get_pull_reviews = AsyncMock(return_value=[])
        ci_monitor.check_pr_ci = AsyncMock(
            return_value=_make_ci_status(200, CIOverallStatus.PASSED, both_ci=True)
        )

        await health_checker.poll()
        text = health_checker._slack.post_message_with_context.call_args.kwargs["text"]
        assert "#200" in text
        assert "等待 Review (1)" in text

    @pytest.mark.asyncio
    async def test_ci_stalled(self, health_checker, db, mock_gh, ci_monitor):
        """PR with approval but CI not run → ci_stalled."""
        await queries.upsert_pr(db.conn, {
            "pr_number": 300, "title": "CI Stalled", "author": "carol",
            "state": "open", "head_sha": "sha300", "updated_at": "2025-03-01",
            "changed_files": 3,
        })
        mock_gh.get_pull = AsyncMock(return_value=_make_pr_data(300))
        mock_gh.get_pull_reviews = AsyncMock(return_value=[
            {"user": {"login": "bob"}, "state": "APPROVED"},
        ])
        ci_monitor.check_pr_ci = AsyncMock(
            return_value=_make_ci_status(300, CIOverallStatus.NO_CI)
        )

        await health_checker.poll()
        text = health_checker._slack.post_message_with_context.call_args.kwargs["text"]
        assert "#300" in text
        assert "CI 需处理 (1)" in text

    @pytest.mark.asyncio
    async def test_ci_failed_stalled(self, health_checker, db, mock_gh, ci_monitor):
        """PR with approval but CI failed and completed → ci_stalled."""
        await queries.upsert_pr(db.conn, {
            "pr_number": 400, "title": "CI Failed", "author": "dave",
            "state": "open", "head_sha": "sha400", "updated_at": "2025-03-01",
            "changed_files": 1,
        })
        mock_gh.get_pull = AsyncMock(return_value=_make_pr_data(400))
        mock_gh.get_pull_reviews = AsyncMock(return_value=[
            {"user": {"login": "bob"}, "state": "APPROVED"},
        ])
        ci_monitor.check_pr_ci = AsyncMock(
            return_value=_make_ci_status(400, CIOverallStatus.FAILED, completed=True)
        )

        await health_checker.poll()
        text = health_checker._slack.post_message_with_context.call_args.kwargs["text"]
        assert "#400" in text
        assert "CI 需处理" in text

    @pytest.mark.asyncio
    async def test_ci_failed_shows_workflow_detail(self, health_checker, db, mock_gh, ci_monitor):
        """When CI fails, report should show which workflow (Nvidia/AMD) failed."""
        await queries.upsert_pr(db.conn, {
            "pr_number": 500, "title": "AMD Failed", "author": "eve",
            "state": "open", "head_sha": "sha500", "updated_at": "2025-03-01",
            "changed_files": 1,
        })
        mock_gh.get_pull = AsyncMock(return_value=_make_pr_data(500))
        mock_gh.get_pull_reviews = AsyncMock(return_value=[
            {"user": {"login": "bob"}, "state": "APPROVED"},
        ])
        # Nvidia passed, AMD failed
        ci_monitor.check_pr_ci = AsyncMock(
            return_value=_make_ci_status(
                500, CIOverallStatus.FAILED, completed=True,
                both_ci=True, nvidia_conclusion="success", amd_conclusion="failure",
            )
        )

        await health_checker.poll()
        text = health_checker._slack.post_message_with_context.call_args.kwargs["text"]
        assert "#500" in text
        assert "Nvidia ✅" in text
        assert "AMD ❌" in text

    @pytest.mark.asyncio
    async def test_ci_partial_shows_workflow_detail(self, health_checker, db, mock_gh, ci_monitor):
        """Partial CI (one workflow missing) shows which is missing."""
        await queries.upsert_pr(db.conn, {
            "pr_number": 600, "title": "Only Nvidia", "author": "frank",
            "state": "open", "head_sha": "sha600", "updated_at": "2025-03-01",
            "changed_files": 1,
        })
        mock_gh.get_pull = AsyncMock(return_value=_make_pr_data(600))
        mock_gh.get_pull_reviews = AsyncMock(return_value=[
            {"user": {"login": "bob"}, "state": "APPROVED"},
        ])
        # Only nvidia passed, no AMD jobs at all
        status = CIStatus(
            pr_number=600, head_sha="sha600",
            overall=CIOverallStatus.PASSED, has_run_ci_label=True,
            all_runs_completed=True,
            nvidia_jobs=[_make_job("build", "nvidia")],
            amd_jobs=[],
        )
        ci_monitor.check_pr_ci = AsyncMock(return_value=status)

        await health_checker.poll()
        text = health_checker._slack.post_message_with_context.call_args.kwargs["text"]
        assert "#600" in text
        assert "Nvidia ✅" in text
        assert "AMD ⚪ 未运行" in text
