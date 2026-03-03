"""Tests for CI failure analysis and classification."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from sgldhelper.db import queries
from sgldhelper.github.ci_analyzer import (
    CIAnalyzer,
    FailureCategory,
)
from tests.conftest import load_fixture


@pytest.fixture
def analyzer(settings, db):
    client = AsyncMock()
    return CIAnalyzer(client, db, settings)


@pytest.fixture
def workflow_runs():
    return load_fixture("workflow_runs.json")


@pytest.fixture
def workflow_jobs():
    return load_fixture("workflow_jobs.json")


class TestIsDiffusionJob:
    def test_matches_1gpu(self, analyzer):
        assert analyzer._is_diffusion_job("multimodal-gen-test-1-gpu") is True

    def test_matches_2gpu(self, analyzer):
        assert analyzer._is_diffusion_job("multimodal-gen-test-2-gpu") is True

    def test_case_insensitive(self, analyzer):
        assert analyzer._is_diffusion_job("Multimodal-Gen-Test-1-GPU") is True

    def test_no_match(self, analyzer):
        assert analyzer._is_diffusion_job("unit-test") is False
        assert analyzer._is_diffusion_job("e2e-test") is False


class TestClassifyFailure:
    @pytest.mark.asyncio
    async def test_flaky_safetensor(self, analyzer):
        analyzer._client.get_job_logs = AsyncMock(
            return_value="Loading model...\nSafetensorError: corrupted file\nProcess exited"
        )
        cat, summary = await analyzer._classify_failure({"id": 1})
        assert cat == FailureCategory.FLAKY
        assert "SafetensorError" in summary

    @pytest.mark.asyncio
    async def test_infra_oom(self, analyzer):
        analyzer._client.get_job_logs = AsyncMock(
            return_value="Running test...\nout of memory\nKilled"
        )
        cat, summary = await analyzer._classify_failure({"id": 2})
        assert cat == FailureCategory.INFRA
        assert "out of memory" in summary

    @pytest.mark.asyncio
    async def test_perf_regression(self, analyzer):
        analyzer._client.get_job_logs = AsyncMock(
            return_value="AssertionError in test_server_utils: throughput too low"
        )
        cat, summary = await analyzer._classify_failure({"id": 3})
        assert cat == FailureCategory.PERF_REGRESSION

    @pytest.mark.asyncio
    async def test_code_failure(self, analyzer):
        analyzer._client.get_job_logs = AsyncMock(
            return_value="TypeError: unsupported operand\nFAILED test_something"
        )
        cat, summary = await analyzer._classify_failure({"id": 4})
        assert cat == FailureCategory.CODE

    @pytest.mark.asyncio
    async def test_log_fetch_error(self, analyzer):
        analyzer._client.get_job_logs = AsyncMock(side_effect=Exception("403"))
        cat, summary = await analyzer._classify_failure({"id": 5})
        assert cat == FailureCategory.UNKNOWN


class TestAnalyzePR:
    @pytest.mark.asyncio
    async def test_analyze_filters_diffusion_jobs(
        self, analyzer, workflow_runs, workflow_jobs
    ):
        analyzer._client.get_workflow_runs_for_ref = AsyncMock(
            return_value=workflow_runs["workflow_runs"]
        )
        analyzer._client.get_workflow_run_jobs = AsyncMock(
            return_value=workflow_jobs["jobs"]
        )
        analyzer._client.get_job_logs = AsyncMock(
            return_value="SafetensorError: bad file"
        )

        # Seed the PR so foreign key constraint passes
        await queries.upsert_pr(analyzer._db.conn, {
            "pr_number": 1234,
            "title": "Test PR",
            "author": "testuser",
            "state": "open",
            "head_sha": "abc123def456",
            "updated_at": "2025-03-01T10:00:00Z",
            "changed_files": 5,
            "labels": [],
        })

        results = await analyzer.analyze_pr(1234, "abc123def456")
        # Should only include the 2 diffusion jobs, not unit-test
        assert len(results) == 2
        job_names = {r.job_name for r in results}
        assert "unit-test" not in job_names
        assert "multimodal-gen-test-1-gpu" in job_names
        assert "multimodal-gen-test-2-gpu" in job_names
