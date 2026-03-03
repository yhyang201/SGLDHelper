"""Tests for Slack Block Kit message builders."""

from __future__ import annotations

import pytest

from sgldhelper.github.ci_analyzer import CIResult, FailureCategory
from sgldhelper.github.issue_tracker import FeatureProgress
from sgldhelper.github.pr_tracker import PRChange, PREvent
from sgldhelper.slack import messages

REPO = "sgl-project/sglang"


class TestPRMessages:
    def _make_change(self, event, pr_number=1234):
        pr = {
            "pr_number": pr_number,
            "title": "Add Wan2.1 support",
            "author": "mickqian",
            "state": "open",
            "head_sha": "abc123def456",
            "changed_files": 12,
            "labels": ["diffusion"],
        }
        old_state = {**pr, "head_sha": "old000sha111"} if event == PREvent.UPDATED else None
        return PRChange(event=event, pr=pr, old_state=old_state)

    def test_pr_opened_message(self):
        change = self._make_change(PREvent.OPENED)
        msg = messages.build_pr_opened(change, REPO)
        assert "1234" in msg["text"]
        assert msg["blocks"][0]["type"] == "section"
        assert "mickqian" in msg["blocks"][0]["text"]["text"]

    def test_pr_updated_message(self):
        change = self._make_change(PREvent.UPDATED)
        msg = messages.build_pr_updated(change, REPO)
        assert "updated" in msg["text"]
        assert "old000sh" in msg["blocks"][0]["text"]["text"]
        assert "abc123de" in msg["blocks"][0]["text"]["text"]

    def test_pr_merged_message(self):
        change = self._make_change(PREvent.MERGED)
        msg = messages.build_pr_merged(change, REPO)
        assert "merged" in msg["text"]

    def test_pr_closed_message(self):
        change = self._make_change(PREvent.CLOSED)
        msg = messages.build_pr_closed(change, REPO)
        assert "closed" in msg["text"]


class TestCIMessages:
    def _make_result(self, category=FailureCategory.FLAKY):
        return CIResult(
            run_id=9001,
            pr_number=1234,
            job_name="multimodal-gen-test-1-gpu",
            head_sha="abc123",
            status="completed",
            conclusion="failure",
            failure_category=category,
            failure_summary="SafetensorError: corrupted",
            html_url="https://github.com/sgl-project/sglang/actions/runs/9001",
            auto_rerun_count=0,
        )

    def test_ci_failure_message(self):
        result = self._make_result()
        msg = messages.build_ci_failure(result, REPO)
        assert "CI Failure" in msg["text"] or "CI failure" in msg["text"]
        assert "flaky" in msg["blocks"][0]["text"]["text"]
        # Should have rerun button
        actions = msg["blocks"][1]
        assert actions["type"] == "actions"
        assert len(actions["elements"]) == 2

    def test_ci_success_message(self):
        msg = messages.build_ci_success(1234, REPO)
        assert "passed" in msg["text"]

    def test_ci_rerun_message(self):
        result = self._make_result()
        msg = messages.build_ci_rerun(result, auto=True, repo=REPO)
        assert "Auto-rerun" in msg["blocks"][0]["text"]["text"]

    def test_ci_rerun_manual_message(self):
        result = self._make_result()
        msg = messages.build_ci_rerun(result, auto=False, repo=REPO)
        assert "Manual rerun" in msg["blocks"][0]["text"]["text"]


class TestFeatureMessages:
    def test_feature_progress_message(self):
        progress = FeatureProgress(
            issue_number=14199,
            title="dLLM Diffusion Roadmap",
            total=10,
            completed=4,
            items=[
                {"title": "Flux support", "state": "completed"},
                {"title": "Wan 2.0", "state": "completed"},
                {"title": "CogVideo", "state": "open"},
            ],
        )
        msg = messages.build_feature_progress(progress, REPO)
        assert "40%" in msg["blocks"][0]["text"]["text"]
        assert "4/10" in msg["blocks"][0]["text"]["text"]

    def test_daily_digest(self):
        prs = [
            {"pr_number": 1234, "title": "Test PR", "author": "dev1"},
        ]
        progress = [
            FeatureProgress(
                issue_number=14199,
                title="Roadmap",
                total=10,
                completed=4,
                items=[],
            )
        ]
        msg = messages.build_daily_digest(prs, progress, REPO)
        assert "Daily" in msg["text"]
        assert "1234" in msg["blocks"][1]["text"]["text"]
