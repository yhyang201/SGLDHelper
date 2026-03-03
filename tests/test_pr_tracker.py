"""Tests for PR tracking and diffusion detection."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from sgldhelper.db import queries
from sgldhelper.github.pr_tracker import PREvent, PRTracker
from tests.conftest import load_fixture


@pytest.fixture
def pr_list():
    return load_fixture("pr_list.json")


@pytest.fixture
def pr_files_diffusion():
    return load_fixture("pr_files_diffusion.json")


@pytest.fixture
def pr_files_other():
    return load_fixture("pr_files_other.json")


@pytest.fixture
def tracker(settings, db):
    client = AsyncMock()
    return PRTracker(client, db, settings)


class TestIsDiffusionPR:
    def test_no_files_returns_false(self, tracker, pr_list):
        """Without file info, is_diffusion_pr must return False (even with label)."""
        assert tracker.is_diffusion_pr(pr_list[0]) is False

    def test_label_alone_not_enough(self, tracker, pr_list):
        """Label 'diffusion' without multimodal_gen files is NOT a match."""
        other_files = load_fixture("pr_files_other.json")
        assert tracker.is_diffusion_pr(pr_list[0], other_files) is False

    def test_diffusion_files(self, tracker, pr_list, pr_files_diffusion):
        """PR with multimodal_gen file paths is detected."""
        assert tracker.is_diffusion_pr(pr_list[2], pr_files_diffusion) is True

    def test_non_diffusion_files(self, tracker, pr_list, pr_files_other):
        """PR with non-diffusion files is not detected."""
        assert tracker.is_diffusion_pr(pr_list[2], pr_files_other) is False

    def test_diffusion_llm_excluded(self, tracker, pr_list):
        """PR touching only diffusion-llm paths must NOT match."""
        llm_files = [
            {"filename": "python/sglang/srt/models/diffusion_llm/model.py"},
        ]
        assert tracker.is_diffusion_pr(pr_list[0], llm_files) is False


class TestPRNormalize:
    def test_normalize_open_pr(self, tracker, pr_list):
        result = tracker._normalize_pr(pr_list[0])
        assert result["pr_number"] == 1234
        assert result["author"] == "mickqian"
        assert result["state"] == "open"
        assert result["head_sha"] == "abc123def456"
        assert result["changed_files"] == 12
        assert "diffusion" in result["labels"]

    def test_normalize_merged_pr(self, tracker, pr_list):
        pr = pr_list[0].copy()
        pr["merged"] = True
        pr["state"] = "closed"
        result = tracker._normalize_pr(pr)
        assert result["state"] == "merged"


class TestPRChangeDetection:
    def test_sha_change_detected(self, tracker, pr_list):
        old = tracker._normalize_pr(pr_list[0])
        new = tracker._normalize_pr(pr_list[0])
        new["head_sha"] = "newsha999"

        changes = tracker._detect_changes(old, new, pr_list[0])
        assert len(changes) == 1
        assert changes[0].event == PREvent.UPDATED

    def test_no_change(self, tracker, pr_list):
        record = tracker._normalize_pr(pr_list[0])
        changes = tracker._detect_changes(record, record, pr_list[0])
        assert len(changes) == 0


class TestPoll:
    @pytest.mark.asyncio
    async def test_poll_detects_by_files_only(self, tracker, pr_list, db):
        """Only PRs with multimodal_gen files are detected (label is ignored)."""
        diffusion_files = load_fixture("pr_files_diffusion.json")
        tracker._client.get_open_pulls = AsyncMock(return_value=pr_list)
        tracker._client.get_pull_files = AsyncMock(
            # PR 1234 and 1236 touch multimodal_gen files; 1235 does not
            side_effect=lambda n: diffusion_files if n in (1234, 1236) else []
        )
        tracker._client.get_pull = AsyncMock(side_effect=Exception("not found"))

        changes = await tracker.poll()
        prs_found = {c.pr["pr_number"] for c in changes if c.event == PREvent.OPENED}
        assert 1234 in prs_found  # has diffusion files
        assert 1236 in prs_found  # has diffusion files
        assert 1235 not in prs_found  # no diffusion files

    @pytest.mark.asyncio
    async def test_poll_excludes_diffusion_llm(self, tracker, pr_list, db):
        """PRs that only touch diffusion-llm paths are excluded."""
        llm_files = [{"filename": "python/sglang/srt/models/diffusion_llm/x.py"}]
        tracker._client.get_open_pulls = AsyncMock(return_value=pr_list)
        tracker._client.get_pull_files = AsyncMock(return_value=llm_files)
        tracker._client.get_pull = AsyncMock(side_effect=Exception("not found"))

        changes = await tracker.poll()
        assert len(changes) == 0


class TestClassificationCache:
    """Tests for the pr_classification cache used by poll()."""

    @pytest.mark.asyncio
    async def test_cache_hit_skips_get_pull_files(self, tracker, pr_list, db):
        """Second poll should NOT call get_pull_files for unchanged PRs."""
        diffusion_files = load_fixture("pr_files_diffusion.json")
        tracker._client.get_open_pulls = AsyncMock(return_value=pr_list)
        tracker._client.get_pull_files = AsyncMock(
            side_effect=lambda n: diffusion_files if n in (1234, 1236) else []
        )
        tracker._client.get_pull = AsyncMock(side_effect=Exception("not found"))

        # First poll — cold cache, must call get_pull_files for every PR.
        await tracker.poll()
        assert tracker._client.get_pull_files.call_count == len(pr_list)

        # Reset mock counts.
        tracker._client.get_pull_files.reset_mock()

        # Second poll — same SHAs, cache should prevent any get_pull_files calls.
        await tracker.poll()
        assert tracker._client.get_pull_files.call_count == 0

    @pytest.mark.asyncio
    async def test_sha_change_triggers_reclassification(self, tracker, pr_list, db):
        """When a PR's head_sha changes, poll() must re-fetch files."""
        diffusion_files = load_fixture("pr_files_diffusion.json")
        tracker._client.get_open_pulls = AsyncMock(return_value=pr_list)
        tracker._client.get_pull_files = AsyncMock(
            side_effect=lambda n: diffusion_files if n in (1234, 1236) else []
        )
        tracker._client.get_pull = AsyncMock(side_effect=Exception("not found"))

        # First poll — populates cache.
        await tracker.poll()
        tracker._client.get_pull_files.reset_mock()

        # Simulate SHA change on PR #1234.
        updated_pr_list = [pr.copy() for pr in pr_list]
        updated_pr_list[0] = {**pr_list[0], "head": {"sha": "newsha999999"}}
        tracker._client.get_open_pulls = AsyncMock(return_value=updated_pr_list)

        await tracker.poll()
        # Only PR #1234 should trigger get_pull_files (SHA changed).
        called_prs = [call.args[0] for call in tracker._client.get_pull_files.call_args_list]
        assert 1234 in called_prs
        assert 1235 not in called_prs
        assert 1236 not in called_prs

    @pytest.mark.asyncio
    async def test_new_pr_triggers_classification(self, tracker, pr_list, db):
        """A PR not in the cache must trigger get_pull_files."""
        diffusion_files = load_fixture("pr_files_diffusion.json")
        # Start with only first two PRs.
        initial_prs = pr_list[:2]
        tracker._client.get_open_pulls = AsyncMock(return_value=initial_prs)
        tracker._client.get_pull_files = AsyncMock(
            side_effect=lambda n: diffusion_files if n in (1234, 1236) else []
        )
        tracker._client.get_pull = AsyncMock(side_effect=Exception("not found"))

        await tracker.poll()
        tracker._client.get_pull_files.reset_mock()

        # Now a third PR appears.
        tracker._client.get_open_pulls = AsyncMock(return_value=pr_list)
        await tracker.poll()
        # Only the new PR #1236 should need get_pull_files.
        called_prs = [call.args[0] for call in tracker._client.get_pull_files.call_args_list]
        assert 1236 in called_prs
        assert 1234 not in called_prs
        assert 1235 not in called_prs
