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
    def test_no_signals_returns_false(self, tracker, pr_list):
        """Without file info, title or label signal, returns False."""
        # PR #1235 has no diffusion label or title keyword
        assert tracker.is_diffusion_pr(pr_list[1]) is False

    def test_label_match(self, tracker, pr_list):
        """PR with 'diffusion' label is detected even without files."""
        # PR #1234 has label 'diffusion'
        assert tracker.is_diffusion_pr(pr_list[0]) is True

    def test_title_match(self, tracker):
        """PR with 'diffusion' in title (not diffusion-llm) is detected."""
        pr = {"title": "Add diffusion model support", "labels": []}
        assert tracker.is_diffusion_pr(pr) is True

    def test_title_diffusion_llm_excluded(self, tracker):
        """PR with 'diffusion-llm' in title is NOT detected by title signal."""
        pr_llm = {"title": "Fix diffusion-llm inference bug", "labels": []}
        assert tracker.is_diffusion_pr(pr_llm) is False

        pr_llm2 = {"title": "Fix diffusion llm inference bug", "labels": []}
        assert tracker.is_diffusion_pr(pr_llm2) is False

    def test_diffusion_files(self, tracker, pr_list, pr_files_diffusion):
        """PR with multimodal_gen file paths is detected."""
        assert tracker.is_diffusion_pr(pr_list[2], pr_files_diffusion) is True

    def test_non_diffusion_files(self, tracker, pr_list, pr_files_other):
        """PR with non-diffusion files is not detected."""
        assert tracker.is_diffusion_pr(pr_list[2], pr_files_other) is False

    def test_diffusion_llm_files_excluded(self, tracker, pr_list):
        """PR touching only diffusion-llm paths must NOT match (no label/title signal)."""
        llm_files = [
            {"filename": "python/sglang/srt/models/diffusion_llm/model.py"},
        ]
        # PR #1236 has no label or title signal
        assert tracker.is_diffusion_pr(pr_list[2], llm_files) is False


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
    async def test_poll_detects_diffusion_prs(self, tracker, pr_list, db):
        """PRs matching by label, title, or files are detected."""
        diffusion_files = load_fixture("pr_files_diffusion.json")
        # Cold start uses get_open_pulls_all
        tracker._client.get_open_pulls_all = AsyncMock(return_value=pr_list)
        tracker._client.get_pull_files = AsyncMock(
            # PR 1234 and 1236 touch multimodal_gen files; 1235 does not
            side_effect=lambda n: diffusion_files if n in (1234, 1236) else []
        )
        tracker._client.get_pull = AsyncMock(side_effect=Exception("not found"))

        changes = await tracker.poll()
        prs_found = {c.pr["pr_number"] for c in changes if c.event == PREvent.OPENED}
        assert 1234 in prs_found  # has diffusion label
        assert 1236 in prs_found  # has diffusion files
        assert 1235 not in prs_found  # no diffusion signal

    @pytest.mark.asyncio
    async def test_poll_excludes_diffusion_llm(self, tracker, pr_list, db):
        """PRs that only touch diffusion-llm paths are excluded."""
        llm_files = [{"filename": "python/sglang/srt/models/diffusion_llm/x.py"}]
        # Use a pr_list where no PR has diffusion label/title
        clean_prs = [pr_list[1], pr_list[2]]  # #1235 (bug), #1236 (no label)
        tracker._client.get_open_pulls_all = AsyncMock(return_value=clean_prs)
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
        # Cold start first poll
        tracker._client.get_open_pulls_all = AsyncMock(return_value=pr_list)
        tracker._client.get_open_pulls = AsyncMock(return_value=pr_list)
        tracker._client.get_pull_files = AsyncMock(
            side_effect=lambda n: diffusion_files if n in (1234, 1236) else []
        )
        tracker._client.get_pull = AsyncMock(side_effect=Exception("not found"))

        # First poll — cold cache (get_open_pulls_all).
        # PR #1234 matches by label → no get_pull_files needed.
        # PR #1235 and #1236 need get_pull_files (no label/title match).
        await tracker.poll()
        first_call_count = tracker._client.get_pull_files.call_count
        # Only #1235 and #1236 need file fetch (label match skips #1234)
        assert first_call_count == 2

        # Reset mock counts.
        tracker._client.get_pull_files.reset_mock()

        # Second poll — same SHAs, cache should prevent any get_pull_files calls.
        await tracker.poll()
        assert tracker._client.get_pull_files.call_count == 0

    @pytest.mark.asyncio
    async def test_sha_change_triggers_reclassification(self, tracker, pr_list, db):
        """When a PR's head_sha changes, poll() must re-fetch files."""
        diffusion_files = load_fixture("pr_files_diffusion.json")
        tracker._client.get_open_pulls_all = AsyncMock(return_value=pr_list)
        tracker._client.get_pull_files = AsyncMock(
            side_effect=lambda n: diffusion_files if n in (1234, 1236) else []
        )
        tracker._client.get_pull = AsyncMock(side_effect=Exception("not found"))

        # First poll — populates cache.
        await tracker.poll()
        tracker._client.get_pull_files.reset_mock()

        # Simulate SHA change on PR #1236 (no label, needs files re-check).
        updated_pr_list = [pr.copy() for pr in pr_list]
        updated_pr_list[2] = {**pr_list[2], "head": {"sha": "newsha999999"}}
        tracker._client.get_open_pulls = AsyncMock(return_value=updated_pr_list)

        await tracker.poll()
        called_prs = [call.args[0] for call in tracker._client.get_pull_files.call_args_list]
        assert 1236 in called_prs
        assert 1235 not in called_prs
        # PR #1234 has label match → no file fetch even on SHA change
        # (but SHA didn't change anyway)

    @pytest.mark.asyncio
    async def test_new_pr_triggers_classification(self, tracker, pr_list, db):
        """A PR not in the cache must trigger get_pull_files (if no label/title match)."""
        diffusion_files = load_fixture("pr_files_diffusion.json")
        # Start with only first two PRs.
        initial_prs = pr_list[:2]
        tracker._client.get_open_pulls_all = AsyncMock(return_value=initial_prs)
        tracker._client.get_open_pulls = AsyncMock(return_value=pr_list)
        tracker._client.get_pull_files = AsyncMock(
            side_effect=lambda n: diffusion_files if n in (1234, 1236) else []
        )
        tracker._client.get_pull = AsyncMock(side_effect=Exception("not found"))

        await tracker.poll()
        tracker._client.get_pull_files.reset_mock()

        # Now all three PRs appear. #1236 is new and has no label.
        await tracker.poll()
        called_prs = [call.args[0] for call in tracker._client.get_pull_files.call_args_list]
        assert 1236 in called_prs
        assert 1234 not in called_prs  # label match, no files needed
        assert 1235 not in called_prs  # cached from first poll


class TestColdStart:
    """Tests for cold start behavior."""

    @pytest.mark.asyncio
    async def test_cold_start_uses_get_open_pulls_all(self, tracker, pr_list, db):
        """First poll with empty cache calls get_open_pulls_all."""
        tracker._client.get_open_pulls_all = AsyncMock(return_value=pr_list)
        tracker._client.get_open_pulls = AsyncMock(return_value=pr_list)
        tracker._client.get_pull_files = AsyncMock(return_value=[])
        tracker._client.get_pull = AsyncMock(side_effect=Exception("not found"))

        await tracker.poll()
        tracker._client.get_open_pulls_all.assert_called_once()
        tracker._client.get_open_pulls.assert_not_called()

    @pytest.mark.asyncio
    async def test_subsequent_poll_uses_get_open_pulls(self, tracker, pr_list, db):
        """After cache is populated, use regular get_open_pulls."""
        tracker._client.get_open_pulls_all = AsyncMock(return_value=pr_list)
        tracker._client.get_open_pulls = AsyncMock(return_value=pr_list)
        tracker._client.get_pull_files = AsyncMock(return_value=[])
        tracker._client.get_pull = AsyncMock(side_effect=Exception("not found"))

        # First poll — cold start
        await tracker.poll()

        tracker._client.get_open_pulls_all.reset_mock()
        tracker._client.get_open_pulls.reset_mock()

        # Second poll — cache populated
        await tracker.poll()
        tracker._client.get_open_pulls.assert_called_once()
        tracker._client.get_open_pulls_all.assert_not_called()
