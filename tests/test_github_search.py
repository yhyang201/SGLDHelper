"""Tests for GitHub search with progressive keyword fallback."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch, MagicMock

import pytest

from sgldhelper.github.client import GitHubClient


def _mock_search_response(items):
    """Create a mock httpx response for search results."""
    resp = MagicMock()
    resp.status_code = 200
    resp.raise_for_status = MagicMock()
    resp.json.return_value = {"total_count": len(items), "items": items}
    return resp


def _make_item(id: int, title: str):
    return {
        "id": id,
        "number": id,
        "title": title,
        "user": {"login": "alice"},
        "state": "open",
        "labels": [],
        "updated_at": "2025-03-01T00:00:00Z",
        "html_url": f"https://github.com/test/repo/pull/{id}",
    }


@pytest.fixture
def gh():
    client = GitHubClient("ghp_test", "sgl-project", "sglang")
    return client


class TestSearchFallback:
    @pytest.mark.asyncio
    async def test_all_keywords_sufficient(self, gh):
        """When all-keywords search returns enough results, no fallback."""
        items = [_make_item(1, "Fix diffusion interpolation")]

        gh._search_once = AsyncMock(return_value=items)

        result = await gh.search_issues(
            ["diffusion", "interpolation", "fix"], min_results=1,
        )
        assert len(result) == 1
        assert result[0]["number"] == 1
        # Only one search call (all keywords)
        gh._search_once.assert_called_once()

    @pytest.mark.asyncio
    async def test_fallback_to_two_keywords(self, gh):
        """When all-keywords returns too few, tries 2-keyword subsets."""
        all_kw_result = []  # nothing for all 3
        two_kw_result_1 = [_make_item(10, "diffusion interpolation")]
        two_kw_result_2 = [_make_item(11, "diffusion fix")]
        two_kw_result_3 = [_make_item(12, "interpolation fix")]

        call_count = 0
        async def mock_search(keywords, qualifiers, per_page):
            nonlocal call_count
            call_count += 1
            if len(keywords) == 3:
                return all_kw_result
            if set(keywords) == {"diffusion", "interpolation"}:
                return two_kw_result_1
            if set(keywords) == {"diffusion", "fix"}:
                return two_kw_result_2
            if set(keywords) == {"interpolation", "fix"}:
                return two_kw_result_3
            return []

        gh._search_once = mock_search

        result = await gh.search_issues(
            ["diffusion", "interpolation", "fix"], min_results=3,
        )
        # Should have collected items from 2-keyword subsets
        ids = {r["number"] for r in result}
        assert ids == {10, 11, 12}

    @pytest.mark.asyncio
    async def test_fallback_deduplicates(self, gh):
        """Same PR found in multiple subsets should appear only once."""
        same_item = _make_item(42, "diffusion fix interpolation")

        async def mock_search(keywords, qualifiers, per_page):
            if len(keywords) == 3:
                return []
            # All 2-keyword subsets return the same item
            return [same_item]

        gh._search_once = mock_search

        result = await gh.search_issues(
            ["diffusion", "interpolation", "fix"], min_results=3,
        )
        assert len(result) == 1
        assert result[0]["number"] == 42

    @pytest.mark.asyncio
    async def test_fallback_to_single_keywords(self, gh):
        """When 2-keyword subsets aren't enough, falls back to single keywords."""
        async def mock_search(keywords, qualifiers, per_page):
            if len(keywords) >= 2:
                return []
            if keywords == ["diffusion"]:
                return [_make_item(1, "diffusion model")]
            if keywords == ["interpolation"]:
                return [_make_item(2, "frame interpolation")]
            if keywords == ["fix"]:
                return [_make_item(3, "fix bug")]
            return []

        gh._search_once = mock_search

        result = await gh.search_issues(
            ["diffusion", "interpolation", "fix"], min_results=3,
        )
        ids = {r["number"] for r in result}
        assert ids == {1, 2, 3}

    @pytest.mark.asyncio
    async def test_single_keyword_no_fallback(self, gh):
        """With only 1 keyword, no fallback is possible."""
        gh._search_once = AsyncMock(return_value=[])

        result = await gh.search_issues(["diffusion"], min_results=3)
        assert result == []
        gh._search_once.assert_called_once()

    @pytest.mark.asyncio
    async def test_respects_max_results(self, gh):
        """Should not return more than max_results."""
        many_items = [_make_item(i, f"PR {i}") for i in range(20)]

        async def mock_search(keywords, qualifiers, per_page):
            return many_items[:per_page]

        gh._search_once = mock_search

        result = await gh.search_issues(
            ["diffusion"], max_results=5, min_results=1,
        )
        assert len(result) <= 5

    @pytest.mark.asyncio
    async def test_state_filter_passed(self, gh):
        """State qualifier should be included in search."""
        calls = []
        async def mock_search(keywords, qualifiers, per_page):
            calls.append(qualifiers)
            return [_make_item(1, "test")]

        gh._search_once = mock_search

        await gh.search_issues(["test"], state="open", min_results=1)
        assert "state:open" in calls[0]

    @pytest.mark.asyncio
    async def test_stops_early_when_enough(self, gh):
        """Should stop trying subsets once min_results is reached."""
        call_count = 0
        async def mock_search(keywords, qualifiers, per_page):
            nonlocal call_count
            call_count += 1
            if len(keywords) == 3:
                return []
            # First 2-keyword subset returns enough
            if set(keywords) == {"diffusion", "interpolation"}:
                return [_make_item(1, "a"), _make_item(2, "b"), _make_item(3, "c")]
            return [_make_item(10 + call_count, "extra")]

        gh._search_once = mock_search

        result = await gh.search_issues(
            ["diffusion", "interpolation", "fix"], min_results=3,
        )
        assert len(result) >= 3
        # Should have stopped after all-keywords (1) + first subset (2) = 2 calls
        assert call_count == 2
