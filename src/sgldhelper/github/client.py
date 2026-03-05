"""Async GitHub API client with ETag caching and rate limiting."""

from __future__ import annotations

from itertools import combinations as _combinations
from typing import Any

import httpx
import structlog

from sgldhelper.utils.rate_limiter import TokenBucketLimiter

log = structlog.get_logger()

_NOT_MODIFIED = object()

_NO_TOKEN_MSG = (
    "GITHUB_TOKEN is not configured. "
    "Set it in your .env file to enable GitHub integration."
)


class GitHubNotConfiguredError(Exception):
    """Raised when a GitHub API call is attempted without a configured token."""

    def __init__(self) -> None:
        super().__init__(_NO_TOKEN_MSG)


class GitHubClient:
    """httpx-based async GitHub REST API client.

    Features:
    - ETag-based conditional requests to save API quota
    - Token-bucket rate limiting
    - Automatic pagination
    """

    BASE_URL = "https://api.github.com"

    def __init__(self, token: str, owner: str, repo: str) -> None:
        self._owner = owner
        self._repo = repo
        self._configured = bool(token)
        self._limiter = TokenBucketLimiter()
        self._etag_cache: dict[str, tuple[str, Any]] = {}  # url -> (etag, data)
        headers: dict[str, str] = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        if token:
            headers["Authorization"] = f"Bearer {token}"
        self._client = httpx.AsyncClient(
            base_url=self.BASE_URL,
            headers=headers,
            timeout=30.0,
        )

    def _require_token(self) -> None:
        if not self._configured:
            raise GitHubNotConfiguredError()

    async def close(self) -> None:
        await self._client.aclose()

    def _repo_url(self, path: str) -> str:
        return f"/repos/{self._owner}/{self._repo}/{path.lstrip('/')}"

    async def get(
        self,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        use_etag: bool = True,
        repo_scoped: bool = True,
    ) -> Any:
        """GET request with optional ETag caching.

        Returns cached data if the server responds 304 Not Modified.
        """
        await self._limiter.acquire()
        url = self._repo_url(path) if repo_scoped else path
        headers: dict[str, str] = {}

        cache_key = f"{url}?{params}" if params else url
        if use_etag and cache_key in self._etag_cache:
            etag, _ = self._etag_cache[cache_key]
            headers["If-None-Match"] = etag

        resp = await self._client.get(url, params=params, headers=headers)

        if resp.status_code == 304:
            _, cached_data = self._etag_cache[cache_key]
            log.debug("github.cache_hit", url=url)
            return cached_data

        resp.raise_for_status()
        data = resp.json()

        if use_etag and "ETag" in resp.headers:
            self._etag_cache[cache_key] = (resp.headers["ETag"], data)

        remaining = resp.headers.get("X-RateLimit-Remaining")
        if remaining and int(remaining) < 200:
            log.warning("github.rate_limit_low", remaining=remaining)

        return data

    async def get_paginated(
        self,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        max_pages: int = 3,
    ) -> list[dict[str, Any]]:
        """GET with pagination, collecting up to max_pages of results."""
        params = dict(params or {})
        params.setdefault("per_page", 30)
        all_items: list[dict[str, Any]] = []

        for page in range(1, max_pages + 1):
            params["page"] = page
            # Disable ETag for paginated requests (page param changes cache key anyway)
            items = await self.get(path, params=params, use_etag=False)
            if not items:
                break
            all_items.extend(items)
            if len(items) < params["per_page"]:
                break

        return all_items

    async def post(
        self,
        path: str,
        *,
        json: dict[str, Any] | None = None,
    ) -> Any:
        """POST request (used for CI rerun etc.)."""
        self._require_token()
        await self._limiter.acquire()
        url = self._repo_url(path)
        resp = await self._client.post(url, json=json)
        resp.raise_for_status()
        if resp.status_code == 204:
            return None
        return resp.json()

    async def put(
        self,
        path: str,
        *,
        json: dict[str, Any] | None = None,
    ) -> Any:
        """PUT request."""
        self._require_token()
        await self._limiter.acquire()
        url = self._repo_url(path)
        resp = await self._client.put(url, json=json)
        resp.raise_for_status()
        if resp.status_code == 204:
            return None
        return resp.json()

    # ---- Convenience methods ----

    async def get_open_pulls(self) -> list[dict[str, Any]]:
        return await self.get_paginated(
            "pulls",
            params={"state": "open", "sort": "updated", "direction": "desc"},
        )

    async def get_open_pulls_all(self, max_prs: int = 500) -> list[dict[str, Any]]:
        """Fetch up to *max_prs* open PRs. Used for cold start seeding."""
        max_pages = max(1, max_prs // 100)
        return await self.get_paginated(
            "pulls",
            params={
                "state": "open",
                "sort": "updated",
                "direction": "desc",
                "per_page": 100,
            },
            max_pages=max_pages,
        )

    async def get_pull(self, pr_number: int) -> dict[str, Any]:
        return await self.get(f"pulls/{pr_number}")

    async def get_pull_files(self, pr_number: int) -> list[dict[str, Any]]:
        return await self.get_paginated(f"pulls/{pr_number}/files")

    async def get_pull_reviews(self, pr_number: int) -> list[dict[str, Any]]:
        return await self.get_paginated(f"pulls/{pr_number}/reviews")

    async def get_workflow_runs_for_ref(
        self, head_sha: str
    ) -> list[dict[str, Any]]:
        data = await self.get(
            "actions/runs",
            params={"head_sha": head_sha, "per_page": 30},
        )
        return data.get("workflow_runs", [])

    async def get_workflow_run_jobs(
        self, run_id: int
    ) -> list[dict[str, Any]]:
        data = await self.get(f"actions/runs/{run_id}/jobs", params={"per_page": 50})
        return data.get("jobs", [])

    async def get_job_logs(self, job_id: int) -> str:
        """Download job logs as text."""
        self._require_token()
        await self._limiter.acquire()
        url = self._repo_url(f"actions/jobs/{job_id}/logs")
        resp = await self._client.get(url, follow_redirects=True)
        resp.raise_for_status()
        return resp.text

    async def get_pull_diff(self, pr_number: int) -> str:
        """Download PR diff from patch-diff.githubusercontent.com."""
        self._require_token()
        await self._limiter.acquire()
        diff_url = (
            f"https://patch-diff.githubusercontent.com/raw/"
            f"{self._owner}/{self._repo}/pull/{pr_number}.diff"
        )
        async with httpx.AsyncClient(timeout=30.0) as tmp:
            resp = await tmp.get(diff_url, follow_redirects=True)
            resp.raise_for_status()
            return resp.text

    async def rerun_failed_jobs(self, run_id: int) -> None:
        await self.post(f"actions/runs/{run_id}/rerun-failed-jobs")

    async def get_issue(self, issue_number: int) -> dict[str, Any]:
        return await self.get(f"issues/{issue_number}")

    async def get_issue_comments(
        self, issue_number: int
    ) -> list[dict[str, Any]]:
        return await self.get_paginated(f"issues/{issue_number}/comments")

    async def create_issue_comment(
        self, issue_number: int, body: str
    ) -> dict[str, Any]:
        """Create a comment on an issue or pull request."""
        return await self.post(
            f"issues/{issue_number}/comments", json={"body": body}
        )

    async def merge_pull(
        self, pr_number: int, merge_method: str = "squash"
    ) -> dict[str, Any]:
        """Merge a pull request via squash (default) or merge/rebase."""
        return await self.put(
            f"pulls/{pr_number}/merge",
            json={"merge_method": merge_method},
        )

    async def get_pull_commits(
        self, pr_number: int
    ) -> list[dict[str, Any]]:
        """Get the list of commits on a pull request."""
        return await self.get_paginated(f"pulls/{pr_number}/commits")

    async def search_issues(
        self,
        keywords: list[str],
        *,
        is_pr: bool = True,
        state: str | None = None,
        max_results: int = 10,
        min_results: int = 3,
    ) -> list[dict[str, Any]]:
        """Search issues/PRs via the GitHub Search API with fallback.

        Strategy: search all keywords (AND). If fewer than *min_results*
        are found and there are 2+ keywords, progressively drop one keyword
        at a time (from the end) and merge results until we have enough.
        """
        # Build the fixed qualifiers that are always appended
        qualifiers: list[str] = []
        qualifiers.append(f"repo:{self._owner}/{self._repo}")
        if is_pr:
            qualifiers.append("type:pr")
        if state:
            qualifiers.append(f"state:{state}")

        # --- First try: all keywords ---
        items = await self._search_once(keywords, qualifiers, max_results)

        if len(items) >= min_results or len(keywords) <= 1:
            return items[:max_results]

        # --- Fallback: progressively drop keywords ---
        seen_ids: set[int] = {item["id"] for item in items}
        merged = list(items)

        # Try subsets: drop one keyword at a time, then two, etc.
        for drop_count in range(1, len(keywords)):
            if len(merged) >= min_results:
                break
            subset_len = len(keywords) - drop_count
            for combo in _combinations(keywords, subset_len):
                if len(merged) >= min_results:
                    break
                extra = await self._search_once(
                    list(combo), qualifiers, max_results - len(merged),
                )
                for item in extra:
                    if item["id"] not in seen_ids:
                        seen_ids.add(item["id"])
                        merged.append(item)

        return merged[:max_results]

    async def _search_once(
        self,
        keywords: list[str],
        qualifiers: list[str],
        per_page: int,
    ) -> list[dict[str, Any]]:
        """Execute a single GitHub search request."""
        await self._limiter.acquire()
        q = " ".join(keywords + qualifiers)
        resp = await self._client.get(
            "/search/issues",
            params={"q": q, "per_page": per_page, "sort": "updated", "order": "desc"},
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("items", [])
