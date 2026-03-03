"""Async GitHub API client with ETag caching and rate limiting."""

from __future__ import annotations

from typing import Any

import httpx
import structlog

from sgldhelper.utils.rate_limiter import TokenBucketLimiter

log = structlog.get_logger()

_NOT_MODIFIED = object()


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
        self._limiter = TokenBucketLimiter()
        self._etag_cache: dict[str, tuple[str, Any]] = {}  # url -> (etag, data)
        self._client = httpx.AsyncClient(
            base_url=self.BASE_URL,
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
            },
            timeout=30.0,
        )

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
        await self._limiter.acquire()
        url = self._repo_url(path)
        resp = await self._client.post(url, json=json)
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
        await self._limiter.acquire()
        url = self._repo_url(f"actions/jobs/{job_id}/logs")
        resp = await self._client.get(url, follow_redirects=True)
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
