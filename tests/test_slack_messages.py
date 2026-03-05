"""Tests for Slack Block Kit message builders."""

from __future__ import annotations

import pytest

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
