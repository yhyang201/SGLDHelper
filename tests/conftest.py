"""Shared test fixtures."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import pytest_asyncio

from sgldhelper.config import Settings
from sgldhelper.db.engine import Database

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def load_fixture(name: str):
    return json.loads((FIXTURES_DIR / name).read_text())


@pytest.fixture
def settings(tmp_path):
    """Minimal settings for testing (no real tokens)."""
    return Settings(
        github_token="ghp_test_token",
        slack_bot_token="xoxb-test",
        slack_app_token="xapp-test",
        slack_pr_channel="C_TEST_PR",
        slack_ci_channel="C_TEST_CI",
        moonshot_api_key="sk-test",
        db_path=str(tmp_path / "test.db"),
    )


@pytest_asyncio.fixture
async def db(settings):
    """In-memory test database."""
    database = Database(settings.db_path)
    await database.connect()
    yield database
    await database.close()
