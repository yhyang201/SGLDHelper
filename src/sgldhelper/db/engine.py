"""aiosqlite connection management."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import AsyncIterator

import aiosqlite
import structlog

from sgldhelper.db.models import SCHEMA_SQL

log = structlog.get_logger()


class Database:
    """Thin wrapper around aiosqlite with schema initialisation."""

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        self._conn: aiosqlite.Connection | None = None

    async def connect(self) -> None:
        os.makedirs(os.path.dirname(self._db_path) or ".", exist_ok=True)
        self._conn = await aiosqlite.connect(self._db_path)
        self._conn.row_factory = aiosqlite.Row
        await self._conn.execute("PRAGMA journal_mode=WAL")
        await self._conn.execute("PRAGMA foreign_keys=ON")
        await self._conn.executescript(SCHEMA_SQL)
        await self._conn.commit()
        log.info("database.connected", path=self._db_path)

    async def close(self) -> None:
        if self._conn:
            await self._conn.close()
            self._conn = None
            log.info("database.closed")

    @property
    def conn(self) -> aiosqlite.Connection:
        assert self._conn is not None, "Database not connected"
        return self._conn

    @asynccontextmanager
    async def transaction(self) -> AsyncIterator[aiosqlite.Connection]:
        """Context manager that commits on success, rolls back on error."""
        try:
            yield self.conn
            await self.conn.commit()
        except Exception:
            await self.conn.rollback()
            raise
