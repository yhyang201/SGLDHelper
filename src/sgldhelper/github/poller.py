"""Polling scheduler using asyncio TaskGroup."""

from __future__ import annotations

import asyncio
from typing import Any, Callable, Coroutine

import structlog

log = structlog.get_logger()


class Poller:
    """Periodically execute an async callback at a fixed interval."""

    def __init__(
        self,
        name: str,
        interval: float,
        callback: Callable[[], Coroutine[Any, Any, Any]],
    ) -> None:
        self.name = name
        self.interval = interval
        self._callback = callback
        self._running = False

    async def run(self) -> None:
        """Run the poller loop. Call within a TaskGroup or as a standalone task."""
        self._running = True
        log.info("poller.started", name=self.name, interval=self.interval)

        while self._running:
            try:
                await self._callback()
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("poller.error", name=self.name)

            try:
                await asyncio.sleep(self.interval)
            except asyncio.CancelledError:
                break

        log.info("poller.stopped", name=self.name)

    def stop(self) -> None:
        self._running = False
