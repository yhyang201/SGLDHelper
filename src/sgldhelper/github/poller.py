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
        self._stop_event = asyncio.Event()

    async def run(self) -> None:
        """Run the poller loop. Call within a TaskGroup or as a standalone task."""
        self._stop_event.clear()
        log.info("poller.started", name=self.name, interval=self.interval)

        while not self._stop_event.is_set():
            try:
                await self._callback()
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("poller.error", name=self.name)

            # Wait for interval OR stop signal, whichever comes first
            try:
                await asyncio.wait_for(
                    self._stop_event.wait(), timeout=self.interval
                )
            except asyncio.TimeoutError:
                pass
            except asyncio.CancelledError:
                break

        log.info("poller.stopped", name=self.name)

    def stop(self) -> None:
        self._stop_event.set()
