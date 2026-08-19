#!/usr/bin/env python
"""Read fail-closed lift limit signals from the dedicated serial controller."""

from __future__ import annotations

import logging
import re
import threading
import time

import serial

logger = logging.getLogger(__name__)


class LiftLimitReader:
    """Read ESP32 lift limit states from serial lines like:
    LIMIT,TOP=0,BOTTOM=1
    """

    _PATTERN = re.compile(r"LIMIT,TOP=(\d),BOTTOM=(\d)")

    def __init__(self, port: str, baudrate: int = 115200, stale_timeout_s: float = 1.0):
        self.port = port
        self.baudrate = baudrate
        self.stale_timeout_s = stale_timeout_s

        self.top_limit = False
        self.bottom_limit = False
        self.last_update = 0.0
        self.last_line = ""

        self._running = False
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True, name="lift-limit-reader")
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

    def is_stale(self) -> bool:
        with self._lock:
            last_update = self.last_update
        if last_update <= 0:
            return True
        return (time.time() - last_update) > self.stale_timeout_s

    def get_state(self) -> dict[str, bool | float | str]:
        with self._lock:
            return {
                "top_limit": self.top_limit,
                "bottom_limit": self.bottom_limit,
                "last_update": self.last_update,
                "stale": self.is_stale(),
                "last_line": self.last_line,
            }

    def _loop(self) -> None:
        while self._running:
            try:
                logger.info("Opening lift limit serial: %s @ %s", self.port, self.baudrate)
                with serial.Serial(self.port, self.baudrate, timeout=0.5) as ser:
                    ser.reset_input_buffer()
                    while self._running:
                        raw = ser.readline()
                        if not raw:
                            continue
                        line = raw.decode("utf-8", errors="ignore").strip()
                        m = self._PATTERN.match(line)
                        if not m:
                            continue
                        with self._lock:
                            self.top_limit = m.group(1) == "1"
                            self.bottom_limit = m.group(2) == "1"
                            self.last_update = time.time()
                            self.last_line = line
            except Exception as exc:
                logger.warning("Lift limit reader error on %s: %s", self.port, exc)
                time.sleep(1.0)
