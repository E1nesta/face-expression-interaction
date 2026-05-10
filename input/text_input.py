from __future__ import annotations

import builtins
import threading
import time
from collections.abc import Callable

from core.data_structures import TextInputState


class AsyncTextInput:
    def __init__(
        self,
        input_func: Callable[[], str] = builtins.input,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.input_func = input_func
        self.clock = clock
        self._state = TextInputState()
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                text = self.input_func()
            except EOFError:
                break
            self.submit_text(text)

    def submit_text(self, text: str) -> None:
        with self._lock:
            self._state = TextInputState(text=text, timestamp=self.clock())

    def latest_state(self) -> TextInputState:
        with self._lock:
            return TextInputState(text=self._state.text, timestamp=self._state.timestamp)

    def stop(self) -> None:
        self._stop_event.set()

    def join(self, timeout: float | None = None) -> None:
        if self._thread is not None:
            self._thread.join(timeout=timeout)
