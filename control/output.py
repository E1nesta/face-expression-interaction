from __future__ import annotations

import json
from collections.abc import Callable

from core.data_structures import ControlCommand
from control.protocol import build_control_payload


class ConsoleJsonOutput:
    def __init__(self, write_line: Callable[[str], None] = print) -> None:
        self.write_line = write_line

    def send(self, command: ControlCommand) -> None:
        payload = build_control_payload(command)
        self.write_line(json.dumps(payload, ensure_ascii=False, sort_keys=True))
