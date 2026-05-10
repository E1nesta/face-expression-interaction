from __future__ import annotations

import time
from typing import Any

from core.data_structures import ControlCommand, DecisionResult, ExpressionCommand


def build_control_command(
    decision: DecisionResult,
    expression: ExpressionCommand,
    timestamp: float | None = None,
) -> ControlCommand:
    return ControlCommand(
        robot_state=decision.robot_state,
        expression=expression,
        timestamp=time.time() if timestamp is None else timestamp,
    )


def build_control_payload(command: ControlCommand) -> dict[str, Any]:
    return command.to_dict()

