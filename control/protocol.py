from __future__ import annotations

import time
from typing import Any

from core.data_structures import (
    ControlCommand,
    DecisionResult,
    EmotionResult,
    ExpressionCommand,
)


def build_control_command(
    decision: DecisionResult,
    expression: ExpressionCommand,
    timestamp: float | None = None,
    emotion: EmotionResult | None = None,
    emotion_backend: str = "",
    raw_face_emotion: str = "",
    raw_face_confidence: float = 0.0,
) -> ControlCommand:
    return ControlCommand(
        robot_state=decision.robot_state,
        expression=expression,
        timestamp=time.time() if timestamp is None else timestamp,
        emotion=emotion,
        emotion_backend=emotion_backend,
        raw_face_emotion=raw_face_emotion,
        raw_face_confidence=raw_face_confidence,
    )


def build_control_payload(command: ControlCommand) -> dict[str, Any]:
    return command.to_dict()
