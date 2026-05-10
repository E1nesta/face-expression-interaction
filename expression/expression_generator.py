from __future__ import annotations

from copy import deepcopy

from core.data_structures import DecisionResult, ExpressionCommand, clamp


EXPRESSION_PROFILES = {
    "idle_blink": {
        "eye_open": 55,
        "mouth_left": 30,
        "mouth_right": 30,
        "eyebrow_left": 25,
        "eyebrow_right": 25,
        "head_pitch": 0,
        "head_yaw": 0,
    },
    "neutral_listen": {
        "eye_open": 70,
        "mouth_left": 35,
        "mouth_right": 35,
        "eyebrow_left": 30,
        "eyebrow_right": 30,
        "head_pitch": 0,
        "head_yaw": 0,
    },
    "happy_smile": {
        "eye_open": 75,
        "mouth_left": 65,
        "mouth_right": 65,
        "eyebrow_left": 35,
        "eyebrow_right": 35,
        "head_pitch": 3,
        "head_yaw": 0,
    },
    "gentle_care": {
        "eye_open": 65,
        "mouth_left": 35,
        "mouth_right": 35,
        "eyebrow_left": 20,
        "eyebrow_right": 20,
        "head_pitch": -8,
        "head_yaw": 0,
    },
    "soft_care": {
        "eye_open": 58,
        "mouth_left": 32,
        "mouth_right": 32,
        "eyebrow_left": 22,
        "eyebrow_right": 22,
        "head_pitch": -5,
        "head_yaw": 0,
    },
    "curious_open_eye": {
        "eye_open": 90,
        "mouth_left": 42,
        "mouth_right": 42,
        "eyebrow_left": 55,
        "eyebrow_right": 55,
        "head_pitch": 6,
        "head_yaw": 0,
    },
}


def generate_expression_command(
    decision: DecisionResult,
    default_duration_ms: int = 1500,
    default_transition_ms: int = 400,
) -> ExpressionCommand:
    profile = EXPRESSION_PROFILES.get(
        decision.expression,
        EXPRESSION_PROFILES["neutral_listen"],
    )
    return ExpressionCommand(
        expression=decision.expression,
        intensity=clamp(decision.intensity),
        duration_ms=default_duration_ms,
        transition_ms=default_transition_ms,
        servo_params=deepcopy(profile),
    )

