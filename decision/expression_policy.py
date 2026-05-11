from __future__ import annotations

from core.data_structures import DecisionResult, EmotionResult


POLICY_TABLE = {
    "no_face": DecisionResult(
        robot_state="idle",
        expression="idle_blink",
        intensity=0.2,
    ),
    "happy": DecisionResult(
        robot_state="happy_reply",
        expression="happy_smile",
        intensity=0.75,
    ),
    "sad": DecisionResult(
        robot_state="comforting",
        expression="gentle_care",
        intensity=0.45,
    ),
    "tired": DecisionResult(
        robot_state="comforting",
        expression="soft_care",
        intensity=0.35,
    ),
    "surprise": DecisionResult(
        robot_state="curious",
        expression="curious_open_eye",
        intensity=0.65,
    ),
    "angry": DecisionResult(
        robot_state="calming",
        expression="neutral_listen",
        intensity=0.3,
    ),
    "fear": DecisionResult(
        robot_state="comforting",
        expression="gentle_care",
        intensity=0.5,
    ),
    "disgust": DecisionResult(
        robot_state="calming",
        expression="neutral_listen",
        intensity=0.3,
    ),
    "contempt": DecisionResult(
        robot_state="calming",
        expression="neutral_listen",
        intensity=0.3,
    ),
    "neutral": DecisionResult(
        robot_state="listening",
        expression="neutral_listen",
        intensity=0.35,
    ),
}


def select_expression_policy(emotion: EmotionResult) -> DecisionResult:
    result = POLICY_TABLE.get(emotion.user_state, POLICY_TABLE["neutral"])
    return DecisionResult(
        robot_state=result.robot_state,
        expression=result.expression,
        intensity=result.intensity,
        stable=True,
    )
