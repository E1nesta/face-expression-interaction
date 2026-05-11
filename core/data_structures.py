from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any


PROTOCOL_VERSION = "1.0"


def clamp(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    return max(minimum, min(maximum, value))


@dataclass(slots=True)
class FrameInput:
    frame_id: int
    timestamp: float
    frame: Any
    source: str


@dataclass(slots=True)
class TextInputState:
    text: str = ""
    timestamp: float = 0.0

    @property
    def normalized_text(self) -> str:
        return self.text.strip()


@dataclass(slots=True)
class HeadPose:
    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0


@dataclass(slots=True)
class FaceLandmarkResult:
    face_detected: bool = False
    landmarks: list[tuple[float, float, float]] = field(default_factory=list)
    image_width: int = 0
    image_height: int = 0
    timestamp_ms: int = 0


@dataclass(slots=True)
class FacialFeatures:
    mouth_open: float = 0.0
    smile_score: float = 0.0
    eye_open: float = 0.0
    brow_raise: float = 0.0
    head_yaw: float = 0.0
    head_pitch: float = 0.0
    head_roll: float = 0.0

    def clamped(self) -> "FacialFeatures":
        return replace(
            self,
            mouth_open=clamp(self.mouth_open),
            smile_score=clamp(self.smile_score),
            eye_open=clamp(self.eye_open),
            brow_raise=clamp(self.brow_raise),
        )


@dataclass(slots=True)
class EmotionResult:
    face_emotion: str = "neutral"
    text_emotion: str = "neutral"
    user_state: str = "neutral"
    confidence: float = 0.0

    def clamped(self) -> "EmotionResult":
        return replace(self, confidence=clamp(self.confidence))


@dataclass(slots=True)
class EmotionClassificationResult:
    label: str
    confidence: float = 1.0
    scores: dict[str, float] = field(default_factory=dict)

    def clamped(self) -> "EmotionClassificationResult":
        return replace(self, confidence=clamp(self.confidence))


@dataclass(slots=True)
class DecisionResult:
    robot_state: str = "listening"
    expression: str = "neutral_listen"
    intensity: float = 0.35
    stable: bool = True

    def clamped(self) -> "DecisionResult":
        return replace(self, intensity=clamp(self.intensity))


@dataclass(slots=True)
class ExpressionCommand:
    expression: str
    intensity: float
    duration_ms: int
    transition_ms: int
    servo_params: dict[str, Any] = field(default_factory=dict)

    def clamped(self) -> "ExpressionCommand":
        return replace(self, intensity=clamp(self.intensity))


@dataclass(slots=True)
class ControlCommand:
    robot_state: str
    expression: ExpressionCommand
    timestamp: float
    cmd: str = "set_expression"
    protocol_version: str = PROTOCOL_VERSION
    emotion: EmotionResult | None = None
    emotion_backend: str = ""
    raw_face_emotion: str = ""
    raw_face_confidence: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        expression = self.expression.clamped()
        payload = {
            "protocol_version": self.protocol_version,
            "timestamp": self.timestamp,
            "cmd": self.cmd,
            "robot_state": self.robot_state,
            "expression": expression.expression,
            "intensity": expression.intensity,
            "duration_ms": expression.duration_ms,
            "transition_ms": expression.transition_ms,
            "servo_params": expression.servo_params,
        }
        if self.emotion is not None:
            emotion = self.emotion.clamped()
            payload.update(
                {
                    "emotion_backend": self.emotion_backend,
                    "raw_face_emotion": self.raw_face_emotion,
                    "raw_face_confidence": clamp(self.raw_face_confidence),
                    "stable_face_emotion": emotion.face_emotion,
                    "face_emotion": emotion.face_emotion,
                    "text_emotion": emotion.text_emotion,
                    "user_state": emotion.user_state,
                    "emotion_confidence": emotion.confidence,
                }
            )
        return payload
