from __future__ import annotations

from typing import Any

from core.data_structures import (
    EmotionClassificationResult,
    FaceLandmarkResult,
    FacialFeatures,
)
from emotion.classifier import validate_face_emotion_label
from emotion.face_emotion import classify_face_emotion


class RuleEmotionClassifier:
    backend_name = "rule"

    def __init__(self, thresholds: dict[str, float] | None = None) -> None:
        self.thresholds = thresholds

    def classify(
        self,
        *,
        frame: Any,
        face: FaceLandmarkResult,
        features: FacialFeatures,
    ) -> EmotionClassificationResult:
        emotion = classify_face_emotion(features, thresholds=self.thresholds)
        label = validate_face_emotion_label(emotion)
        return EmotionClassificationResult(
            label=label,
            confidence=0.7 if label != "neutral" else 0.5,
            scores={label: 0.7 if label != "neutral" else 0.5},
        )
