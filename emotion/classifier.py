from __future__ import annotations

from typing import Any, Protocol

from core.data_structures import (
    EmotionClassificationResult,
    FaceLandmarkResult,
    FacialFeatures,
)


DEFAULT_FACE_EMOTION_LABELS = (
    "neutral",
    "happy",
    "happiness",
    "surprise",
    "sad",
    "sadness",
    "tired",
    "anger",
    "angry",
    "disgust",
    "fear",
    "contempt",
)


class EmotionClassifierError(RuntimeError):
    pass


class EmotionClassifier(Protocol):
    backend_name: str

    def classify(
        self,
        *,
        frame: Any,
        face: FaceLandmarkResult,
        features: FacialFeatures,
    ) -> EmotionClassificationResult:
        pass


def validate_face_emotion_label(label: str) -> str:
    normalized = str(label).strip().lower()
    if not normalized or normalized == "no_face":
        raise EmotionClassifierError(f"Unsupported emotion label: {label}")
    return normalized


def create_emotion_classifier(config: dict[str, Any]) -> EmotionClassifier:
    backend = str(config.get("face_backend", "rule")).lower()
    thresholds = config.get("thresholds")

    if backend == "rule":
        from emotion.rule_classifier import RuleEmotionClassifier

        return RuleEmotionClassifier(thresholds=thresholds)

    if backend == "onnx":
        try:
            from emotion.onnx_classifier import OnnxEmotionClassifier

            return OnnxEmotionClassifier(config.get("onnx", {}))
        except EmotionClassifierError:
            if bool(config.get("fallback_to_rule", True)):
                from emotion.rule_classifier import RuleEmotionClassifier

                return RuleEmotionClassifier(thresholds=thresholds)
            raise

    raise EmotionClassifierError(f"Unsupported emotion backend: {backend}")
