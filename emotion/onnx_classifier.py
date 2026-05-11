from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from core.data_structures import (
    EmotionClassificationResult,
    FaceLandmarkResult,
    FacialFeatures,
)
from emotion.classifier import (
    DEFAULT_FACE_EMOTION_LABELS,
    EmotionClassifierError,
    validate_face_emotion_label,
)


DEFAULT_MODEL_PATH = "models/emotion.onnx"


class OnnxEmotionClassifier:
    backend_name = "onnx"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        values = config or {}
        self.model_path = Path(values.get("model_path") or DEFAULT_MODEL_PATH)
        if not self.model_path.exists():
            raise EmotionClassifierError(f"ONNX emotion model not found: {self.model_path}")

        self.input_type = str(values.get("input_type", "features")).lower()
        if self.input_type not in {"features", "face_image"}:
            raise EmotionClassifierError(f"Unsupported ONNX emotion input type: {self.input_type}")

        self.image_size = _parse_image_size(values.get("image_size", [64, 64]))
        self.color_mode = str(values.get("color_mode", "grayscale")).lower()
        if self.color_mode not in {"grayscale", "rgb"}:
            raise EmotionClassifierError(f"Unsupported ONNX emotion color mode: {self.color_mode}")

        self.labels = tuple(
            validate_face_emotion_label(label)
            for label in (values.get("labels") or DEFAULT_FACE_EMOTION_LABELS)
        )
        for label in self.labels:
            validate_face_emotion_label(str(label))

        try:
            ort = importlib.import_module("onnxruntime")
        except ImportError as exc:
            raise EmotionClassifierError("onnxruntime is not installed") from exc

        self.session = ort.InferenceSession(str(self.model_path))
        self.input_name = self.session.get_inputs()[0].name

    def classify(
        self,
        *,
        frame: Any,
        face: FaceLandmarkResult,
        features: FacialFeatures,
    ) -> EmotionClassificationResult:
        if not face.face_detected:
            return EmotionClassificationResult(label="neutral", confidence=1.0)

        model_input = (
            _features_to_vector(features)
            if self.input_type == "features"
            else _face_to_image_tensor(
                frame=frame,
                face=face,
                image_size=self.image_size,
                color_mode=self.color_mode,
            )
        )
        outputs = self.session.run(None, {self.input_name: model_input})
        logits = np.asarray(outputs[0]).reshape(-1)
        if logits.size == 0:
            raise EmotionClassifierError("ONNX emotion model returned empty output")

        scores = _softmax(logits)
        label_index = int(np.argmax(scores))
        if label_index >= len(self.labels):
            raise EmotionClassifierError(
                f"ONNX emotion label index out of range: {label_index}"
            )
        label = validate_face_emotion_label(str(self.labels[label_index]))
        scores_by_label = {
            str(label): float(scores[index])
            for index, label in enumerate(self.labels[: scores.size])
        }
        return EmotionClassificationResult(
            label=label,
            confidence=float(scores[label_index]),
            scores=scores_by_label,
        )


def _features_to_vector(features: FacialFeatures) -> np.ndarray:
    return np.asarray(
        [
            [
                features.mouth_open,
                features.smile_score,
                features.eye_open,
                features.brow_raise,
                features.head_yaw,
                features.head_pitch,
                features.head_roll,
            ]
        ],
        dtype=np.float32,
    )


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values.astype(np.float32) - np.max(values)
    exp = np.exp(shifted)
    total = np.sum(exp)
    if total <= 0.0:
        return np.zeros_like(exp, dtype=np.float32)
    return exp / total


def _face_to_image_tensor(
    frame: Any,
    face: FaceLandmarkResult,
    image_size: tuple[int, int],
    color_mode: str,
) -> np.ndarray:
    if frame is None:
        raise EmotionClassifierError("ONNX face image input requires a frame")

    image = np.asarray(frame)
    if image.ndim != 3 or image.shape[2] < 3:
        raise EmotionClassifierError("ONNX face image input requires a BGR image")

    roi = _crop_face_roi(image, face)
    width, height = image_size
    resized = cv2.resize(roi, (width, height), interpolation=cv2.INTER_AREA)

    if color_mode == "grayscale":
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        return gray.astype(np.float32).reshape(1, 1, height, width)

    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    return np.transpose(rgb.astype(np.float32), (2, 0, 1)).reshape(1, 3, height, width)


def _crop_face_roi(frame: np.ndarray, face: FaceLandmarkResult) -> np.ndarray:
    height, width = frame.shape[:2]
    if not face.landmarks:
        return frame

    xs = [point[0] * width for point in face.landmarks]
    ys = [point[1] * height for point in face.landmarks]
    x_min, x_max = max(0, min(xs)), min(width, max(xs))
    y_min, y_max = max(0, min(ys)), min(height, max(ys))

    margin_x = (x_max - x_min) * 0.15
    margin_y = (y_max - y_min) * 0.15
    left = max(0, int(x_min - margin_x))
    right = min(width, int(x_max + margin_x))
    top = max(0, int(y_min - margin_y))
    bottom = min(height, int(y_max + margin_y))

    if right <= left or bottom <= top:
        return frame
    return frame[top:bottom, left:right]


def _parse_image_size(value: Any) -> tuple[int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise EmotionClassifierError("ONNX emotion image_size must be [width, height]")
    width, height = int(value[0]), int(value[1])
    if width <= 0 or height <= 0:
        raise EmotionClassifierError("ONNX emotion image_size values must be positive")
    return (width, height)
