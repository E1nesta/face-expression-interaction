from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from core.data_structures import FaceLandmarkResult


class FaceLandmarkBackendError(ValueError):
    pass


class FaceLandmarkDetector(ABC):
    @abstractmethod
    def detect(self, frame: Any, timestamp_ms: int = 0) -> FaceLandmarkResult:
        raise NotImplementedError

    def close(self) -> None:
        return None


def create_face_landmark_detector(config: dict[str, Any]) -> FaceLandmarkDetector:
    backend = config.get("backend")
    if backend != "mediapipe_face_landmarker":
        raise FaceLandmarkBackendError(f"unsupported face detector backend: {backend}")

    model_path = config.get("model_path")
    if not model_path:
        raise ValueError("face detector model_path is required")

    if not Path(model_path).exists():
        raise FileNotFoundError(f"face detector model file not found: {model_path}")

    from perception.mediapipe_face_landmarker import MediaPipeFaceLandmarkerDetector

    return MediaPipeFaceLandmarkerDetector(config)

