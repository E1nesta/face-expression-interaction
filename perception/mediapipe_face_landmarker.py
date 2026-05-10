from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import RunningMode
from mediapipe.tasks.python.vision.face_landmarker import FaceLandmarker

from core.data_structures import FaceLandmarkResult
from perception.face_landmark import FaceLandmarkDetector


class MediaPipeFaceLandmarkerDetector(FaceLandmarkDetector):
    def __init__(self, config: dict[str, Any]) -> None:
        model_path = config.get("model_path")
        if not model_path:
            raise ValueError("face detector model_path is required")
        if not Path(model_path).exists():
            raise FileNotFoundError(f"face detector model file not found: {model_path}")

        self.running_mode = _parse_running_mode(config.get("running_mode", "video"))
        options = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=self.running_mode,
            num_faces=int(config.get("max_num_faces", 1)),
            min_face_detection_confidence=float(
                config.get("min_detection_confidence", 0.5)
            ),
            min_tracking_confidence=float(config.get("min_tracking_confidence", 0.5)),
        )
        self._landmarker = FaceLandmarker.create_from_options(options)

    def detect(self, frame: Any, timestamp_ms: int = 0) -> FaceLandmarkResult:
        height, width = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        if self.running_mode == RunningMode.IMAGE:
            result = self._landmarker.detect(image)
        else:
            result = self._landmarker.detect_for_video(image, timestamp_ms)

        if not result.face_landmarks:
            return FaceLandmarkResult(
                face_detected=False,
                image_width=width,
                image_height=height,
                timestamp_ms=timestamp_ms,
            )

        landmarks = [
            (float(landmark.x), float(landmark.y), float(landmark.z))
            for landmark in result.face_landmarks[0]
        ]
        return FaceLandmarkResult(
            face_detected=True,
            landmarks=landmarks,
            image_width=width,
            image_height=height,
            timestamp_ms=timestamp_ms,
        )

    def close(self) -> None:
        self._landmarker.close()


def _parse_running_mode(value: str) -> RunningMode:
    normalized = value.lower()
    if normalized == "image":
        return RunningMode.IMAGE
    if normalized == "video":
        return RunningMode.VIDEO
    raise ValueError(f"unsupported FaceLandmarker running_mode: {value}")
