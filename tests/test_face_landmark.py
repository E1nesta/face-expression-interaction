import numpy as np
import pytest

from perception.face_landmark import (
    FaceLandmarkBackendError,
    create_face_landmark_detector,
)


def test_factory_rejects_unknown_backend():
    with pytest.raises(FaceLandmarkBackendError):
        create_face_landmark_detector({"backend": "unknown"})


def test_factory_requires_model_path_for_face_landmarker():
    with pytest.raises(ValueError, match="model_path"):
        create_face_landmark_detector({"backend": "mediapipe_face_landmarker"})


def test_factory_rejects_missing_model_path(tmp_path):
    with pytest.raises(FileNotFoundError):
        create_face_landmark_detector(
            {
                "backend": "mediapipe_face_landmarker",
                "model_path": str(tmp_path / "missing.task"),
            }
        )


class FakeLandmarker:
    created_options = None

    @classmethod
    def create_from_options(cls, options):
        cls.created_options = options
        return cls()

    def detect_for_video(self, image, timestamp_ms):
        return type(
            "FakeResult",
            (),
            {
                "face_landmarks": [
                    [
                        type("Landmark", (), {"x": 0.1, "y": 0.2, "z": -0.01})(),
                        type("Landmark", (), {"x": 0.3, "y": 0.4, "z": -0.02})(),
                    ]
                ]
            },
        )()

    def close(self):
        self.closed = True


def test_face_landmarker_backend_returns_unified_landmarks(monkeypatch, tmp_path):
    from perception import mediapipe_face_landmarker as backend

    model_path = tmp_path / "face_landmarker.task"
    model_path.write_bytes(b"fake")
    frame = np.zeros((4, 5, 3), dtype=np.uint8)

    monkeypatch.setattr(backend, "FaceLandmarker", FakeLandmarker)

    detector = backend.MediaPipeFaceLandmarkerDetector(
        {
            "model_path": str(model_path),
            "max_num_faces": 1,
            "min_detection_confidence": 0.5,
            "min_tracking_confidence": 0.5,
            "running_mode": "video",
        }
    )

    result = detector.detect(frame, timestamp_ms=123)
    detector.close()

    assert result.face_detected is True
    assert result.image_width == 5
    assert result.image_height == 4
    assert result.timestamp_ms == 123
    assert result.landmarks == [(0.1, 0.2, -0.01), (0.3, 0.4, -0.02)]


def test_face_landmarker_backend_handles_no_face(monkeypatch, tmp_path):
    from perception import mediapipe_face_landmarker as backend

    class NoFaceLandmarker(FakeLandmarker):
        def detect_for_video(self, image, timestamp_ms):
            return type("FakeResult", (), {"face_landmarks": []})()

    model_path = tmp_path / "face_landmarker.task"
    model_path.write_bytes(b"fake")
    frame = np.zeros((4, 5, 3), dtype=np.uint8)

    monkeypatch.setattr(backend, "FaceLandmarker", NoFaceLandmarker)

    detector = backend.MediaPipeFaceLandmarkerDetector(
        {
            "model_path": str(model_path),
            "running_mode": "video",
        }
    )

    result = detector.detect(frame, timestamp_ms=456)

    assert result.face_detected is False
    assert result.landmarks == []
    assert result.timestamp_ms == 456
