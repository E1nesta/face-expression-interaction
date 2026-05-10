import numpy as np

from core.data_structures import FaceLandmarkResult
from perception.head_pose import estimate_head_pose


def test_head_pose_returns_default_when_no_face():
    result = estimate_head_pose(FaceLandmarkResult(face_detected=False))

    assert result.yaw == 0.0
    assert result.pitch == 0.0
    assert result.roll == 0.0


def test_head_pose_returns_default_when_landmarks_are_insufficient():
    result = estimate_head_pose(
        FaceLandmarkResult(
            face_detected=True,
            landmarks=[(0.1, 0.2, 0.0)],
            image_width=640,
            image_height=480,
        )
    )

    assert result.yaw == 0.0
    assert result.pitch == 0.0
    assert result.roll == 0.0


def test_head_pose_uses_solve_pnp_when_required_landmarks_exist(monkeypatch):
    landmarks = [(0.5, 0.5, 0.0) for _ in range(468)]
    calls = {}

    def fake_solve_pnp(model_points, image_points, camera_matrix, dist_coeffs):
        calls["model_points"] = model_points
        calls["image_points"] = image_points
        calls["camera_matrix"] = camera_matrix
        return True, np.zeros((3, 1), dtype=np.float64), np.zeros((3, 1), dtype=np.float64)

    monkeypatch.setattr("perception.head_pose.cv2.solvePnP", fake_solve_pnp)
    monkeypatch.setattr(
        "perception.head_pose.cv2.Rodrigues",
        lambda rotation_vector: (np.eye(3, dtype=np.float64), None),
    )
    monkeypatch.setattr(
        "perception.head_pose.cv2.RQDecomp3x3",
        lambda rotation_matrix: ((1.0, 2.0, 3.0), None, None, None, None, None),
    )

    result = estimate_head_pose(
        FaceLandmarkResult(
            face_detected=True,
            landmarks=landmarks,
            image_width=640,
            image_height=480,
        )
    )

    assert result.pitch == 1.0
    assert result.yaw == 2.0
    assert result.roll == 3.0
    assert calls["image_points"].shape == (6, 2)

