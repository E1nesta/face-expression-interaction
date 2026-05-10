from __future__ import annotations

import cv2
import numpy as np

from core.data_structures import FaceLandmarkResult, HeadPose


POSE_LANDMARK_INDICES = (1, 152, 33, 263, 61, 291)


MODEL_POINTS = np.array(
    [
        (0.0, 0.0, 0.0),
        (0.0, -63.6, -12.5),
        (-43.3, 32.7, -26.0),
        (43.3, 32.7, -26.0),
        (-28.9, -28.9, -24.1),
        (28.9, -28.9, -24.1),
    ],
    dtype=np.float64,
)


def estimate_head_pose(face: FaceLandmarkResult) -> HeadPose:
    if not face.face_detected or not _has_required_landmarks(face):
        return HeadPose()

    image_points = np.array(
        [
            _landmark_to_pixel(face.landmarks[index], face.image_width, face.image_height)
            for index in POSE_LANDMARK_INDICES
        ],
        dtype=np.float64,
    )

    focal_length = float(face.image_width)
    center = (face.image_width / 2.0, face.image_height / 2.0)
    camera_matrix = np.array(
        [
            [focal_length, 0.0, center[0]],
            [0.0, focal_length, center[1]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    ok, rotation_vector, _translation_vector = cv2.solvePnP(
        MODEL_POINTS,
        image_points,
        camera_matrix,
        dist_coeffs,
    )
    if not ok:
        return HeadPose()

    rotation_matrix, _jacobian = cv2.Rodrigues(rotation_vector)
    angles = cv2.RQDecomp3x3(rotation_matrix)[0]
    pitch, yaw, roll = (float(angles[0]), float(angles[1]), float(angles[2]))
    return HeadPose(yaw=yaw, pitch=pitch, roll=roll)


def _has_required_landmarks(face: FaceLandmarkResult) -> bool:
    if face.image_width <= 0 or face.image_height <= 0:
        return False
    return len(face.landmarks) > max(POSE_LANDMARK_INDICES)


def _landmark_to_pixel(
    landmark: tuple[float, float, float],
    image_width: int,
    image_height: int,
) -> tuple[float, float]:
    return (landmark[0] * image_width, landmark[1] * image_height)

