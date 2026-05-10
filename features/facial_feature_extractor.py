from __future__ import annotations

from math import dist

from core.data_structures import FaceLandmarkResult, FacialFeatures, HeadPose, clamp


FACE_TOP = 10
FACE_BOTTOM = 152
UPPER_LIP = 13
LOWER_LIP = 14
MOUTH_LEFT = 61
MOUTH_RIGHT = 291
LEFT_EYE_OUTER = 33
LEFT_EYE_INNER = 133
LEFT_EYE_UPPER = 159
LEFT_EYE_LOWER = 145
RIGHT_EYE_INNER = 362
RIGHT_EYE_OUTER = 263
RIGHT_EYE_UPPER = 386
RIGHT_EYE_LOWER = 374
LEFT_BROW = 70
RIGHT_BROW = 300

REQUIRED_INDICES = (
    FACE_TOP,
    FACE_BOTTOM,
    UPPER_LIP,
    LOWER_LIP,
    MOUTH_LEFT,
    MOUTH_RIGHT,
    LEFT_EYE_OUTER,
    LEFT_EYE_INNER,
    LEFT_EYE_UPPER,
    LEFT_EYE_LOWER,
    RIGHT_EYE_INNER,
    RIGHT_EYE_OUTER,
    RIGHT_EYE_UPPER,
    RIGHT_EYE_LOWER,
    LEFT_BROW,
    RIGHT_BROW,
)


def extract_facial_features(
    face: FaceLandmarkResult,
    head_pose: HeadPose,
) -> FacialFeatures:
    if not face.face_detected or len(face.landmarks) <= max(REQUIRED_INDICES):
        return FacialFeatures(
            head_yaw=head_pose.yaw,
            head_pitch=head_pose.pitch,
            head_roll=head_pose.roll,
        )

    scale = _face_scale(face)
    if scale <= 0.0:
        return FacialFeatures(
            head_yaw=head_pose.yaw,
            head_pitch=head_pose.pitch,
            head_roll=head_pose.roll,
        )

    mouth_open = _distance(face, UPPER_LIP, LOWER_LIP) / scale
    mouth_width = _distance(face, MOUTH_LEFT, MOUTH_RIGHT) / scale
    eye_open = (_eye_open(face, left=True) + _eye_open(face, left=False)) / 2.0 / scale
    brow_raise = _brow_raise(face) / scale

    features = FacialFeatures(
        mouth_open=mouth_open * 2.5,
        smile_score=mouth_width * 1.5,
        eye_open=eye_open * 4.0,
        brow_raise=brow_raise * 2.0,
        head_yaw=head_pose.yaw,
        head_pitch=head_pose.pitch,
        head_roll=head_pose.roll,
    )
    return features.clamped()


def _face_scale(face: FaceLandmarkResult) -> float:
    return _distance(face, FACE_TOP, FACE_BOTTOM)


def _eye_open(face: FaceLandmarkResult, left: bool) -> float:
    if left:
        return _distance(face, LEFT_EYE_UPPER, LEFT_EYE_LOWER)
    return _distance(face, RIGHT_EYE_UPPER, RIGHT_EYE_LOWER)


def _brow_raise(face: FaceLandmarkResult) -> float:
    left_raise = abs(face.landmarks[LEFT_EYE_UPPER][1] - face.landmarks[LEFT_BROW][1])
    right_raise = abs(face.landmarks[RIGHT_EYE_UPPER][1] - face.landmarks[RIGHT_BROW][1])
    return (left_raise + right_raise) / 2.0


def _distance(face: FaceLandmarkResult, first: int, second: int) -> float:
    return dist(face.landmarks[first][:2], face.landmarks[second][:2])


def clamp_feature(value: float) -> float:
    return clamp(value)

