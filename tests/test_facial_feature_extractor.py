from core.data_structures import FaceLandmarkResult, HeadPose
from features.facial_feature_extractor import extract_facial_features


def make_landmarks():
    landmarks = [(0.5, 0.5, 0.0) for _ in range(468)]
    landmarks[10] = (0.5, 0.1, 0.0)
    landmarks[152] = (0.5, 0.9, 0.0)
    landmarks[13] = (0.5, 0.55, 0.0)
    landmarks[14] = (0.5, 0.65, 0.0)
    landmarks[61] = (0.35, 0.6, 0.0)
    landmarks[291] = (0.65, 0.6, 0.0)
    landmarks[33] = (0.4, 0.4, 0.0)
    landmarks[133] = (0.48, 0.4, 0.0)
    landmarks[159] = (0.44, 0.37, 0.0)
    landmarks[145] = (0.44, 0.43, 0.0)
    landmarks[362] = (0.52, 0.4, 0.0)
    landmarks[263] = (0.6, 0.4, 0.0)
    landmarks[386] = (0.56, 0.37, 0.0)
    landmarks[374] = (0.56, 0.43, 0.0)
    landmarks[70] = (0.44, 0.28, 0.0)
    landmarks[300] = (0.56, 0.28, 0.0)
    return landmarks


def test_extract_facial_features_returns_default_when_no_face():
    features = extract_facial_features(FaceLandmarkResult(face_detected=False), HeadPose())

    assert features.mouth_open == 0.0
    assert features.smile_score == 0.0
    assert features.eye_open == 0.0
    assert features.brow_raise == 0.0


def test_extract_facial_features_normalizes_values_and_carries_head_pose():
    features = extract_facial_features(
        FaceLandmarkResult(
            face_detected=True,
            landmarks=make_landmarks(),
            image_width=640,
            image_height=480,
        ),
        HeadPose(yaw=5.0, pitch=-3.0, roll=1.0),
    )

    assert 0.0 <= features.mouth_open <= 1.0
    assert 0.0 <= features.smile_score <= 1.0
    assert 0.0 <= features.eye_open <= 1.0
    assert 0.0 <= features.brow_raise <= 1.0
    assert features.mouth_open > 0.0
    assert features.eye_open > 0.0
    assert features.head_yaw == 5.0
    assert features.head_pitch == -3.0
    assert features.head_roll == 1.0

