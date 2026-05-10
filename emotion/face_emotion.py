from __future__ import annotations

from core.data_structures import FacialFeatures


DEFAULT_THRESHOLDS = {
    "happy_smile": 0.62,
    "surprise_mouth_open": 0.58,
    "surprise_brow_raise": 0.45,
    "tired_eye_open": 0.32,
    "sad_smile": 0.25,
    "sad_head_pitch": -8.0,
}


def classify_face_emotion(
    features: FacialFeatures,
    thresholds: dict[str, float] | None = None,
) -> str:
    values = DEFAULT_THRESHOLDS | (thresholds or {})

    if (
        features.mouth_open >= values["surprise_mouth_open"]
        and features.brow_raise >= values["surprise_brow_raise"]
    ):
        return "surprise"

    if features.eye_open <= values["tired_eye_open"]:
        return "tired"

    if (
        features.smile_score <= values["sad_smile"]
        and features.head_pitch <= values["sad_head_pitch"]
    ):
        return "sad"

    if features.smile_score >= values["happy_smile"]:
        return "happy"

    return "neutral"

