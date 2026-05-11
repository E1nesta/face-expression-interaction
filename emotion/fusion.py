from __future__ import annotations

from core.data_structures import EmotionResult, HeadPose


DEFAULT_FACE_LABEL_MAPPING = {
    "happy": "happy",
    "happiness": "happy",
    "sad": "sad",
    "sadness": "sad",
    "surprise": "surprise",
    "tired": "tired",
    "anger": "angry",
    "angry": "angry",
    "fear": "fear",
    "disgust": "disgust",
    "contempt": "contempt",
    "neutral": "neutral",
}


def fuse_emotions(
    face_detected: bool,
    face_emotion: str,
    text_emotion: str,
    head_pose: HeadPose,
    face_label_mapping: dict[str, str] | None = None,
) -> EmotionResult:
    mapped_face_state = map_face_emotion_to_user_state(
        face_emotion,
        face_label_mapping=face_label_mapping,
    )

    if not face_detected:
        return EmotionResult(
            face_emotion="neutral",
            text_emotion=text_emotion,
            user_state="no_face",
            confidence=1.0,
        )

    if text_emotion != "neutral":
        return EmotionResult(
            face_emotion=face_emotion,
            text_emotion=text_emotion,
            user_state=text_emotion,
            confidence=0.8,
        )

    if mapped_face_state != "neutral":
        return EmotionResult(
            face_emotion=face_emotion,
            text_emotion=text_emotion,
            user_state=mapped_face_state,
            confidence=0.7,
        )

    if head_pose.pitch <= -12.0:
        return EmotionResult(
            face_emotion=face_emotion,
            text_emotion=text_emotion,
            user_state="tired",
            confidence=0.55,
        )

    return EmotionResult(
        face_emotion=face_emotion,
        text_emotion=text_emotion,
        user_state="neutral",
        confidence=0.5,
    )


def map_face_emotion_to_user_state(
    face_emotion: str,
    face_label_mapping: dict[str, str] | None = None,
) -> str:
    label = str(face_emotion).strip().lower()
    mapping = DEFAULT_FACE_LABEL_MAPPING | {
        str(key).strip().lower(): str(value).strip().lower()
        for key, value in (face_label_mapping or {}).items()
    }
    return mapping.get(label, label or "neutral")
