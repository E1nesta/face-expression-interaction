from __future__ import annotations

from core.data_structures import EmotionResult, HeadPose


def fuse_emotions(
    face_detected: bool,
    face_emotion: str,
    text_emotion: str,
    head_pose: HeadPose,
) -> EmotionResult:
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

    if face_emotion != "neutral":
        return EmotionResult(
            face_emotion=face_emotion,
            text_emotion=text_emotion,
            user_state=face_emotion,
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
