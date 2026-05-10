from core.data_structures import FacialFeatures, HeadPose, TextInputState
from emotion.face_emotion import classify_face_emotion
from emotion.fusion import fuse_emotions
from emotion.text_emotion import classify_text_emotion


def test_classify_face_emotion_happy():
    emotion = classify_face_emotion(FacialFeatures(smile_score=0.8, eye_open=0.7))

    assert emotion == "happy"


def test_classify_face_emotion_surprise_has_priority_over_happy():
    emotion = classify_face_emotion(
        FacialFeatures(smile_score=0.8, mouth_open=0.8, brow_raise=0.6)
    )

    assert emotion == "surprise"


def test_classify_face_emotion_tired_and_sad():
    assert classify_face_emotion(FacialFeatures(eye_open=0.2)) == "tired"
    assert (
        classify_face_emotion(
            FacialFeatures(smile_score=0.1, eye_open=0.6, head_pitch=-12.0)
        )
        == "sad"
    )


def test_classify_face_emotion_neutral():
    emotion = classify_face_emotion(FacialFeatures(smile_score=0.4, eye_open=0.5))

    assert emotion == "neutral"


def test_classify_text_emotion_keywords():
    assert classify_text_emotion(TextInputState("今天很开心")) == "happy"
    assert classify_text_emotion(TextInputState("我有点困")) == "tired"
    assert classify_text_emotion(TextInputState("有点伤心")) == "sad"
    assert classify_text_emotion(TextInputState("太震惊了")) == "surprise"
    assert classify_text_emotion(TextInputState("   ")) == "neutral"
    assert classify_text_emotion(TextInputState("普通的一天")) == "neutral"


def test_fuse_emotions_uses_no_face_state():
    result = fuse_emotions(
        face_detected=False,
        face_emotion="happy",
        text_emotion="sad",
        head_pose=HeadPose(),
    )

    assert result.face_emotion == "neutral"
    assert result.text_emotion == "sad"
    assert result.user_state == "no_face"
    assert result.confidence == 1.0


def test_fuse_emotions_prefers_non_neutral_text_over_neutral_face():
    result = fuse_emotions(
        face_detected=True,
        face_emotion="neutral",
        text_emotion="tired",
        head_pose=HeadPose(),
    )

    assert result.user_state == "tired"
    assert result.confidence == 0.8


def test_fuse_emotions_keeps_strong_face_emotion_when_text_is_neutral():
    result = fuse_emotions(
        face_detected=True,
        face_emotion="happy",
        text_emotion="neutral",
        head_pose=HeadPose(),
    )

    assert result.user_state == "happy"
    assert result.confidence == 0.7


def test_fuse_emotions_head_pose_can_support_tired_state():
    result = fuse_emotions(
        face_detected=True,
        face_emotion="neutral",
        text_emotion="neutral",
        head_pose=HeadPose(pitch=-15.0),
    )

    assert result.user_state == "tired"
    assert result.confidence == 0.55
