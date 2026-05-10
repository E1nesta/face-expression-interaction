from __future__ import annotations

from core.data_structures import TextInputState


KEYWORDS = {
    "happy": ("开心", "高兴", "快乐", "不错", "喜欢"),
    "tired": ("累", "困", "疲惫", "没精神"),
    "sad": ("难过", "伤心", "低落", "沮丧"),
    "surprise": ("惊讶", "震惊", "意外", "没想到"),
}


def classify_text_emotion(text_state: TextInputState) -> str:
    text = text_state.normalized_text
    if not text:
        return "neutral"

    for emotion, keywords in KEYWORDS.items():
        if any(keyword in text for keyword in keywords):
            return emotion

    return "neutral"

