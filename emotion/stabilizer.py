from __future__ import annotations

from collections import Counter, deque

from core.data_structures import EmotionClassificationResult


class EmotionStabilizer:
    def __init__(
        self,
        window_size: int = 7,
        min_votes: int = 4,
        min_confidence: float = 0.45,
        enabled: bool = True,
        reset_on_no_face: bool = True,
    ) -> None:
        self.window_size = max(1, int(window_size))
        self.min_votes = max(1, int(min_votes))
        self.min_confidence = float(min_confidence)
        self.enabled = bool(enabled)
        self.reset_on_no_face = bool(reset_on_no_face)
        self._labels = deque(maxlen=self.window_size)
        self._stable = EmotionClassificationResult(label="neutral", confidence=1.0)

    def update(
        self,
        result: EmotionClassificationResult,
        *,
        face_detected: bool,
    ) -> EmotionClassificationResult:
        if not self.enabled:
            self._stable = result.clamped()
            return self._stable

        if not face_detected and self.reset_on_no_face:
            self._labels.clear()
            self._stable = EmotionClassificationResult(label="neutral", confidence=1.0)
            return self._stable

        candidate = result.clamped()
        if candidate.confidence < self.min_confidence:
            return self._stable

        self._labels.append(candidate.label)
        label, votes = Counter(self._labels).most_common(1)[0]
        if votes >= self.min_votes or self._stable.label == "neutral":
            self._stable = EmotionClassificationResult(
                label=label,
                confidence=candidate.confidence,
                scores=candidate.scores,
            )
        return self._stable
