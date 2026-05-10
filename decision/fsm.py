from __future__ import annotations

from core.data_structures import DecisionResult, EmotionResult, clamp
from decision.expression_policy import select_expression_policy


class ExpressionDecisionFsm:
    def __init__(self, stable_frames: int = 5) -> None:
        self.stable_frames = max(1, int(stable_frames))
        self._current = select_expression_policy(EmotionResult(user_state="neutral"))
        self._candidate_state: str | None = None
        self._candidate_count = 0

    def current(self) -> DecisionResult:
        return self._copy(self._current, stable=True)

    def update(self, emotion: EmotionResult) -> DecisionResult:
        target = select_expression_policy(emotion)
        if emotion.user_state == "no_face":
            self._current = target
            self._candidate_state = None
            self._candidate_count = 0
            return self._copy(self._current, stable=True)

        if target.robot_state == self._current.robot_state and target.expression == self._current.expression:
            self._candidate_state = None
            self._candidate_count = 0
            return self._copy(self._current, stable=True)

        candidate_key = _decision_key(target)
        if candidate_key != self._candidate_state:
            self._candidate_state = candidate_key
            self._candidate_count = 1
        else:
            self._candidate_count += 1

        if self._candidate_count >= self.stable_frames:
            self._current = target
            self._candidate_state = None
            self._candidate_count = 0
            return self._copy(self._current, stable=True)

        return self._copy(self._current, stable=False)

    @staticmethod
    def _copy(result: DecisionResult, stable: bool) -> DecisionResult:
        return DecisionResult(
            robot_state=result.robot_state,
            expression=result.expression,
            intensity=clamp(result.intensity),
            stable=stable,
        )


def _decision_key(result: DecisionResult) -> str:
    return f"{result.robot_state}:{result.expression}"
