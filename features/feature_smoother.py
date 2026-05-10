from __future__ import annotations

from dataclasses import fields

from core.data_structures import FacialFeatures, clamp


class FeatureSmoother:
    def __init__(self, ema_alpha: float) -> None:
        self.ema_alpha = clamp(ema_alpha)
        self._previous: FacialFeatures | None = None

    def update(self, current: FacialFeatures) -> FacialFeatures:
        if self._previous is None:
            self._previous = current
            return current

        values = {}
        for field in fields(FacialFeatures):
            current_value = getattr(current, field.name)
            previous_value = getattr(self._previous, field.name)
            values[field.name] = (
                self.ema_alpha * current_value
                + (1.0 - self.ema_alpha) * previous_value
            )

        smoothed = FacialFeatures(**values)
        self._previous = smoothed
        return smoothed

    def reset(self) -> None:
        self._previous = None
