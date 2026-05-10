from core.data_structures import FacialFeatures
from features.feature_smoother import FeatureSmoother


def test_feature_smoother_returns_first_sample_unchanged():
    sample = FacialFeatures(mouth_open=0.2, smile_score=0.4)
    smoother = FeatureSmoother(ema_alpha=0.5)

    assert smoother.update(sample) == sample


def test_feature_smoother_applies_ema_to_continuous_values():
    smoother = FeatureSmoother(ema_alpha=0.25)
    smoother.update(FacialFeatures(mouth_open=0.0, smile_score=0.0, head_yaw=0.0))

    result = smoother.update(FacialFeatures(mouth_open=1.0, smile_score=0.8, head_yaw=20.0))

    assert result.mouth_open == 0.25
    assert result.smile_score == 0.2
    assert result.head_yaw == 5.0


def test_feature_smoother_clamps_alpha():
    smoother = FeatureSmoother(ema_alpha=2.0)
    smoother.update(FacialFeatures(mouth_open=0.0))

    result = smoother.update(FacialFeatures(mouth_open=1.0))

    assert result.mouth_open == 1.0
