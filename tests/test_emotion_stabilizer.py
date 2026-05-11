from core.data_structures import EmotionClassificationResult
from emotion.stabilizer import EmotionStabilizer


def result(label: str, confidence: float = 0.9) -> EmotionClassificationResult:
    return EmotionClassificationResult(label=label, confidence=confidence)


def test_stabilizer_uses_window_votes_to_confirm_emotion():
    stabilizer = EmotionStabilizer(window_size=5, min_votes=3, min_confidence=0.5)

    assert stabilizer.update(result("happiness"), face_detected=True).label == "happiness"
    assert stabilizer.update(result("neutral"), face_detected=True).label == "happiness"
    assert stabilizer.update(result("happiness"), face_detected=True).label == "happiness"
    assert stabilizer.update(result("sadness"), face_detected=True).label == "happiness"
    assert stabilizer.update(result("happiness"), face_detected=True).label == "happiness"


def test_stabilizer_ignores_low_confidence_candidate():
    stabilizer = EmotionStabilizer(window_size=3, min_votes=2, min_confidence=0.5)

    stabilizer.update(result("happiness", 0.9), face_detected=True)
    stable = stabilizer.update(result("sadness", 0.2), face_detected=True)

    assert stable.label == "happiness"


def test_stabilizer_switches_after_enough_votes():
    stabilizer = EmotionStabilizer(window_size=3, min_votes=2, min_confidence=0.5)

    stabilizer.update(result("happiness"), face_detected=True)
    stabilizer.update(result("sadness"), face_detected=True)
    stable = stabilizer.update(result("sadness"), face_detected=True)

    assert stable.label == "sadness"


def test_stabilizer_resets_on_no_face():
    stabilizer = EmotionStabilizer(window_size=3, min_votes=2, min_confidence=0.5)

    stabilizer.update(result("happiness"), face_detected=True)
    stable = stabilizer.update(result("sadness"), face_detected=False)

    assert stable.label == "neutral"
    assert stable.confidence == 1.0
