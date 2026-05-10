from core.data_structures import EmotionResult
from decision.expression_policy import select_expression_policy
from decision.fsm import ExpressionDecisionFsm


def test_expression_policy_maps_user_states():
    cases = {
        "no_face": ("idle", "idle_blink", 0.2),
        "happy": ("happy_reply", "happy_smile", 0.75),
        "sad": ("comforting", "gentle_care", 0.45),
        "tired": ("comforting", "soft_care", 0.35),
        "surprise": ("curious", "curious_open_eye", 0.65),
        "neutral": ("listening", "neutral_listen", 0.35),
    }

    for user_state, expected in cases.items():
        result = select_expression_policy(EmotionResult(user_state=user_state))
        assert (result.robot_state, result.expression, result.intensity) == expected


def test_expression_policy_falls_back_to_neutral_for_unknown_state():
    result = select_expression_policy(EmotionResult(user_state="unknown"))

    assert result.robot_state == "listening"
    assert result.expression == "neutral_listen"
    assert result.intensity == 0.35


def test_fsm_starts_from_neutral_listening():
    fsm = ExpressionDecisionFsm(stable_frames=2)

    result = fsm.current()

    assert result.robot_state == "listening"
    assert result.expression == "neutral_listen"
    assert result.stable is True


def test_fsm_requires_stable_frames_before_switching():
    fsm = ExpressionDecisionFsm(stable_frames=2)

    first = fsm.update(EmotionResult(user_state="happy"))
    second = fsm.update(EmotionResult(user_state="happy"))

    assert first.robot_state == "listening"
    assert first.stable is False
    assert second.robot_state == "happy_reply"
    assert second.expression == "happy_smile"
    assert second.stable is True


def test_fsm_resets_candidate_when_user_state_changes():
    fsm = ExpressionDecisionFsm(stable_frames=3)

    fsm.update(EmotionResult(user_state="happy"))
    reset = fsm.update(EmotionResult(user_state="sad"))
    still_waiting = fsm.update(EmotionResult(user_state="sad"))
    switched = fsm.update(EmotionResult(user_state="sad"))

    assert reset.robot_state == "listening"
    assert reset.stable is False
    assert still_waiting.robot_state == "listening"
    assert switched.robot_state == "comforting"
    assert switched.expression == "gentle_care"
