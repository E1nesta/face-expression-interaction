from core.data_structures import (
    ControlCommand,
    DecisionResult,
    EmotionResult,
    ExpressionCommand,
    FacialFeatures,
)


def test_facial_features_clamp_normalized_values():
    features = FacialFeatures(
        mouth_open=1.4,
        smile_score=-0.2,
        eye_open=0.5,
        brow_raise=2.0,
        head_yaw=12.0,
        head_pitch=-4.0,
        head_roll=1.5,
    ).clamped()

    assert features.mouth_open == 1.0
    assert features.smile_score == 0.0
    assert features.eye_open == 0.5
    assert features.brow_raise == 1.0
    assert features.head_yaw == 12.0


def test_control_command_to_dict_preserves_protocol_fields():
    command = ControlCommand(
        robot_state="comforting",
        expression=ExpressionCommand(
            expression="gentle_care",
            intensity=1.7,
            duration_ms=1500,
            transition_ms=400,
            servo_params={"eye_open": 65},
        ),
        timestamp=1710000000.123,
    )

    payload = command.to_dict()

    assert payload["protocol_version"] == "1.0"
    assert payload["timestamp"] == 1710000000.123
    assert payload["cmd"] == "set_expression"
    assert payload["robot_state"] == "comforting"
    assert payload["expression"] == "gentle_care"
    assert payload["intensity"] == 1.0
    assert payload["servo_params"] == {"eye_open": 65}


def test_decision_and_emotion_results_have_stable_defaults():
    emotion = EmotionResult()
    decision = DecisionResult()

    assert emotion.face_emotion == "neutral"
    assert emotion.text_emotion == "neutral"
    assert emotion.user_state == "neutral"
    assert decision.robot_state == "listening"
    assert decision.expression == "neutral_listen"
