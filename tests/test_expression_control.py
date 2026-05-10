import json

from control.output import ConsoleJsonOutput
from control.protocol import build_control_command, build_control_payload
from core.data_structures import DecisionResult, ExpressionCommand
from expression.expression_generator import generate_expression_command


def test_generate_expression_command_uses_expression_profile_defaults():
    command = generate_expression_command(
        DecisionResult(
            robot_state="comforting",
            expression="gentle_care",
            intensity=0.45,
        ),
        default_duration_ms=1500,
        default_transition_ms=400,
    )

    assert command.expression == "gentle_care"
    assert command.intensity == 0.45
    assert command.duration_ms == 1500
    assert command.transition_ms == 400
    assert command.servo_params["eye_open"] == 65
    assert command.servo_params["head_pitch"] == -8


def test_generate_expression_command_clamps_intensity():
    command = generate_expression_command(
        DecisionResult(
            robot_state="happy_reply",
            expression="happy_smile",
            intensity=2.0,
        )
    )

    assert command.intensity == 1.0


def test_build_control_payload_is_json_serializable():
    command = build_control_command(
        DecisionResult(robot_state="comforting", expression="gentle_care", intensity=0.45),
        ExpressionCommand(
            expression="gentle_care",
            intensity=0.45,
            duration_ms=1500,
            transition_ms=400,
            servo_params={"eye_open": 65},
        ),
        timestamp=1710000000.123,
    )

    payload = build_control_payload(command)

    assert payload["protocol_version"] == "1.0"
    assert payload["timestamp"] == 1710000000.123
    assert payload["cmd"] == "set_expression"
    assert payload["robot_state"] == "comforting"
    assert payload["expression"] == "gentle_care"
    assert payload["servo_params"] == {"eye_open": 65}
    assert json.loads(json.dumps(payload)) == payload


def test_console_json_output_writes_one_json_line():
    lines = []
    output = ConsoleJsonOutput(write_line=lines.append)

    output.send(
        build_control_command(
            DecisionResult(
                robot_state="listening",
                expression="neutral_listen",
                intensity=0.35,
            ),
            ExpressionCommand(
                expression="neutral_listen",
                intensity=0.35,
                duration_ms=1500,
                transition_ms=400,
                servo_params={},
            ),
            timestamp=1.5,
        )
    )

    assert len(lines) == 1
    assert json.loads(lines[0])["robot_state"] == "listening"
