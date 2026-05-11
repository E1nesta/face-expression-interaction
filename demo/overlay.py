from __future__ import annotations

import cv2

from core.data_structures import (
    DecisionResult,
    EmotionResult,
    ExpressionCommand,
    FaceLandmarkResult,
)


TEXT_COLOR = (0, 255, 0)
POINT_COLOR = (0, 200, 255)


def draw_overlay(
    frame,
    face: FaceLandmarkResult,
    emotion: EmotionResult,
    decision: DecisionResult,
    expression: ExpressionCommand,
    fps: float,
    draw_landmarks: bool = True,
    draw_fps: bool = True,
    draw_status: bool = True,
):
    if draw_landmarks and face.face_detected:
        _draw_landmarks(frame, face)

    lines = []
    if draw_fps:
        lines.append(f"FPS: {fps:.1f}")
    if draw_status:
        lines.extend(
            [
                f"user: {emotion.user_state}",
                f"face: {emotion.face_emotion}",
                f"text: {emotion.text_emotion}",
                f"robot: {decision.robot_state}",
                f"expr: {expression.expression}",
            ]
        )
    for index, text in enumerate(lines):
        cv2.putText(
            frame,
            text,
            (10, 24 + index * 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            TEXT_COLOR,
            2,
            cv2.LINE_AA,
        )
    return frame


def _draw_landmarks(frame, face: FaceLandmarkResult) -> None:
    step = max(1, len(face.landmarks) // 80)
    for x, y, _z in face.landmarks[::step]:
        cv2.circle(
            frame,
            (int(x * face.image_width), int(y * face.image_height)),
            1,
            POINT_COLOR,
            -1,
        )
