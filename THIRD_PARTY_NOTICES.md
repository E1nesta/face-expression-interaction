# Third-Party Notices

This project uses open-source runtime libraries and demo model assets.

## MediaPipe FaceLandmarker

- Source: https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker
- Model download: https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task
- Notes: used for face landmark detection in camera/video frames.

## EmotiEffLib ONNX Emotion Model

- Source: https://github.com/sb-ai-lab/EmotiEffLib
- Model download: https://github.com/sb-ai-lab/EmotiEffLib/raw/main/models/affectnet_emotions/onnx/enet_b0_8_best_vgaf.onnx
- License: Apache-2.0 according to the upstream project documentation.
- Notes: used as the default ONNX visual emotion backend.

If the model is used in reports or publications, cite the upstream EmotiEffLib papers listed in the upstream README.
