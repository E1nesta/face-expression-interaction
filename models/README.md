# Model Assets

This repository ships the small demo models required for the default experience:

```text
face_landmarker.task  MediaPipe FaceLandmarker task bundle
emotion.onnx          EmotiEffLib AffectNet 8-class ONNX model
```

They are included so a fresh clone can run the camera/video demo without a separate model download step.

If the files are missing or need to be refreshed:

```bash
python scripts/download_models.py
```

Sources:

- MediaPipe FaceLandmarker model: https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task
- EmotiEffLib ONNX model: https://github.com/sb-ai-lab/EmotiEffLib/raw/main/models/affectnet_emotions/onnx/enet_b0_8_best_vgaf.onnx

Licensing notes:

- EmotiEffLib is published under Apache-2.0 and its documentation states there is no limitation for academic or commercial usage.
- MediaPipe model usage should follow the Google AI Edge / MediaPipe model and documentation terms.
