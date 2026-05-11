# Face Expression Interaction

基于 OpenCV、MediaPipe Tasks、ONNXRuntime 和 FSM 的仿生机器人表情交互控制 Demo。

项目主线：

```text
camera/video frame
-> face landmarks
-> head pose
-> facial features
-> ONNX/rule face emotion
-> emotion stabilizer
-> text emotion fusion
-> FSM decision
-> expression command
-> JSON control payload
```

## Environment

推荐使用项目外部的通用 Python 环境，避免把虚拟环境放进仓库。

Windows 一键部署：

```bat
deploy.bat
```

Linux / macOS 一键部署：

```bash
./deploy.sh
```

跨平台 Python 入口：

```bash
python scripts/bootstrap.py
```

如果使用已有环境，也可以手动安装：

```bash
python -m pip install -r requirements.txt
python -m pip install -r requirements-model.txt
```

## Model

仓库已随代码提供默认 Demo 模型：

```text
models/face_landmarker.task
models/emotion.onnx
```

模型缺失或需要刷新时，重新下载到 `models/`：

```bash
python scripts/download_models.py
```

ONNX 推理依赖由 `requirements-model.txt` 提供。PyTorch 仅用于后续训练、导出或实验，不是 Demo 运行必需项：

```bash
python -m pip install -r requirements-torch.txt
```

当前 ONNX backend 默认读取：

```text
models/emotion.onnx
```

该文件随仓库提供，下载脚本默认使用 EmotiEffLib 的 8 类 ONNX 模型。

## Run

Windows 一键运行摄像头 Demo：

```bat
run.bat
```

Windows 一键运行视频 Demo：

```bat
run_video.bat
```

Linux / macOS 一键运行摄像头 Demo：

```bash
./run.sh
```

摄像头模式：

```bash
python main.py --source camera --camera-id 0
```

摄像头短跑验收：

```bash
python main.py --source camera --camera-id 0 --max-frames 60
```

视频模式：

```bash
python main.py --source video --video-path samples/smoke_no_face.mp4
```

通用运行脚本也支持参数透传：

```bash
python scripts/run_demo.py --source camera --camera-id 0 --max-frames 60
python scripts/run_demo.py --source video --video-path samples/smoke_no_face.mp4
```

运行时在控制台输入中文文本，例如 `开心`、`有点累`、`伤心`，主循环会读取最新文本并参与情绪融合。

退出方式：在 OpenCV 窗口按 `q` 或 `Esc`。

## Emotion Backend

默认使用 ONNX 模型视觉情绪识别：

```json
{
  "emotion": {
    "face_backend": "onnx"
  }
}
```

模型输出会先经过情绪稳定层，避免单帧误判直接驱动机器人状态：

```json
{
  "emotion": {
    "stabilizer": {
      "enabled": true,
      "window_size": 7,
      "min_votes": 4,
      "min_confidence": 0.45,
      "reset_on_no_face": true
    }
  }
}
```

如需回退到规则版，可切换为：

```json
{
  "emotion": {
    "face_backend": "rule"
  }
}
```

ONNX backend 示例：

```json
{
  "emotion": {
    "face_backend": "onnx",
    "fallback_to_rule": true,
    "onnx": {
      "input_type": "face_image",
      "model_path": "models/emotion.onnx",
      "image_size": [224, 224],
      "color_mode": "rgb",
      "labels": ["neutral", "happiness", "surprise", "sadness", "anger", "disgust", "fear", "contempt"]
    },
    "label_mapping": {
      "happiness": "happy",
      "sadness": "sad",
      "anger": "angry"
    }
  }
}
```

`onnx.input_type` 支持 `features` 和 `face_image`。`face_image` 适合 FERPlus、EmotiEff 等开源人脸图像情绪模型；`label_mapping` 用于把模型原始标签映射成机器人用户状态。`fallback_to_rule` 为 `true` 时，模型文件缺失或 ONNXRuntime 不可用会回退到规则版。默认 Demo 模型进入 Git，额外实验模型不进入 Git。

控制台 JSON 会同时输出当前帧和稳定后的结果：

```text
raw_face_emotion
raw_face_confidence
stable_face_emotion
user_state
robot_state
```

## Verify

```bash
python -m pytest -q
python -m ruff check .
```

## Notes

- `config/default.json` 管理输入源、摄像头、模型路径、平滑系数、FSM 防抖帧数和展示开关。
- 默认 Demo 模型进入 Git；额外模型、视频样本和内部规划文档默认不进入 Git。
- 当前版本不引入 Qt、串口或真实硬件控制，先保持可运行的模型推理和机器人表情控制闭环。
