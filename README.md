# Face Expression Interaction

基于 OpenCV、MediaPipe Tasks 和规则式决策的仿生机器人表情交互 Demo。

项目主线：

```text
camera/video frame
-> face landmarks
-> head pose
-> facial features
-> face/text emotion
-> FSM decision
-> expression command
-> JSON control payload
```

## Environment

推荐使用项目外部的通用 Python 环境，避免把虚拟环境放进仓库。

当前开发环境示例：

```bash
/mnt/g/environment/python/venvs/common314/Scripts/python.exe -m pip install -r requirements.txt
```

## Model

MediaPipe FaceLandmarker 模型文件不提交到仓库。首次运行前下载到 `models/`：

```bash
/mnt/g/environment/python/venvs/common314/Scripts/python.exe scripts/download_models.py
```

## Run

摄像头模式：

```bash
/mnt/g/environment/python/venvs/common314/Scripts/python.exe main.py --source camera --camera-id 0
```

摄像头短跑验收：

```bash
/mnt/g/environment/python/venvs/common314/Scripts/python.exe main.py --source camera --camera-id 0 --max-frames 60
```

视频模式：

```bash
/mnt/g/environment/python/venvs/common314/Scripts/python.exe main.py --source video --video-path samples/smoke_no_face.mp4
```

运行时在控制台输入中文文本，例如 `开心`、`有点累`、`伤心`，主循环会读取最新文本并参与情绪融合。

退出方式：在 OpenCV 窗口按 `q` 或 `Esc`。

## Verify

```bash
/mnt/g/environment/python/venvs/common314/Scripts/python.exe -m pytest -q
/mnt/g/environment/python/venvs/common314/Scripts/python.exe -m ruff check .
```

## Notes

- `config/default.json` 管理输入源、摄像头、模型路径、平滑系数、FSM 防抖帧数和展示开关。
- `models/*.task`、视频样本和内部规划文档默认不进入 Git。
- V1 不引入 Qt、串口、ONNX 或硬件控制，先保持可运行的模块化业务闭环。
