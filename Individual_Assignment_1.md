# Individual Assignment #1

## Realtime Multi-Face Facial Expression Recognition

This assignment implements a realtime computer vision system that detects human faces from a webcam or IP camera and recognizes the visible facial expression for each detected face. The system is designed as a lightweight production-style inference pipeline using OpenVINO, gRPC, OpenCV, and a browser-based monitoring UI.

The current implementation focuses on visible facial expression recognition, not psychological diagnosis and not true person identity recognition. The output should be interpreted as the expression observed in the current camera frame.

## Problem Description

Realtime face analysis is useful in many interactive AI applications, such as classroom engagement demos, smart kiosks, user experience monitoring, and human-computer interaction. A practical demo system must be able to:

- Capture frames from a laptop webcam or IP camera.
- Detect one or more faces in each frame.
- Crop each face region separately.
- Recognize the visible expression for every detected face.
- Display bounding boxes, labels, confidence scores, and top predictions in realtime.
- Keep latency low enough for live camera demonstration.

The main challenge is that camera capture, face detection, model inference, and visualization have different computational profiles. To make the application easier to maintain and closer to a production design, the project separates stream ingestion from model serving.

## Solutions

The proposed solution is a realtime multi-face facial-expression recognition system with the following components:

| Component | Role |
| --- | --- |
| OpenCV camera capture | Reads frames from webcam or IP camera URL. |
| FaceCropper | Detects faces and crops square face regions. |
| YOLO26n classification model | Classifies each cropped face into expression classes. |
| OpenVINO runtime | Runs optimized local inference on CPU/GPU/NPU-capable hardware. |
| gRPC model server | Serves model inference through a separate process. |
| Stream ingest pipeline | Captures frames, batches them, calls gRPC, and publishes UI state. |
| FastAPI browser UI | Displays MJPEG stream, face cards, pause/resume, save snapshot, and history. |
| SQLite history | Stores manual recognition snapshots for review. |
| Lightweight face tracker | Keeps stable `Face #ID` labels across nearby frames for demo readability. |

The system supports multiple faces by applying a detect-many and classify-each-crop pipeline:

```text
camera frame
  -> detect all faces
  -> crop each face
  -> classify each crop with OpenVINO
  -> assign lightweight tracking IDs
  -> draw bbox + expression label
  -> publish browser UI state
```

## Data Collection

The training data is based on a Kaggle facial-expression dataset that contains images with YOLO-format face bounding-box labels. The original detection annotations are used to crop face regions and convert the dataset into a classification-ready folder structure.

Raw dataset location expected by the notebook:

```text
data/kaggle_fer_yolo_raw/9 Facial Expressions you need
```

Prepared classification dataset:

```text
data/fer_expression_yolo_prepared/
  train/
  val/
  test/
```

Expression classes:

| Class ID | Class Name |
| --- | --- |
| 0 | angry |
| 1 | contempt |
| 2 | disgust |
| 3 | fear |
| 4 | happy |
| 5 | neutral |
| 6 | sad |
| 7 | sleepy |
| 8 | surprise |

Prepared dataset summary:

| Split | Number of Images |
| --- | ---: |
| Train | 106,656 |
| Validation | 1,720 |
| Test | 1,700 |

Preprocessing steps:

1. Load image files and YOLO label files.
2. Convert normalized YOLO boxes to pixel coordinates.
3. Crop face boxes from source images.
4. Save cropped faces into class folders.
5. Train the classifier on the prepared face-crop dataset.

## Finetune YOLO26 Model

The training notebook is:

```text
train_kaggle_fer_openvino_run_all.ipynb
```

The model starts from an Ultralytics YOLO26 classification checkpoint:

```text
yolo26n-cls.pt
```

Training configuration:

| Item | Value |
| --- | --- |
| Task | Image classification |
| Base model | YOLO26n-cls |
| Image size | 224 x 224 |
| Batch size | 16 |
| Device | CPU in the current recorded run |
| Frozen layers | 8 |
| Optimizer | Ultralytics auto optimizer |
| Epoch target | 20 |

The notebook performs:

1. Environment and dependency validation.
2. Dataset preparation from YOLO face annotations.
3. YOLO26n classification fine-tuning.
4. Validation on the prepared dataset.
5. Export to OpenVINO IR.
6. Publishing the exported model into the runtime model folder.
7. Runtime verification with the OpenVINO classifier wrapper.

Recorded training result:

| Metric | Value |
| --- | ---: |
| Best recorded top-1 accuracy | 79.42% |
| Best recorded top-5 accuracy | 98.90% |
| Final recorded validation loss | 0.60156 |

Exported runtime model:

```text
models/fer_expression_yolo26n_openvino/
  best.xml
  best.bin
  metadata.yaml
```

## Inference System Architecture

The realtime inference system is split into two main services.

```text
Camera source
  webcam: IP_CAMERA_URL=0
  IP camera: IP_CAMERA_URL=http://<phone-ip>:<port>/video
        |
        v
Stream ingest service
  thread 1: capture camera frames
  thread 2: batch frames and call gRPC model server
  thread 3: annotate frames and publish browser state
        |
        v
OpenVINO model serving service
  gRPC over HTTP/2
  model: models/fer_expression_yolo26n_openvino/best.xml
        |
        v
Browser UI
  MJPEG preview
  per-face expression cards
  pause/resume
  save snapshot
  history
```

Key modules:

| Module | Description |
| --- | --- |
| `fer_realtime/model.py` | OpenVINO model wrapper and OpenCV face cropper. |
| `fer_realtime/analyzer.py` | Realtime state, overlay drawing, and expression result model. |
| `fer_realtime/tracking.py` | Lightweight IoU-based face tracking IDs. |
| `fer_realtime/history.py` | SQLite snapshot persistence. |
| `fer_grpc/server.py` | OpenVINO gRPC inference server. |
| `fer_grpc/client.py` | gRPC inference client. |
| `fer_grpc/stream_ingest.py` | Three-thread capture, inference, and visualization pipeline. |
| `fer_grpc/stream_ingest_server.py` | FastAPI browser UI and MJPEG endpoints. |

Browser endpoints:

| Endpoint | Purpose |
| --- | --- |
| `/` | Browser dashboard |
| `/mjpeg` | Realtime annotated camera stream |
| `/state` | Latest recognition state JSON |
| `/snapshot.jpg` | Latest annotated frame |
| `/save` | Save current recognition state |
| `/history` | Recent saved snapshots |
| `/health` | Pipeline health check |

## Hardware Acceleration

The project uses OpenVINO for hardware-aware inference. OpenVINO can compile and execute models on supported Intel CPU, integrated GPU, and NPU devices. The runtime can be selected through the model server argument:

```powershell
python -m fer_grpc.server --model "models\fer_expression_yolo26n_openvino" --device CPU --address 127.0.0.1:50051
```

For automatic device selection:

```powershell
python -m fer_grpc.server --model "models\fer_expression_yolo26n_openvino" --device AUTO --address 127.0.0.1:50051
```

Current practical demo recommendation:

| Setting | Recommended Value | Reason |
| --- | --- | --- |
| `--device` | `CPU` | Stable on most laptops. |
| `FACE_DETECTOR_MODE` | `fast` | Lower CPU load for live demo. |
| `INGEST_BATCH_SIZE` | `1` or `4` | `1` lowers latency; `4` improves throughput. |
| `INGEST_BATCH_TIMEOUT_SECONDS` | `0.03` to `0.05` | Prevents waiting too long for a batch. |

Face detector modes:

| Mode | Behavior | Trade-off |
| --- | --- | --- |
| `fast` | Uses main frontal Haar cascade only. | Best for low-latency demo. |
| `balanced` | Uses frontal default and alt cascade. | Better recall, slightly slower. |
| `robust` | Adds profile-face fallback. | Better for side faces, highest CPU cost. |

## Model Compression / Batch Processing

The model is exported from PyTorch/Ultralytics into OpenVINO IR:

```text
best.xml
best.bin
```

This removes PyTorch from the realtime inference path and allows OpenVINO to optimize model execution for the selected device.

Current model export status:

| Feature | Current Status |
| --- | --- |
| OpenVINO IR export | Implemented |
| Runtime PyTorch dependency | Removed from inference path |
| FP16 compression | Not enabled in current exported model |
| INT8 quantization | Not enabled in current exported model |
| Dynamic model batch | Attempted by wrapper, falls back safely if unsupported |
| gRPC request batching | Implemented |

Batch processing is implemented at the stream-ingest and gRPC request level. The ingest service collects up to `INGEST_BATCH_SIZE` frames, sends them to the model server in one gRPC request, and receives a result list. This reduces transport overhead. If the OpenVINO model cannot run true dynamic batch, the server still processes the frames safely inside the same request.

For lowest camera lag during demo:

```powershell
$env:INGEST_BATCH_SIZE="1"
$env:INGEST_BATCH_TIMEOUT_SECONDS="0.03"
$env:FACE_DETECTOR_MODE="fast"
```

For higher throughput with acceptable latency:

```powershell
$env:INGEST_BATCH_SIZE="4"
$env:INGEST_BATCH_TIMEOUT_SECONDS="0.05"
$env:FACE_DETECTOR_MODE="fast"
```

## Performance Testing

The project includes automated unit tests for core realtime and gRPC behavior:

```powershell
python -m unittest discover -s tests
```

Current test result:

```text
Ran 15 tests
OK
```

Covered test areas:

| Test Area | Description |
| --- | --- |
| OpenVINO path validation | Ensures runtime model path accepts OpenVINO XML and rejects `.pt` for realtime runtime. |
| Probability smoothing | Verifies rolling average of prediction vectors. |
| Face tracking | Verifies stable ID assignment for overlapping face boxes. |
| SQLite history | Verifies saving and fetching recognition snapshots. |
| gRPC payload | Verifies JSON payload roundtrip over the lightweight protocol helper. |
| Stream ingest config | Verifies local webcam index and default batch settings. |
| Browser state payload | Verifies per-face results and tracking IDs in `/state` payload. |

Runtime validation endpoints:

| Endpoint | Use |
| --- | --- |
| `/health` | Checks whether the ingest pipeline is running. |
| `/state` | Shows frame ID, status, face count, labels, confidence, and latency. |
| `/snapshot.jpg` | Confirms the latest annotated frame is produced. |

Suggested benchmark procedure:

1. Start the gRPC model server.
2. Start the stream ingest UI.
3. Test `INGEST_BATCH_SIZE=1`, `4`, `8`, and `16`.
4. Record observed latency from `/state`.
5. Record whether the UI remains smooth.
6. Compare `FACE_DETECTOR_MODE=fast`, `balanced`, and `robust`.

Example low-latency run commands:

Terminal 1:

```powershell
cd "C:\Users\DELL\Desktop\Vinh Hoang\Master Program\AI trong sản xuất DevOps, DataOps, MLOps\Lab1&2"
$env:FACE_DETECTOR_MODE="fast"
python -m fer_grpc.server --model "models\fer_expression_yolo26n_openvino" --device CPU --address 127.0.0.1:50051
```

Terminal 2:

```powershell
cd "C:\Users\DELL\Desktop\Vinh Hoang\Master Program\AI trong sản xuất DevOps, DataOps, MLOps\Lab1&2"
$env:IP_CAMERA_URL="0"
$env:MODEL_SERVER_ADDRESS="127.0.0.1:50051"
$env:MODEL_SERVER_TIMEOUT_SECONDS="90"
$env:INGEST_BATCH_SIZE="1"
$env:INGEST_BATCH_TIMEOUT_SECONDS="0.03"
$env:INGEST_FRAME_QUEUE_SIZE="4"
$env:INGEST_RESULT_QUEUE_SIZE="4"
$env:FACE_DETECTOR_MODE="fast"
python -m uvicorn fer_grpc.stream_ingest_server:app --host 127.0.0.1 --port 8080
```

Open:

```text
http://127.0.0.1:8080
```

## Conclusion

This assignment demonstrates a complete realtime face-expression recognition pipeline with model fine-tuning, OpenVINO export, gRPC model serving, camera stream ingestion, multi-face recognition, browser visualization, snapshot history, and lightweight face tracking IDs. The system is simple enough to run on a laptop CPU while still following a production-inspired separation between stream ingestion and inference serving.
