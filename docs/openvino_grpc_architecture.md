# OpenVINO gRPC IP-Camera FER Architecture

## Target Flow

```text
Machine B
  phone/IP camera stream
        |
        v
Machine A
  stream ingest process
    thread 1: capture frames from IP camera
    thread 2: collect up to 16 frames and call gRPC model server
    thread 3: visualize annotated frames
        |
        v
  OpenVINO model server process
    gRPC over HTTP/2
    OpenVINO IR: best.xml + best.bin
```

There is no Docker and no TorchServe in the active flow.

## Modules

- `fer_realtime/model.py`: OpenVINO model wrapper, preprocessing, softmax/probability parsing.
- `fer_grpc/server.py`: local gRPC model server.
- `fer_grpc/client.py`: stream-ingest gRPC client.
- `fer_grpc/stream_ingest.py`: three-thread capture/infer/visualize pipeline.
- `fer_grpc/stream_ingest_server.py`: browser preview for the ingest output.

## Thread Design

1. Capture thread
   - Opens `IP_CAMERA_URL` with OpenCV `VideoCapture`.
   - Reconnects when the stream drops.
   - Pushes newest frames into a bounded queue.
   - Drops old frames when the queue is full to keep latency low.

2. Inference thread
   - Pulls up to `INGEST_BATCH_SIZE` frames, default `16`.
   - Waits at most `INGEST_BATCH_TIMEOUT_SECONDS`, default `0.25s`.
   - Sends one gRPC request with the frame batch.
   - Applies probability smoothing on returned probabilities.

3. Visualize thread
   - Draws per-face bounding boxes and expression labels.
   - Publishes latest annotated JPEG to `/snapshot.jpg` and MJPEG preview to `/mjpeg`.

## Batch Note

The current exported OpenVINO metadata says `batch: 1`. The gRPC request still groups up to 16 frames to reduce transport overhead. The model wrapper tries dynamic batch at load time; if the IR cannot accept dynamic batch, it safely falls back to per-frame inference inside the same request.

For true batch-16 inference, export the model with a dynamic or batch-16 input shape and point `--model` to that new OpenVINO folder.

## Run

Install:

```powershell
python -m pip install -r requirements-realtime.txt
```

Model server:

```powershell
python -m fer_grpc.server --model "models\fer_expression_yolo26n_openvino" --device AUTO --address 127.0.0.1:50051
```

Stream ingest:

```powershell
set IP_CAMERA_URL=http://192.168.1.50:8080/video
set MODEL_SERVER_ADDRESS=127.0.0.1:50051
set INGEST_BATCH_SIZE=16
python -m uvicorn fer_grpc.stream_ingest_server:app --host 127.0.0.1 --port 8080
```

Open:

```text
http://127.0.0.1:8080
```

## Configuration

| Variable | Default | Used By |
| --- | --- | --- |
| `IP_CAMERA_URL` | `0` | stream ingest |
| `MODEL_SERVER_ADDRESS` | `127.0.0.1:50051` | stream ingest |
| `INGEST_BATCH_SIZE` | `16` | stream ingest |
| `INGEST_BATCH_TIMEOUT_SECONDS` | `0.25` | stream ingest |
| `INGEST_FRAME_QUEUE_SIZE` | `64` | stream ingest |
| `INGEST_RESULT_QUEUE_SIZE` | `64` | stream ingest |
| `INGEST_AVERAGE_WINDOW` | `3` | stream ingest |
| `INGEST_FACE_CROP` | `true` | stream ingest/model server |
| `INGEST_SHOW_WINDOW` | `false` | stream ingest |

## Research Notes Used

- OpenVINO `Core` reads IR/ONNX/Paddle/TensorFlow/TFLite models and compiles them for a target device.
- OpenVINO automatic device selection is available through `AUTO`.
- gRPC is based on service methods and supports unary request/response calls over HTTP/2.
- gRPC uses Protocol Buffers by default, but other payload formats are possible. This lab uses JSON bytes to avoid generated files.
- OpenCV `VideoCapture` is suitable for local cameras and IP stream URLs.

Sources:

- https://docs.openvino.ai/2025/api/ie_python_api/_autosummary/openvino.Core.html
- https://docs.openvino.ai/2025/openvino-workflow/running-inference/inference-devices-and-modes/auto-device-selection.html
- https://grpc.io/docs/what-is-grpc/core-concepts/
- https://grpc.io/docs/languages/python/basics/
- https://docs.opencv.org/4.x/d8/dfe/classcv_1_1VideoCapture.html
