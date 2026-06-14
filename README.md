# Lab1 - Realtime Multi-Face Facial Expression Recognition

## Quick Setup

Di chuyen vao thu muc project:

```powershell
cd "<duong-dan-toi-Lab1>"
```

Vi du tren may hien tai:

```powershell
cd "C:\Users\DELL\Desktop\Vinh Hoang\Master Program\AI trong sản xuất DevOps, DataOps, MLOps\Lab1"
```

Chay model server OpenVINO gRPC:

```powershell
python -m pip install -r requirements-realtime.txt
python -m fer_grpc.server --model "models\fer_expression_yolo26n_openvino" --device AUTO --address 127.0.0.1:50051
```

Mo terminal thu hai va chay stream ingest + browser UI bang camera laptop:

```powershell
cd "<duong-dan-toi-Lab1>"
set IP_CAMERA_URL=0
set MODEL_SERVER_ADDRESS=127.0.0.1:50051
set MODEL_SERVER_TIMEOUT_SECONDS=90
set INGEST_BATCH_SIZE=4
python -m uvicorn fer_grpc.stream_ingest_server:app --host 127.0.0.1 --port 8080
```

Mo UI:

```text
http://127.0.0.1:8080
```

Neu muon xem UI tu dien thoai cung Wi-Fi:

```powershell
python -m uvicorn fer_grpc.stream_ingest_server:app --host 0.0.0.0 --port 8080
```

Sau do mo tren dien thoai:

```text
http://<IP-laptop>:8080
```

## Project Purpose

Project nay la he thong realtime facial-expression recognition. He thong chi tap trung vao detect khuon mat va nhan dien bieu cam nhin thay tren tung khuon mat.

Output cua model chi nen hieu la visible facial expression tren camera frame, khong phai chan doan cam xuc noi tam hay ket luan tam ly.

## What The System Can Do

- Lay frame realtime tu webcam laptop hoac IP camera URL.
- Detect mot hoac nhieu khuon mat trong cung mot frame.
- Crop tung khuon mat rieng.
- Classify expression cua tung face crop bang OpenVINO.
- Ve bounding box va label ngay gan khuon mat tuong ung.
- Hien ket qua moi face trong browser UI.
- Co nut `Dung lai` / `Chay tiep` de doc ket qua ro hon.
- Co nut `Luu snapshot` de luu ket qua hien tai vao SQLite.
- Tach model serving va stream ingest thanh hai process rieng qua gRPC HTTP/2.

## Multi-Face Logic

Model expression classifier duoc train cho tung face crop. Vi vay model khong tu no xu ly truc tiep anh co 2 nguoi.

Multi-face support duoc lam bang runtime pipeline:

```text
camera frame
  -> detect all faces
  -> crop face #1, face #2, ...
  -> classify each crop
  -> draw each label beside its own face
  -> return faces[] to UI
```

Neu camera nhin thay 2 khuon mat va face detector bat duoc ca 2, UI se hien 2 box va 2 label rieng.

## Dataset Note

Dataset Kaggle dang dung co anh kem YOLO bbox/label cho facial expressions. Dataset nay phu hop de:

- doc bbox khuon mat.
- crop khuon mat thanh classification dataset.
- train/fine-tune classifier bieu cam.
- export model sang OpenVINO.

Dataset khong co nghia la model classification tu dong nhan dien nhieu nguoi trong mot anh. Phan nhieu nguoi nam o runtime face detection + per-face classification.

## Architecture

```text
Camera source
  webcam laptop: IP_CAMERA_URL=0
  phone/IP camera: IP_CAMERA_URL=http://<phone-ip>:<port>/video
        |
        v
Stream ingest service
  thread 1: capture frames
  thread 2: batch frames and call gRPC model server
  thread 3: annotate frames and publish UI state
        |
        v
OpenVINO model serving service
  gRPC over HTTP/2
  model: models/fer_expression_yolo26n_openvino/best.xml
        |
        v
Browser UI
  annotated camera stream
  per-face recognition cards
  pause/save/history controls
```

Hien tai project khong dung Docker, khong dung TorchServe, va khong load `.pt` trong realtime runtime. File `.pt` chi thuoc training/export pipeline.

## Repository Structure

```text
fer_realtime/
  config.py       project paths, class names, default model path
  model.py        OpenVINO classifier and OpenCV face cropper
  analyzer.py     realtime state and OpenCV overlay
  smoothing.py    frame sampling and probability averaging helpers
  history.py      SQLite recognition snapshot storage

fer_grpc/
  server.py               OpenVINO gRPC model server
  client.py               gRPC inference client
  codec.py                JPEG/base64 frame encoding helpers
  protocol.py             JSON bytes protocol for gRPC generic handler
  stream_ingest.py        3-thread capture/infer/visualize pipeline
  stream_ingest_server.py FastAPI browser UI and MJPEG/state endpoints

models/
  fer_expression_yolo26n_openvino/
    best.xml
    best.bin
    metadata.yaml

tests/
  test_realtime_core.py
  test_grpc_core.py

train_kaggle_fer_openvino_run_all.ipynb
requirements-realtime.txt
requirements-train.txt
```

## Runtime Model

Realtime app dung OpenVINO IR:

```text
models\fer_expression_yolo26n_openvino
```

Folder model gom:

```text
best.xml
best.bin
metadata.yaml
```

Classes hien tai:

```text
angry
contempt
disgust
fear
happy
neutral
sad
sleepy
surprise
```

## Browser Endpoints

| Endpoint | Purpose |
| --- | --- |
| `/` | Browser UI |
| `/mjpeg` | MJPEG annotated camera stream |
| `/snapshot.jpg` | Latest annotated JPEG |
| `/state` | Latest recognition state JSON |
| `/save` | Save current state to SQLite |
| `/history` | Recent saved snapshots |
| `/health` | Pipeline health/status |

## SQLite History

Snapshots duoc luu tai:

```text
data\fer_recognition_history.sqlite3
```

Bang chinh:

```text
recognition_snapshots
```

Moi snapshot luu:

- timestamp.
- source.
- status.
- label/confidence chinh.
- face_count.
- latency/device.
- top emotions JSON.
- faces JSON, gom label, confidence, top-k va bbox cua tung face.
- probabilities JSON.

Thu muc `data/` nam trong `.gitignore`, nen database local khong bi push len GitHub.

## Phone Camera

Neu muon dung camera dien thoai lam camera source, dien thoai can app IP camera rieng.

Flow:

```text
Phone IP camera app
  URL: http://<phone-ip>:<port>/video
        |
        v
Laptop stream ingest
  IP_CAMERA_URL=http://<phone-ip>:<port>/video
        |
        v
Laptop browser UI
  http://127.0.0.1:8080
```

Luu y:

- IP dien thoai la camera source.
- IP laptop la UI/server source.
- `127.0.0.1` chi dung tren chinh may dang chay service.

## Training

Notebook training chinh:

```text
train_kaggle_fer_openvino_run_all.ipynb
```

Dataset raw nen dat tai:

```text
data\kaggle_fer_yolo_raw\9 Facial Expressions you need
```

Notebook thuc hien:

1. Kiem tra raw dataset.
2. Xoa output train/export cu.
3. Doc YOLO bbox labels.
4. Crop face boxes thanh classification dataset.
5. Train/fine-tune Ultralytics YOLO classification model.
6. Validate model.
7. Export best checkpoint sang OpenVINO.
8. Publish model vao `models\fer_expression_yolo26n_openvino`.
9. Verify OpenVINO runtime load duoc model.

## Environment Variables

| Variable | Default | Meaning |
| --- | --- | --- |
| `IP_CAMERA_URL` | `0` | Camera source. `0` la webcam laptop. URL la IP camera/phone. |
| `MODEL_SERVER_ADDRESS` | `127.0.0.1:50051` | gRPC model server address. |
| `MODEL_SERVER_TIMEOUT_SECONDS` | `90` | Timeout cho inference request. |
| `INGEST_BATCH_SIZE` | `16` | So frame toi da trong mot inference request. |
| `INGEST_BATCH_TIMEOUT_SECONDS` | `0.25` | Thoi gian doi gom batch. |
| `INGEST_FRAME_QUEUE_SIZE` | `64` | Queue size cho captured frames. |
| `INGEST_RESULT_QUEUE_SIZE` | `64` | Queue size cho inference results. |
| `INGEST_AVERAGE_WINDOW` | `3` | Fallback smoothing window khi chi co single-result path. |
| `INGEST_FACE_CROP` | `true` | Bat/tat face crop trong model server. |
| `INGEST_SHOW_WINDOW` | `false` | Mo OpenCV local window neu can. |

Voi laptop CPU, nen bat dau bang:

```powershell
set INGEST_BATCH_SIZE=4
```

Batch lon hon co the tang throughput, nhung cung co the lam latency cao hoac timeout neu CPU cham.

## Tests

Chay tests:

```powershell
python -m unittest discover -s tests
```

Tests cover:

- OpenVINO model path validation.
- Probability smoothing helpers.
- SQLite recognition snapshot save/fetch.
- gRPC JSON payload roundtrip.
- Stream ingest config.
- `/save` endpoint.
- Per-face state payload.

## Current Limitations

- Face detector hien tai la OpenCV Haar Cascade, nhe nhung co the miss face nghieng, face nho, bi che, hoac anh sang xau.
- Chua co face tracking ID theo thoi gian. `Face #1` va `Face #2` la thu tu trong frame hien tai.
- OpenVINO model co the dang batch 1; wrapper se thu dynamic batch va fallback per-frame neu can.
- gRPC local hien tai chua co TLS.
- gRPC protocol dung JSON bytes de don gian, chua dung generated protobuf schema.

## Possible Improvements

- Doi face detector sang MediaPipe, RetinaFace, hoac YOLO face detector.
- Them face tracking de giu ID on dinh cho tung nguoi.
- Export OpenVINO dynamic batch hoac batch 16 that su.
- Dung protobuf schema chuan cho gRPC.
- Them TLS neu chay qua network that.
- Them dashboard FPS, latency, queue size, dropped frames.
- Them model registry/versioning de quan ly nhieu ban model.
