# Lab1 - Realtime Multi-Face Facial Expression Recognition

## Quick Setup

Terminal 1 - chay OpenVINO gRPC model server:

```powershell
cd "C:\Users\DELL\Desktop\Vinh Hoang\Master Program\AI trong san xuat DevOps, DataOps, MLOps\Lab1"
python -m pip install -r requirements-realtime.txt
python -m fer_grpc.server --model "models\fer_expression_yolo26n_openvino" --device AUTO --address 127.0.0.1:50051
```

Terminal 2 - chay stream ingest + browser UI bang camera laptop:

```powershell
cd "C:\Users\DELL\Desktop\Vinh Hoang\Master Program\AI trong san xuat DevOps, DataOps, MLOps\Lab1"
set IP_CAMERA_URL=0
set MODEL_SERVER_ADDRESS=127.0.0.1:50051
set MODEL_SERVER_TIMEOUT_SECONDS=90
set INGEST_BATCH_SIZE=4
python -m uvicorn fer_grpc.stream_ingest_server:app --host 127.0.0.1 --port 8080
```

Mo UI tren may laptop:

```text
http://127.0.0.1:8080
```

Neu muon xem UI tu dien thoai cung Wi-Fi, chay `uvicorn` voi `--host 0.0.0.0` va mo `http://<IP-laptop>:8080`.

## Project Summary

Du an nay la he thong realtime facial-expression recognition. Muc tieu hien tai chi tap trung vao:

- Detect mot hoac nhieu khuon mat trong camera frame.
- Crop tung khuon mat.
- Classify bieu cam cua tung crop bang OpenVINO model.
- Ve bounding box va label ngay tai khuon mat tuong ung.
- Hien danh sach ket qua tung face trong browser UI.
- Cho phep dung realtime panel va luu snapshot vao SQLite de xem lai.

Ket qua chi la nhan dien bieu cam nhin thay tren khuon mat, khong phai chan doan cam xuc noi tam hay de xuat cach phan hoi hoi thoai.

## Current Capabilities

- Nhan input tu webcam laptop bang `IP_CAMERA_URL=0`.
- Nhan input tu phone/IP camera URL, vi du `http://<phone-ip>:8080/video`.
- Detect nhieu khuon mat trong cung mot frame bang OpenCV Haar Cascade.
- Classify tung face crop bang OpenVINO IR model.
- Tra ve `faces[]` cho moi frame, moi face co label, confidence, top-k va bbox.
- Ve label ngay tren/gan bounding box cua tung khuon mat.
- Browser UI co hai vung: camera realtime va recognition results.
- Co nut `Dung lai` / `Chay tiep` de doc ket qua ro hon.
- Co nut `Luu snapshot` de ghi ket qua hien tai vao SQLite.
- Co gRPC model server rieng, giao tiep HTTP/2 local.
- Co stream ingest 3 thread: capture, inference batch, visualize.

## Important Dataset Note

Dataset Kaggle dang dung la dataset co anh va YOLO bbox/label cho facial expressions. No ho tro viec tao face crop va train classifier bieu cam.

Model classifier sau train khong tu no xu ly "2 nguoi" trong anh. Multi-face support duoc lam o runtime bang pipeline:

```text
full camera frame
  -> detect all faces
  -> crop face #1, face #2, ...
  -> classify each crop
  -> draw each label beside its own face
```

Vi vay, neu camera thay 2 khuon mat va face detector bat duoc ca 2, he thong co the nhan dien va hien 2 label rieng.

## Architecture

```text
Camera source
  webcam laptop: IP_CAMERA_URL=0
  phone/IP camera: IP_CAMERA_URL=http://<phone-ip>:<port>/video
        |
        v
Stream ingest service
  thread 1: capture frame tu camera/IP stream
  thread 2: gom batch frame va goi gRPC model server
  thread 3: annotate frame va publish MJPEG/UI state
        |
        v
OpenVINO model serving service
  gRPC over HTTP/2
  model: models/fer_expression_yolo26n_openvino/best.xml
        |
        v
Browser UI
  realtime annotated camera
  per-face recognition cards
  pause/save/history controls
```

Hien tai he thong khong dung Docker, khong dung TorchServe, va khong load PyTorch `.pt` trong runtime. Cac service chay truc tiep bang Python/Anaconda tren may local.

## Main Components

`fer_realtime/`

- `config.py`: project paths, OpenVINO model path, SQLite path, class names.
- `model.py`: OpenVINO model wrapper, preprocessing, probability parsing, Haar face cropper.
- `analyzer.py`: realtime state, per-face result dataclass, OpenCV overlay.
- `smoothing.py`: frame sampling va probability averaging helpers.
- `history.py`: SQLite persistence cho recognition snapshots.

`fer_grpc/`

- `server.py`: OpenVINO gRPC model server, detect all faces va infer tung crop.
- `client.py`: gRPC client cho stream ingest.
- `codec.py`: JPEG/base64 encode-decode.
- `protocol.py`: JSON bytes protocol cho gRPC generic handler.
- `stream_ingest.py`: 3-thread capture/infer/visualize pipeline.
- `stream_ingest_server.py`: FastAPI browser UI, MJPEG stream, state API, save/history endpoints.

Training:

- `train_kaggle_fer_openvino_run_all.ipynb`: notebook training/export chinh.

Runtime model:

- `models/fer_expression_yolo26n_openvino/best.xml`
- `models/fer_expression_yolo26n_openvino/best.bin`
- `models/fer_expression_yolo26n_openvino/metadata.yaml`

## Model

Runtime model la OpenVINO IR, khong phai `.pt`:

```text
models\fer_expression_yolo26n_openvino
```

Class hien tai:

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

Ultralytics/PyTorch chi duoc dung trong training/export. Khi app realtime chay, `OpenVINOExpressionClassifier` load `.xml/.bin` va infer bang OpenVINO.

## Realtime Inference Logic

Moi frame realtime di qua cac buoc:

1. Capture frame tu camera.
2. Detect tat ca khuon mat bang OpenCV Haar Cascade.
3. Chuyen moi bbox thanh square crop co margin.
4. Resize/cvtColor/normalize crop ve input shape cua model.
5. Goi OpenVINO model de lay probability vector.
6. Lay top-k expression cho tung face.
7. Ve bbox va label ngay tai khuon mat.
8. Publish annotated frame qua `/mjpeg`.
9. Publish structured JSON qua `/state`.

Voi multi-face, he thong khong average chung giua cac nguoi khac nhau. Moi face co ket qua rieng trong `faces[]`.

## Browser UI

FastAPI UI tai:

```text
http://127.0.0.1:8080
```

UI hien tai co:

- Realtime annotated camera.
- Recognition results panel.
- Card rieng cho tung face.
- Top-k probability bars trong moi face card.
- `Dung lai` de freeze panel.
- `Chay tiep` de tiep tuc realtime update.
- `Luu snapshot` de ghi SQLite.
- Saved history list.

Endpoints:

| Endpoint | Purpose |
| --- | --- |
| `/` | Browser UI. |
| `/mjpeg` | MJPEG annotated camera stream. |
| `/snapshot.jpg` | Latest annotated JPEG. |
| `/state` | Latest recognition state JSON. |
| `/save` | Save current state to SQLite. |
| `/history` | Recent saved snapshots. |
| `/health` | Basic pipeline status. |

## SQLite History

Snapshots duoc luu vao:

```text
data\fer_recognition_history.sqlite3
```

Bang chinh:

```text
recognition_snapshots
```

Moi row luu:

- thoi gian tao snapshot.
- source.
- status.
- label/confidence chinh.
- face_count.
- sample_count.
- latency/device.
- top emotions JSON.
- faces JSON, gom label, confidence, top-k va bbox cua tung face.
- probabilities JSON.

Folder `data/` nam trong `.gitignore`, nen history local khong bi push len GitHub.

## Phone Camera Flow

De dung camera dien thoai lam source, dien thoai can app IP camera rieng. Browser tren dien thoai khong tu bien camera thanh URL stream.

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

Neu muon dien thoai xem UI cua laptop:

```powershell
python -m uvicorn fer_grpc.stream_ingest_server:app --host 0.0.0.0 --port 8080
```

Sau do mo tren dien thoai:

```text
http://<laptop-ip>:8080
```

Can phan biet:

- IP dien thoai la camera source.
- IP laptop la UI/server source.
- `127.0.0.1` chi dung tren chinh may dang chay service.

## Training Pipeline

Notebook training duy nhat:

```text
train_kaggle_fer_openvino_run_all.ipynb
```

Dataset raw duoc dat o:

```text
data\kaggle_fer_yolo_raw\9 Facial Expressions you need
```

Notebook lam cac buoc:

1. Kiem tra project va raw dataset.
2. Xoa output train/export cu de tranh lan model.
3. Doc YOLO bbox labels cua dataset.
4. Crop face boxes thanh classification dataset.
5. Train/fine-tune Ultralytics YOLO classification model.
6. Validate top-1/top-5.
7. Export checkpoint tot nhat sang OpenVINO.
8. Publish model vao `models\fer_expression_yolo26n_openvino`.
9. Verify runtime OpenVINO classifier load duoc model.

## Stream Ingest Configuration

| Variable | Default | Meaning |
| --- | --- | --- |
| `IP_CAMERA_URL` | `0` | Camera source. `0` la webcam laptop, URL la IP camera/phone. |
| `MODEL_SERVER_ADDRESS` | `127.0.0.1:50051` | Dia chi gRPC model server. |
| `MODEL_SERVER_TIMEOUT_SECONDS` | `90` | Timeout cho request inference batch. |
| `INGEST_BATCH_SIZE` | `16` | So frame toi da trong mot inference request. |
| `INGEST_BATCH_TIMEOUT_SECONDS` | `0.25` | Thoi gian doi gom batch. |
| `INGEST_FRAME_QUEUE_SIZE` | `64` | Queue frame capture. |
| `INGEST_RESULT_QUEUE_SIZE` | `64` | Queue frame da infer. |
| `INGEST_AVERAGE_WINDOW` | `3` | Dung cho fallback single-result smoothing. |
| `INGEST_FACE_CROP` | `true` | Bat/tat face crop trong model server. |
| `INGEST_SHOW_WINDOW` | `false` | Mo OpenCV window local neu can. |

Voi laptop CPU, `INGEST_BATCH_SIZE=4` thuong on dinh hon `16`. Batch 16 van duoc ho tro, nhung CPU yeu co the cham hoac timeout.

## Tests

Chay unit tests:

```powershell
python -m unittest discover -s tests
```

Test hien tai cover:

- OpenVINO path validation.
- Probability smoothing.
- SQLite recognition snapshot save/fetch.
- gRPC JSON payload roundtrip.
- Stream ingest config.
- `/save` endpoint.
- Per-face state payload.

## Current Limitations

- Face detector hien tai la OpenCV Haar Cascade, nhe va de chay nhung co the miss face nghieng, nho, bi che, hoac anh sang xau.
- Chua co face tracking ID, nen `Face #1`/`Face #2` la theo frame hien tai, khong phai identity co dinh theo thoi gian.
- OpenVINO model hien tai co the export batch 1; wrapper se thu dynamic batch va fallback per-frame neu can.
- gRPC dang dung local insecure port, chua co TLS.
- gRPC payload dang dung JSON bytes de don gian, chua dung generated protobuf schema.
- Ket qua chi la visible expression recognition, khong nen xem la cam xuc noi tam chac chan.

## Possible Improvements

- Doi Haar Cascade sang MediaPipe/RetinaFace/YOLO face detector de bat nhieu mat on dinh hon.
- Them face tracking de gan ID on dinh cho tung nguoi.
- Export OpenVINO model voi dynamic batch hoac batch 16 that su.
- Dung protobuf schema chuan cho gRPC.
- Them TLS neu deploy qua network that.
- Them dashboard FPS, latency, queue size, dropped frames.
- Them confusion matrix/per-class report trong docs training.
- Them model registry/versioning de quan ly nhieu ban OpenVINO model.
