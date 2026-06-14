# Lab1 - Realtime Facial Expression Coach With OpenVINO

## Quick Setup

Chay AI model server:

```powershell
cd "C:\Users\DELL\Desktop\Vinh Hoang\Master Program\AI trong sản xuất DevOps, DataOps, MLOps\Lab1"
python -m pip install -r requirements-realtime.txt
python -m fer_grpc.server --model "models\fer_expression_yolo26n_openvino" --device AUTO --address 127.0.0.1:50051
```

Chay stream ingest + UI bang camera laptop:

```powershell
cd "C:\Users\DELL\Desktop\Vinh Hoang\Master Program\AI trong sản xuất DevOps, DataOps, MLOps\Lab1"
set IP_CAMERA_URL=0
set MODEL_SERVER_ADDRESS=127.0.0.1:50051
set MODEL_SERVER_TIMEOUT_SECONDS=90
set INGEST_BATCH_SIZE=4
python -m uvicorn fer_grpc.stream_ingest_server:app --host 127.0.0.1 --port 8080
```

Mo trinh duyet:

```text
http://127.0.0.1:8080
```

Neu muon xem UI tu dien thoai cung Wi-Fi, chay `uvicorn` voi `--host 0.0.0.0` va mo `http://<IP-laptop>:8080`.

## Project Summary

Du an nay la mot he thong realtime facial-expression recognition cho ngu canh chatbot/call center. Y tuong chinh la camera ghi lai bieu cam cua nguoi dung, AI model phan loai cam xuc nhin thay tren khuon mat, sau do robot coach goi y cach nhan vien nen dieu chinh tone, cach hoi, va cau tra loi tiep theo.

He thong hien tai tap trung vao inference thuc te bang OpenVINO tren may Intel. Training van dung Ultralytics YOLO/PyTorch de fine-tune model, nhung runtime khong load `.pt` truc tiep. Runtime load OpenVINO IR gom `best.xml`, `best.bin`, va `metadata.yaml`.

Du an hien co hai cach chay realtime:

- Local Streamlit app: browser camera + robot coach panel.
- Distributed local flow: IP camera/webcam -> stream ingest -> gRPC model serving -> browser preview + robot coach.

## Current Capabilities

He thong hien tai lam duoc cac viec sau:

- Nhan frame realtime tu webcam laptop hoac IP camera URL.
- Detect/crop khuon mat bang OpenCV Haar Cascade truoc khi classify.
- Classify bieu cam bang OpenVINO model.
- Lay mau 3 frame moi giay va trung binh latest 3 probability vectors de giam nhieu.
- Hien top-2 cam xuc thay vi chi 1 label.
- Hien robot coach voi headline, tone, action, va suggested response tuong ung cam xuc.
- Co nut `Dung lai` / `Chay tiep` de freeze panel cho nguoi dung doc ky.
- Co nut `Luu snapshot` ghi ket qua hien tai vao SQLite.
- Luu history trong `data\fer_realtime_history.sqlite3`.
- Co gRPC model server rieng, giao tiep qua HTTP/2.
- Co stream ingest 3 thread: capture, inference batch, visualize.
- Co browser preview tai `http://127.0.0.1:8080`.

## Architecture

Flow local/distributed hien tai:

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
  camera panel
  robot coach panel
  pause/save/history controls
```

Hien tai he thong khong dung Docker va khong dung TorchServe. Cac service chay truc tiep bang Python/Anaconda tren may local. gRPC dang chay local insecure port, chua co TLS/HTTPS.

## Main Components

`fer_realtime/` la package inference realtime dung chung:

- `config.py`: path mac dinh, model path, asset path, SQLite path.
- `model.py`: load OpenVINO IR, preprocess frame, infer batch/frame, parse probabilities.
- `analyzer.py`: realtime analyzer, face crop, smoothing, overlay camera.
- `smoothing.py`: frame sampler va probability averager.
- `emotion_policy.py`: mapping cam xuc -> robot coach cue.
- `history.py`: SQLite persistence cho emotion snapshots.

`fer_grpc/` la package cho model serving va stream ingest:

- `server.py`: OpenVINO gRPC model server.
- `client.py`: gRPC client goi inference batch.
- `codec.py`: encode/decode JPEG base64 cho frame.
- `protocol.py`: JSON bytes protocol cho gRPC generic handler.
- `stream_ingest.py`: pipeline 3 thread capture/infer/visualize.
- `stream_ingest_server.py`: FastAPI browser UI, MJPEG stream, state API, save/history endpoints.

`app_realtime_call_center.py` la Streamlit app local co UI hai khung camera + robot coach.

`train_kaggle_fer_openvino_run_all.ipynb` la notebook training duy nhat cua project.

`models/fer_expression_yolo26n_openvino/` la OpenVINO model runtime da export.

## Model

Model runtime hien tai nam o:

```text
models\fer_expression_yolo26n_openvino
```

Folder nay gom:

```text
best.xml
best.bin
metadata.yaml
```

Day la format dung cua OpenVINO IR:

- `best.xml`: graph/kien truc model.
- `best.bin`: weights.
- `metadata.yaml`: class names va metadata tu Ultralytics.

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

OpenVINO runtime load model tu folder nay. File `.pt` chi la checkpoint PyTorch/Ultralytics trong qua trinh train/export, khong phai runtime chinh cua app.

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

1. Kiem tra thu muc project va raw dataset.
2. Xoa output train/export cu de tranh lan model.
3. Doc YOLO detection labels cua Kaggle dataset.
4. Crop face boxes thanh classification dataset.
5. Train Ultralytics YOLO classification model.
6. Validate top-1/top-5.
7. Export checkpoint tot nhat sang OpenVINO.
8. Publish model vao `models\fer_expression_yolo26n_openvino`.
9. Verify runtime OpenVINO classifier load duoc model.

Config hien tai trong notebook:

```python
BASE_MODEL = "yolo26n-cls.pt"
RUN_NAME = "fer_yolo26n_cls_kaggle_last_blocks"
IMG_SIZE = 224
EPOCHS = 20
BATCH = 16
PATIENCE = 5
FREEZE_LAYERS = 8
```

Ly do dung `yolo26n-cls.pt`: day la ban nho hon `yolo26s`, phu hop hon voi laptop CPU. Notebook freeze cac layer dau va fine-tune cac block cuoi + classifier head de giam thoi gian train.

## Realtime Inference Logic

Moi frame realtime di qua cac buoc:

1. Capture frame tu camera.
2. Face detection/crop bang OpenCV Haar Cascade.
3. Resize/cvtColor/normalize ve input shape cua model.
4. OpenVINO infer ra vector logits/probabilities.
5. Lay probability vector cua tung frame.
6. Sample 3 frame/giay.
7. Average latest 3 probability vectors.
8. Lay top-2/top-3 cam xuc.
9. Map cam xuc sang robot coaching cue.
10. Ve overlay tren camera va update UI.

Ket qua hien thi khong nen hieu la chan doan tam ly. Day la visible facial expression cue, dung de ho tro nhan vien dieu chinh cach giao tiep.

## Stream Ingest Service

Stream ingest doc cau hinh tu environment variables:

| Variable | Default | Meaning |
| --- | --- | --- |
| `IP_CAMERA_URL` | `0` | Camera source. `0` la webcam laptop, URL la IP camera/phone. |
| `MODEL_SERVER_ADDRESS` | `127.0.0.1:50051` | Dia chi gRPC model server. |
| `MODEL_SERVER_TIMEOUT_SECONDS` | `90` | Timeout cho request inference batch. |
| `INGEST_BATCH_SIZE` | `16` | So frame toi da trong mot inference request. |
| `INGEST_BATCH_TIMEOUT_SECONDS` | `0.25` | Thoi gian doi gom batch. |
| `INGEST_FRAME_QUEUE_SIZE` | `64` | Queue frame capture. |
| `INGEST_RESULT_QUEUE_SIZE` | `64` | Queue frame da infer. |
| `INGEST_AVERAGE_WINDOW` | `3` | So probability vectors de average. |
| `INGEST_FACE_CROP` | `true` | Bat/tat face crop trong model server. |
| `INGEST_SHOW_WINDOW` | `false` | Mo OpenCV window local neu can. |

Voi laptop CPU, `INGEST_BATCH_SIZE=4` thuong muot hon `16`. Batch 16 van duoc ho tro, nhung co the gap timeout neu CPU bi cham.

## Browser UI

FastAPI UI tai:

```text
http://127.0.0.1:8080
```

UI hien tai co:

- Camera realtime panel.
- Robot coach panel.
- Top emotion bars.
- Tone/action/suggested response.
- `Dung lai` de freeze panel.
- `Chay tiep` de tiep tuc realtime update.
- `Luu snapshot` de ghi SQLite.
- Saved history list.

Endpoints:

| Endpoint | Purpose |
| --- | --- |
| `/` | Browser UI. |
| `/mjpeg` | MJPEG camera stream. |
| `/snapshot.jpg` | Latest annotated JPEG. |
| `/state` | Latest robot coach state JSON. |
| `/save` | Save current state to SQLite. |
| `/history` | Recent saved snapshots. |
| `/health` | Basic pipeline status. |
| `/support_robot.svg` | Robot asset. |

## SQLite History

Snapshots duoc luu vao:

```text
data\fer_realtime_history.sqlite3
```

Bang chinh:

```text
emotion_snapshots
```

Moi row luu:

- thoi gian tao snapshot.
- source manual/auto.
- status.
- label va confidence.
- sample_count.
- latency.
- device.
- top emotions JSON.
- cue sections JSON.
- probabilities JSON.

Folder `data/` nam trong `.gitignore`, nen history local khong bi push len GitHub.

## Phone Camera Flow

De dung camera dien thoai, dien thoai can app IP camera rieng. Browser tren dien thoai khong tu bien camera thanh URL stream.

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

Neu muon dien thoai vua lam camera vua xem UI, laptop phai chay:

```powershell
python -m uvicorn fer_grpc.stream_ingest_server:app --host 0.0.0.0 --port 8080
```

Sau do dien thoai mo:

```text
http://<laptop-ip>:8080
```

Can phan biet:

- IP dien thoai la camera source.
- IP laptop la UI/server source.
- `127.0.0.1` chi dung tren chinh may dang chay service.

## Tests

Chay unit tests:

```powershell
python -m unittest discover -s tests
```

Test hien tai cover:

- OpenVINO path validation.
- probability smoothing.
- SQLite snapshot save/fetch.
- gRPC JSON payload roundtrip.
- stream ingest config.
- save endpoint cua ingest UI.

## Repository And Ignored Files

Branch GitHub hien tai:

```text
Lab1
```

Repo:

```text
https://github.com/Vinhhoang-1312/AI_Ops.git
```

`.gitignore` bo qua:

- `data/`
- `runs/`
- `*.pt`
- Python cache.
- notebook checkpoints.
- virtual env folders.

Model OpenVINO runtime trong `models/` duoc giu lai trong repo de clone ve co the test inference ngay.

## Current Limitations

He thong hien tai con mot so gioi han:

- Chua dung Docker/container trong active flow.
- Chua co HTTPS/TLS cho gRPC.
- gRPC protocol dang dung JSON bytes de don gian, chua dung generated protobuf schema.
- OpenVINO model hien tai export batch 1; request co the gom nhieu frame, nhung wrapper se fallback per-frame neu dynamic batch khong ho tro.
- Face detection dung Haar Cascade, nhe va de chay nhung khong manh bang RetinaFace/MediaPipe/YOLO face detector.
- Dataset co the co label noise, imbalance, crop qua nho, va mot so anh corrupt.
- Emotion output chi la visible expression, khong phai cam xuc noi tam chac chan.

## Possible Improvements

Nhung huong co the nang cap:

- Export OpenVINO model voi dynamic batch hoac batch 16 that su.
- Chuyen gRPC payload sang protobuf schema chuan.
- Them TLS/HTTPS cho model server va stream ingest.
- Dong goi thanh Docker Compose neu can deploy multi-machine ro rang.
- Dung face detector tot hon Haar Cascade.
- Them face tracking de giam chi phi detect moi frame.
- Them health dashboard rieng cho FPS, latency, queue size, dropped frames.
- Them calibration theo user/camera de giam neutral bias.
- Train them voi validation strategy tot hon, confusion matrix, per-class report.
- Them model registry/versioning de quan ly nhieu ban OpenVINO model.
- Tach UI thanh React/Vite neu muon dashboard dep va linh hoat hon FastAPI HTML inline.

## Safety Notes

He thong nay nen duoc xem la decision-support tool cho call center, khong phai cong cu chan doan tam ly. Robot coach chi dua ra cue dua tren bieu cam thay duoc, nen nguoi dung cuoi van can ket hop voi noi dung hoi thoai, ngu canh, va phan hoi truc tiep cua khach hang.
