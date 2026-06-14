# Realtime Call-Center Emotion Coach

This project runs realtime facial-expression recognition for a call-center coach UI.
The active runtime is OpenVINO, not PyTorch/TorchServe/Docker.

Active model:

`models/fer_expression_yolo26n_openvino/best.xml`

The app uses camera input, detects/crops the face, runs the OpenVINO IR model, samples 3 frames per second, and averages the latest 3 probability vectors before updating the robot coach panel.

The robot coach panel has:

- `Dung lai`: freeze the current top-2 emotion mix and suggestions so you can read them.
- `Chay tiep`: return to live updates.
- `Luu`: save the current snapshot manually.
- Auto-save history: writes snapshots to `data/fer_realtime_history.sqlite3` about once per second while realtime inference is active.

## Install In Anaconda

```powershell
cd "C:\Users\DELL\Desktop\Vinh Hoang\Master Program\AI trong sản xuất DevOps, DataOps, MLOps\Lab1"
python -m pip install -r requirements-realtime.txt
python -c "import openvino as ov; print(ov.__version__); print(ov.Core().available_devices)"
```

Use `python -m pip` instead of `pip` on Windows machines where Device Guard blocks `Scripts\pip.exe`.

## Train A Fresh Kaggle Model

Training dependencies:

```powershell
cd "C:\Users\DELL\Desktop\Vinh Hoang\Master Program\AI trong sản xuất DevOps, DataOps, MLOps\Lab1"
python -m pip install -r requirements-train.txt
```

Kaggle download needs an API token:

1. Kaggle account -> Settings -> API -> Create New Token.
2. Put the downloaded file here:

```text
C:\Users\DELL\.kaggle\kaggle.json
```

Use one training notebook only:

`train_kaggle_fer_openvino_run_all.ipynb`

Open it in VS Code/Jupyter, select the Anaconda Python kernel, then click `Run All`.

What it does:

- uses the raw Kaggle dataset in `data/kaggle_fer_yolo_raw/9 Facial Expressions you need`;
- crops the YOLO face boxes into `data/fer_expression_yolo_prepared/train|val|test/<expression>/*.jpg`;
- trains an Ultralytics classification model;
- evaluates top-1/top-5 metrics;
- exports OpenVINO IR;
- publishes the verified model to `models\fer_expression_yolo26n_openvino`;
- freezes the early YOLO layers and fine-tunes only the final blocks/classifier head by default;
- allows up to 100 epochs, with early stopping, cosine LR decay, warmup, momentum, weight decay, and dropout enabled for safer convergence.

CPU-only training will be slow. This machine's current Anaconda PyTorch build reports CPU-only, so use fewer epochs for a smoke test:

In the first notebook cell, set `EPOCHS = 3` and `RUN_NAME = "smoke_test"` before `Run All`.

## Run Local Browser-Camera UI

```powershell
python -m streamlit run app_realtime_call_center.py
```

The sidebar should point to:

```text
models\fer_expression_yolo26n_openvino
```

Use `AUTO` first. OpenVINO will choose an available Intel CPU/GPU/NPU path when supported.

## Run IP-Camera + gRPC HTTP/2 Flow

Terminal 1: start the OpenVINO model server.

```powershell
cd "C:\Users\DELL\Desktop\Vinh Hoang\Master Program\AI trong sản xuất DevOps, DataOps, MLOps\Lab1"
python -m fer_grpc.server --model "models\fer_expression_yolo26n_openvino" --device AUTO --address 127.0.0.1:50051
```

Terminal 2: start stream ingest and browser preview.

```powershell
cd "C:\Users\DELL\Desktop\Vinh Hoang\Master Program\AI trong sản xuất DevOps, DataOps, MLOps\Lab1"
set IP_CAMERA_URL=http://192.168.1.50:8080/video
set MODEL_SERVER_ADDRESS=127.0.0.1:50051
set INGEST_BATCH_SIZE=16
python -m uvicorn fer_grpc.stream_ingest_server:app --host 127.0.0.1 --port 8080
```

Open this in a browser, not in the Anaconda command line:

```text
http://127.0.0.1:8080
```

For a phone camera, install an IP camera app on the phone, connect phone and PC to the same Wi-Fi, then use the app's stream URL as `IP_CAMERA_URL`.

## What Changed

- Added `fer_realtime/` as a small reusable inference package.
- Added `app_realtime_call_center.py` with two live panels: camera and robot coach.
- Added face crop before classification, matching the notebook's future-work note.
- Added probability smoothing: 3 sampled frames per second, average latest 3 predictions.
- Added OpenVINO IR loading and validation. `.pt` checkpoints are no longer loaded directly at runtime.
- Added freeze/save controls and SQLite snapshot history.
- Added `fer_grpc/` for local gRPC over HTTP/2 model serving and IP-camera stream ingest.
- Removed the Docker/TorchServe deployment path.

## Model Replacement

If you train a new YOLO `.pt`, export it to OpenVINO first and point the app/server to the exported folder containing `.xml`, `.bin`, and `metadata.yaml`.

Changing only `best.pt` is no longer enough for this OpenVINO runtime. The training pipeline above publishes the new exported model to `models\fer_expression_yolo26n_openvino`, which the app picks up automatically.

## Notes

- This predicts visible facial expression classes, not true internal emotion.
- The coaching text is a support cue for agents, not a psychological diagnosis.
- OpenVINO is usually a better deployment target on Intel CPU/GPU/NPU hardware than raw PyTorch, but actual speed still depends on the chip, driver, model shape, and preprocessing cost. Run with `AUTO` first, then benchmark `CPU`, `GPU`, and `NPU` if available.

For the gRPC/IP-camera design, see `docs/openvino_grpc_architecture.md`.
