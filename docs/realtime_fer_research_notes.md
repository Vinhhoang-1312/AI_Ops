# Realtime FER Research Notes

Design choices used in the app:

- The trained YOLO classifier is exported to OpenVINO IR and loaded from `models/fer_expression_yolo26n_openvino/best.xml`.
- OpenVINO is used for local Intel CPU/GPU/NPU-friendly inference instead of runtime PyTorch.
- gRPC provides the local HTTP/2 serving boundary between stream ingest and model serving.
- Streamlit WebRTC is a practical fit for browser camera apps because it exposes realtime video frame callbacks inside Streamlit.
- DeepFace's realtime mode waits for sequential face stability before showing results. This project uses a lighter equivalent: sample 3 frames/second and average the latest 3 probability vectors.
- LibreFace and Py-Feat both frame facial-expression analysis as face-first pipelines, so the app adds face detection/cropping before classification instead of classifying the full webcam frame.
- Call-center emotion work stresses continuous, contextual emotion signals. The UI therefore says "cue" and "visible expression" rather than treating the model output as ground-truth internal emotion.

References:

- https://docs.openvino.ai/2025/api/ie_python_api/_autosummary/openvino.Core.html
- https://grpc.io/docs/what-is-grpc/core-concepts/
- https://grpc.io/docs/languages/python/basics/
- https://github.com/whitphx/streamlit-webrtc
- https://github.com/serengil/deepface
- https://github.com/ihp-lab/LibreFace
- https://github.com/cosanlab/py-feat
- https://arxiv.org/abs/2310.02281
- https://arxiv.org/abs/2209.09236
