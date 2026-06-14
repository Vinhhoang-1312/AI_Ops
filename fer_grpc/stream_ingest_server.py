"""Browser preview server for the local OpenVINO gRPC stream ingest."""

from __future__ import annotations

import asyncio

from fastapi import FastAPI, Response
from fastapi.responses import HTMLResponse, StreamingResponse

from .stream_ingest import StreamIngestPipeline, config_from_env


app = FastAPI(title="FER OpenVINO gRPC Stream Ingest", version="2.0.0")
pipeline: StreamIngestPipeline | None = None


@app.on_event("startup")
def startup() -> None:
    global pipeline
    pipeline = StreamIngestPipeline(config_from_env())
    pipeline.start()


@app.on_event("shutdown")
def shutdown() -> None:
    if pipeline is not None:
        pipeline.stop()


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return """
    <!doctype html>
    <html>
      <head>
        <title>FER OpenVINO gRPC Stream Ingest</title>
        <style>
          :root { --ink:#17212b; --muted:#5c6975; --line:#d8e0e7; --bg:#f6f8fb; --teal:#0e766f; }
          body { margin: 0; background: var(--bg); color: var(--ink); font-family: Arial, sans-serif; }
          main { max-width: 1180px; margin: 0 auto; padding: 28px; }
          h1 { margin: 0 0 6px; font-size: 28px; letter-spacing: 0; }
          p { color: var(--muted); margin: 0 0 18px; }
          .stage { background: #fff; border: 1px solid var(--line); border-radius: 8px; padding: 16px; box-shadow: 0 12px 30px rgba(22,38,51,.07); }
          img { display: block; width: 100%; border-radius: 8px; background: #101820; }
          a { color: var(--teal); font-weight: 700; }
        </style>
      </head>
      <body>
        <main>
          <h1>FER OpenVINO gRPC Stream Ingest</h1>
          <p>IP camera capture -> batched gRPC inference -> realtime visualization.</p>
          <div class="stage"><img src="/mjpeg" alt="FER stream preview" /></div>
          <p><a href="/health">Health</a> | <a href="/snapshot.jpg">Snapshot JPEG</a></p>
        </main>
      </body>
    </html>
    """


@app.get("/health")
def health() -> dict[str, object]:
    if pipeline is None:
        return {"ok": False, "status": "not_started"}
    return {
        "ok": True,
        "status": "running" if not pipeline.stop_event.is_set() else "stopped",
        "latest": pipeline.latest_summary(),
    }


@app.get("/snapshot.jpg")
def snapshot() -> Response:
    image = pipeline.latest_jpeg() if pipeline is not None else None
    if image is None:
        return Response(status_code=204)
    return Response(content=image, media_type="image/jpeg")


@app.get("/mjpeg")
async def mjpeg() -> StreamingResponse:
    async def frames():
        while True:
            image = pipeline.latest_jpeg() if pipeline is not None else None
            if image is not None:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + image + b"\r\n"
            await asyncio.sleep(0.1)

    return StreamingResponse(frames(), media_type="multipart/x-mixed-replace; boundary=frame")
