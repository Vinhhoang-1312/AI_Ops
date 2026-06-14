"""Browser preview server for the local OpenVINO gRPC stream ingest."""

from __future__ import annotations

import asyncio

from fastapi import FastAPI, Response
from fastapi.responses import HTMLResponse, StreamingResponse

from fer_realtime.config import ROBOT_SVG_PATH

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
          :root {
            --ink:#121924; --muted:#5c6975; --line:#d8e0e7; --bg:#f4f7fb;
            --panel:#fff; --teal:#0e766f; --gold:#b7791f; --red:#d64545;
          }
          * { box-sizing: border-box; }
          body { margin: 0; background: radial-gradient(circle at top left, #e7f4f2, var(--bg) 34%, #f8fafc); color: var(--ink); font-family: "Trebuchet MS", "Segoe UI", sans-serif; }
          main { max-width: 1440px; margin: 0 auto; padding: 24px; }
          header { margin-bottom: 14px; }
          h1 { margin: 0 0 4px; font-size: 26px; letter-spacing: 0; }
          .subtitle { color: var(--muted); margin: 0; font-size: 15px; }
          .grid { display: grid; grid-template-columns: minmax(0, 1.55fr) minmax(360px, .9fr); gap: 18px; align-items: start; }
          .panel { background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 14px; box-shadow: 0 16px 34px rgba(22,38,51,.08); }
          .camera img { display: block; width: 100%; border-radius: 6px; background: #101820; min-height: 420px; object-fit: contain; }
          .coach-head { display: grid; grid-template-columns: 116px 1fr; gap: 16px; align-items: center; margin-bottom: 16px; }
          .robot { width: 116px; height: 116px; border: 1px solid var(--line); border-radius: 8px; padding: 10px; background: #f8fbfd; }
          .eyebrow { color: var(--teal); font-size: 12px; font-weight: 800; letter-spacing: .08em; text-transform: uppercase; }
          .label { font-size: 30px; font-weight: 900; line-height: 1.05; margin: 5px 0; }
          .meta { color: var(--muted); font-size: 14px; }
          .headline { border-left: 4px solid var(--red); padding: 8px 0 8px 12px; font-size: 19px; font-weight: 800; line-height: 1.35; margin: 12px 0 16px; }
          .cue-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
          .cue-card { border: 1px solid var(--line); border-radius: 8px; padding: 12px; }
          .cue-title { color: var(--teal); font-size: 12px; font-weight: 900; letter-spacing: .08em; text-transform: uppercase; margin-bottom: 8px; }
          .suggestion { margin-top: 14px; border: 1px solid #f0ca83; background: #fff8e8; border-radius: 8px; padding: 12px; color: #5a3510; line-height: 1.45; }
          .bars { margin-top: 14px; }
          .bar-row { margin: 9px 0; }
          .bar-text { display: flex; justify-content: space-between; font-size: 13px; margin-bottom: 4px; }
          .track { height: 8px; border-radius: 99px; background: #edf1f5; overflow: hidden; }
          .fill { height: 100%; background: linear-gradient(90deg, var(--teal), #2b8aef); width: 0%; }
          .links { margin-top: 10px; font-weight: 800; }
          .links a { color: var(--teal); }
          @media (max-width: 980px) { .grid { grid-template-columns: 1fr; } .camera img { min-height: 260px; } }
        </style>
      </head>
      <body>
        <main>
          <header>
            <h1>FER OpenVINO gRPC Stream Ingest</h1>
            <p class="subtitle">IP camera capture -> batched gRPC inference -> realtime robot coach.</p>
          </header>
          <section class="grid">
            <div>
              <div class="panel camera"><img src="/mjpeg" alt="FER stream preview" /></div>
              <div class="links"><a href="/health">Health</a> | <a href="/state">State</a> | <a href="/snapshot.jpg">Snapshot JPEG</a></div>
            </div>
            <aside class="panel coach">
              <div class="coach-head">
                <img class="robot" src="/support_robot.svg" alt="Support robot" />
                <div>
                  <div class="eyebrow">Robot coach</div>
                  <div class="label" id="label">Waiting</div>
                  <div class="meta" id="meta">Waiting for realtime frames</div>
                </div>
              </div>
              <div class="headline" id="headline">Dang doi tin hieu tu camera va model server.</div>
              <div class="cue-grid">
                <div class="cue-card">
                  <div class="cue-title">Tone</div>
                  <div id="tone">Binh tinh, quan sat them.</div>
                </div>
                <div class="cue-card">
                  <div class="cue-title">Action</div>
                  <div id="action">Doi them mau frame on dinh.</div>
                </div>
              </div>
              <div class="suggestion" id="suggestion">Em dang lang nghe. Anh/chi co the chia se them mot chut ve van de minh dang gap khong?</div>
              <div class="bars" id="bars"></div>
            </aside>
          </section>
        </main>
        <script>
          const pct = (v) => `${Math.round((v || 0) * 1000) / 10}%`;
          function escapeHtml(value) {
            return String(value ?? "").replace(/[&<>"']/g, (ch) => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[ch]));
          }
          function renderBars(items) {
            return (items || []).slice(0, 5).map((item) => `
              <div class="bar-row">
                <div class="bar-text"><span>${escapeHtml(item.label)}</span><strong>${pct(item.score)}</strong></div>
                <div class="track"><div class="fill" style="width:${Math.max(0, Math.min(100, (item.score || 0) * 100))}%"></div></div>
              </div>
            `).join("");
          }
          async function refreshState() {
            try {
              const res = await fetch('/state', { cache: 'no-store' });
              const state = await res.json();
              const cue = (state.cues && state.cues[0]) || {};
              document.getElementById('label').textContent = cue.display_name || state.label || 'Waiting';
              document.getElementById('meta').textContent = `Status: ${state.status || 'waiting'} | Samples: ${state.sample_count || 0} | ${state.device || ''}`;
              document.getElementById('headline').textContent = cue.headline || state.summary || 'Dang doi tin hieu.';
              document.getElementById('tone').textContent = cue.tone || 'Binh tinh, quan sat them.';
              document.getElementById('action').textContent = cue.action || 'Doi them mau frame on dinh.';
              document.getElementById('suggestion').textContent = cue.suggested_response || state.summary || '';
              document.getElementById('bars').innerHTML = renderBars(state.top_k);
            } catch (err) {
              document.getElementById('headline').textContent = `State error: ${err}`;
            }
          }
          refreshState();
          setInterval(refreshState, 700);
        </script>
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


@app.get("/state")
def state() -> dict[str, object]:
    if pipeline is None:
        return {"ok": False, "status": "not_started"}
    payload = pipeline.latest_state_snapshot()
    payload["ok"] = True
    return payload


@app.get("/support_robot.svg")
def support_robot() -> Response:
    if not ROBOT_SVG_PATH.exists():
        return Response(status_code=404)
    return Response(content=ROBOT_SVG_PATH.read_bytes(), media_type="image/svg+xml")


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
