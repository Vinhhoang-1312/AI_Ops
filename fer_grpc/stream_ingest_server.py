"""Browser preview server for the local OpenVINO gRPC stream ingest."""

from __future__ import annotations

import asyncio

from fastapi import FastAPI, Response
from fastapi.responses import HTMLResponse, StreamingResponse

from fer_realtime.history import fetch_recent_snapshots, init_history_db, save_state_snapshot

from .stream_ingest import StreamIngestPipeline, config_from_env


app = FastAPI(title="FER OpenVINO gRPC Stream Ingest", version="2.0.0")
pipeline: StreamIngestPipeline | None = None


@app.on_event("startup")
def startup() -> None:
    global pipeline
    init_history_db()
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
            --panel:#fff; --teal:#0e766f; --blue:#2563eb; --red:#d64545;
          }
          * { box-sizing: border-box; }
          body { margin: 0; background: radial-gradient(circle at top left, #e7f4f2, var(--bg) 34%, #f8fafc); color: var(--ink); font-family: "Segoe UI", Arial, sans-serif; }
          main { max-width: 1440px; margin: 0 auto; padding: 24px; }
          header { margin-bottom: 14px; }
          h1 { margin: 0 0 4px; font-size: 26px; letter-spacing: 0; }
          .subtitle { color: var(--muted); margin: 0; font-size: 15px; }
          .grid { display: grid; grid-template-columns: minmax(0, 1.55fr) minmax(360px, .9fr); gap: 18px; align-items: start; }
          .panel { background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 14px; box-shadow: 0 16px 34px rgba(22,38,51,.08); }
          .camera img { display: block; width: 100%; border-radius: 6px; background: #101820; min-height: 420px; object-fit: contain; }
          .eyebrow { color: var(--teal); font-size: 12px; font-weight: 800; letter-spacing: .08em; text-transform: uppercase; }
          .label { font-size: 28px; font-weight: 900; line-height: 1.08; margin: 5px 0; }
          .meta { color: var(--muted); font-size: 14px; }
          .summary { border-left: 4px solid var(--teal); padding: 8px 0 8px 12px; font-size: 16px; font-weight: 800; line-height: 1.35; margin: 12px 0 16px; }
          .controls { display: flex; gap: 10px; flex-wrap: wrap; margin: 12px 0 4px; }
          button { border: 0; border-radius: 8px; padding: 10px 13px; font-weight: 900; cursor: pointer; }
          .primary { background: var(--teal); color: #fff; }
          .secondary { background: #e9eef3; color: var(--ink); }
          .save-note { color: var(--muted); font-size: 13px; min-height: 18px; }
          .faces { display: grid; gap: 12px; margin-top: 12px; }
          .face-card { border: 1px solid var(--line); border-radius: 8px; padding: 12px; background: #fbfdff; }
          .face-top { display: flex; justify-content: space-between; gap: 12px; align-items: start; margin-bottom: 10px; }
          .face-title { font-weight: 900; }
          .face-label { color: var(--teal); font-size: 20px; font-weight: 900; margin-top: 2px; }
          .face-conf { color: var(--blue); font-weight: 900; white-space: nowrap; }
          .bar-row { margin: 9px 0; }
          .bar-text { display: flex; justify-content: space-between; font-size: 13px; margin-bottom: 4px; }
          .track { height: 8px; border-radius: 99px; background: #edf1f5; overflow: hidden; }
          .fill { height: 100%; background: linear-gradient(90deg, var(--teal), #2b8aef); width: 0%; }
          .links { margin-top: 10px; font-weight: 800; }
          .links a { color: var(--teal); }
          .history { margin-top: 14px; border-top: 1px solid var(--line); padding-top: 12px; }
          .history h2 { margin: 0 0 8px; font-size: 16px; }
          .history-row { border: 1px solid var(--line); border-radius: 8px; padding: 9px; margin: 7px 0; background: #fbfdff; }
          .history-row strong { display: block; }
          .history-row span { color: var(--muted); font-size: 12px; }
          .empty { color: var(--muted); border: 1px dashed var(--line); border-radius: 8px; padding: 12px; background: #fbfdff; }
          @media (max-width: 980px) { .grid { grid-template-columns: 1fr; } .camera img { min-height: 260px; } }
        </style>
      </head>
      <body>
        <main>
          <header>
            <h1>FER OpenVINO gRPC Stream Ingest</h1>
            <p class="subtitle">Camera capture -> multi-face detection -> batched gRPC expression recognition.</p>
          </header>
          <section class="grid">
            <div>
              <div class="panel camera"><img src="/mjpeg" alt="FER stream preview" /></div>
              <div class="links"><a href="/health">Health</a> | <a href="/state">State</a> | <a href="/snapshot.jpg">Snapshot JPEG</a></div>
            </div>
            <aside class="panel">
              <div>
                <div class="eyebrow">Recognition results</div>
                <div class="label" id="label">Waiting</div>
                <div class="meta" id="meta">Waiting for realtime frames</div>
              </div>
              <div class="summary" id="summary">Dang doi tin hieu tu camera va model server.</div>
              <div class="controls">
                <button class="primary" id="pauseBtn" type="button">Dung lai</button>
                <button class="secondary" id="saveBtn" type="button">Luu snapshot</button>
              </div>
              <div class="save-note" id="saveNote"></div>
              <div class="faces" id="faces"></div>
              <div class="history">
                <h2>Saved history</h2>
                <div id="history">No saved snapshots yet.</div>
              </div>
            </aside>
          </section>
        </main>
        <script>
          let paused = false;
          let pausedState = null;
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
          function renderFaceCards(faces) {
            if (!faces || faces.length === 0) {
              return `<div class="empty">Chua phat hien khuon mat nao trong frame moi nhat.</div>`;
            }
            return faces.map((face, index) => {
              const region = face.face_region || {};
              const faceId = face.track_id ?? (index + 1);
              return `
                <div class="face-card">
                  <div class="face-top">
                    <div>
                      <div class="face-title">Face #${faceId}</div>
                      <div class="face-label">${escapeHtml(face.label || 'unknown')}</div>
                      <div class="meta">bbox x=${region.x ?? 0}, y=${region.y ?? 0}, w=${region.w ?? 0}, h=${region.h ?? 0}</div>
                    </div>
                    <div class="face-conf">${pct(face.confidence)}</div>
                  </div>
                  ${renderBars(face.top_k)}
                </div>
              `;
            }).join("");
          }
          function renderState(state) {
            const faces = state.faces || [];
            const faceText = `${faces.length} face${faces.length === 1 ? '' : 's'}`;
            document.getElementById('label').textContent = faces.length ? `${faceText} detected` : (state.label || 'Waiting');
            document.getElementById('meta').textContent = `Status: ${state.status || 'waiting'} | Frame: ${state.frame_id ?? '-'} | ${state.device || ''}`;
            document.getElementById('summary').textContent = state.summary || 'Dang doi tin hieu.';
            document.getElementById('faces').innerHTML = renderFaceCards(faces);
          }
          function renderHistory(rows) {
            const target = document.getElementById('history');
            if (!rows || rows.length === 0) {
              target.textContent = 'No saved snapshots yet.';
              return;
            }
            target.innerHTML = rows.slice(0, 6).map((row) => {
              const top = (row.top_emotions || []).map(([label, score]) => `${escapeHtml(label)} ${pct(score)}`).join(' | ');
              return `<div class="history-row"><strong>#${row.id} ${top || escapeHtml(row.label || 'unknown')}</strong><span>${escapeHtml(row.created_at)} | ${escapeHtml(row.source)} | faces=${row.face_count || 0}</span></div>`;
            }).join('');
          }
          async function refreshState() {
            if (paused) return;
            try {
              const res = await fetch('/state', { cache: 'no-store' });
              const state = await res.json();
              pausedState = state;
              renderState(state);
            } catch (err) {
              document.getElementById('summary').textContent = `State error: ${err}`;
            }
          }
          async function refreshHistory() {
            const res = await fetch('/history', { cache: 'no-store' });
            const payload = await res.json();
            renderHistory(payload.items || []);
          }
          document.getElementById('pauseBtn').addEventListener('click', () => {
            paused = !paused;
            document.getElementById('pauseBtn').textContent = paused ? 'Chay tiep' : 'Dung lai';
            document.getElementById('saveNote').textContent = paused ? 'Panel da dung de ban doc ky ket qua nhan dien.' : '';
            if (paused && pausedState) renderState(pausedState);
          });
          document.getElementById('saveBtn').addEventListener('click', async () => {
            const res = await fetch('/save', { method: 'POST' });
            const payload = await res.json();
            document.getElementById('saveNote').textContent = payload.ok ? `Da luu snapshot #${payload.id}` : `Luu that bai: ${payload.error || 'unknown'}`;
            await refreshHistory();
          });
          refreshState();
          refreshHistory();
          setInterval(refreshState, 700);
          setInterval(refreshHistory, 5000);
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


@app.post("/save")
def save() -> dict[str, object]:
    if pipeline is None:
        return {"ok": False, "error": "not_started"}
    state = pipeline.latest_state()
    row_id = save_state_snapshot(state, source="stream_ingest_manual")
    return {"ok": True, "id": row_id}


@app.get("/history")
def history() -> dict[str, object]:
    return {"ok": True, "items": fetch_recent_snapshots(limit=8)}


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
