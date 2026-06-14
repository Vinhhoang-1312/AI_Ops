"""Streamlit WebRTC app for realtime call-center expression coaching."""

from __future__ import annotations

import base64
import time
from html import escape
from pathlib import Path

import av
import streamlit as st
from streamlit_webrtc import VideoProcessorBase, WebRtcMode, webrtc_streamer

from fer_realtime import RealtimeEmotionAnalyzer
from fer_realtime.config import (
    DEFAULT_AVERAGE_WINDOW,
    HISTORY_DB_PATH,
    DEFAULT_OPENVINO_MODEL_PATH,
    DEFAULT_SAMPLE_RATE_HZ,
    IMG_SIZE,
    ROBOT_SVG_PATH,
)
from fer_realtime.emotion_policy import cue_for_expression
from fer_realtime.history import fetch_recent_snapshots, init_history_db, save_state_snapshot


st.set_page_config(page_title="Call Center Emotion Coach", page_icon=":robot_face:", layout="wide")

AUTO_SAVE_INTERVAL_SECONDS = 1.0


def main() -> None:
    inject_css()
    init_session_state()
    init_history_db()

    st.markdown("<h1>Call Center Emotion Coach</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p class='subtitle'>Realtime OpenVINO FER assistant for adapting tone during video calls.</p>",
        unsafe_allow_html=True,
    )

    settings = render_sidebar()
    analyzer = get_analyzer(
        str(settings["model_path"]),
        settings["device"],
        settings["sample_rate_hz"],
        settings["average_window"],
        settings["face_crop"],
    )

    camera_col, coach_col = st.columns([1.18, 0.82], gap="large")

    with camera_col:
        st.markdown("<div class='section-label'>Realtime camera</div>", unsafe_allow_html=True)
        ctx = webrtc_streamer(
            key="fer-realtime-call-center",
            mode=WebRtcMode.SENDRECV,
            media_stream_constraints={"video": True, "audio": False},
            video_processor_factory=lambda: EmotionVideoProcessor(analyzer),
            async_processing=True,
        )

    with coach_col:
        st.markdown("<div class='section-label'>Robot coach</div>", unsafe_allow_html=True)
        controls = st.columns([1, 1, 1], gap="small")
        live_state = analyzer.get_state()
        display_state = st.session_state.frozen_state or live_state

        if controls[0].button("Dung lai", use_container_width=True):
            st.session_state.frozen_state = live_state
            display_state = st.session_state.frozen_state
        if controls[1].button("Chay tiep", use_container_width=True):
            st.session_state.frozen_state = None
            display_state = live_state
        if controls[2].button("Luu", use_container_width=True):
            saved_id = save_state_snapshot(display_state, source="manual")
            st.session_state.last_saved_message = f"Saved snapshot #{saved_id}"

        if st.session_state.frozen_state is not None:
            st.markdown("<div class='freeze-banner'>Da dung panel tai snapshot hien tai. Bam Chay tiep de cap nhat lai.</div>", unsafe_allow_html=True)
        if st.session_state.last_saved_message:
            st.caption(st.session_state.last_saved_message)

        placeholder = st.empty()
        history_placeholder = st.empty()
        render_coach_panel(placeholder, display_state, frozen=st.session_state.frozen_state is not None)
        render_history_panel(history_placeholder)

        while ctx.state.playing:
            live_state = analyzer.get_state()
            maybe_auto_save(live_state, settings["auto_save_history"])
            display_state = st.session_state.frozen_state or live_state
            render_coach_panel(placeholder, display_state, frozen=st.session_state.frozen_state is not None)
            render_history_panel(history_placeholder)
            time.sleep(0.35)


class EmotionVideoProcessor(VideoProcessorBase):
    def __init__(self, analyzer: RealtimeEmotionAnalyzer) -> None:
        self.analyzer = analyzer

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        image = frame.to_ndarray(format="bgr24")
        annotated = self.analyzer.process_frame(image)
        return av.VideoFrame.from_ndarray(annotated, format="bgr24")


@st.cache_resource(show_spinner="Loading OpenVINO expression model...")
def get_analyzer(
    model_path: str,
    device: str,
    sample_rate_hz: float,
    average_window: int,
    face_crop: bool,
) -> RealtimeEmotionAnalyzer:
    return RealtimeEmotionAnalyzer(
        model_path=model_path,
        imgsz=IMG_SIZE,
        device=device,
        sample_rate_hz=sample_rate_hz,
        average_window=average_window,
        face_crop=face_crop,
    )


def render_sidebar() -> dict[str, object]:
    with st.sidebar:
        st.header("Runtime")
        model_path = Path(
            st.text_input(
                "OpenVINO model folder/XML",
                value=str(DEFAULT_OPENVINO_MODEL_PATH),
            )
        )
        device = st.selectbox("OpenVINO device", ["AUTO", "CPU", "GPU", "NPU"], index=0)
        face_crop = st.toggle("Face crop", value=True)
        sample_rate_hz = st.slider("Samples / second", min_value=1.0, max_value=8.0, value=DEFAULT_SAMPLE_RATE_HZ, step=1.0)
        average_window = st.slider("Average frames", min_value=1, max_value=9, value=DEFAULT_AVERAGE_WINDOW, step=1)
        st.divider()
        auto_save_history = st.toggle("Auto-save history", value=True)
        st.caption(f"SQLite: {HISTORY_DB_PATH}")

    return {
        "model_path": model_path,
        "device": device,
        "face_crop": face_crop,
        "sample_rate_hz": float(sample_rate_hz),
        "average_window": int(average_window),
        "auto_save_history": bool(auto_save_history),
    }


def render_coach_panel(placeholder: st.delta_generator.DeltaGenerator, state, frozen: bool = False) -> None:
    top_emotions = top_emotion_mix(state, limit=2)
    robot_data_uri = svg_data_uri(ROBOT_SVG_PATH)
    expression_html = render_emotion_mix_html(top_emotions)
    status = escape("Frozen snapshot" if frozen else status_text(state.status))
    cue_sections_html = render_cue_sections_html(top_emotions)

    with placeholder.container():
        st.markdown(
            f"""
            <div class="coach-panel">
              <div class="robot-row">
                <img class="robot-avatar" src="{robot_data_uri}" alt="Support robot" />
                <div>
                  <div class="eyebrow">{status}</div>
                  <div class="emotion">Top 2 cam xuc hien tai</div>
                  {expression_html}
                  <div class="confidence">Averaged probability vectors | Samples: {state.sample_count}</div>
                </div>
              </div>
              {cue_sections_html}
            </div>
            """,
            unsafe_allow_html=True,
        )

        if state.top_k:
            for label, score in state.top_k[:2]:
                st.progress(min(max(float(score), 0.0), 1.0), text=f"{label}: {score * 100:.1f}%")


def top_emotion_mix(state, limit: int = 2) -> list[tuple[str, float]]:
    if state.top_k:
        return [(str(label), float(score)) for label, score in state.top_k[:limit]]
    if state.label:
        return [(str(state.label), float(state.confidence))]
    return []


def render_emotion_mix_html(top_emotions: list[tuple[str, float]]) -> str:
    if not top_emotions:
        return "<div class='emotion-mix'><span class='emotion-pill empty'>Dang quan sat</span></div>"

    chips = []
    for rank, (label, score) in enumerate(top_emotions, start=1):
        cue = cue_for_expression(label, score)
        chips.append(
            "<span class='emotion-pill'>"
            f"<small>#{rank}</small>{escape(cue.display_name)}"
            f"<strong>{score * 100:.1f}%</strong>"
            "</span>"
        )
    return f"<div class='emotion-mix'>{''.join(chips)}</div>"


def render_cue_sections_html(top_emotions: list[tuple[str, float]]) -> str:
    if not top_emotions:
        cue = cue_for_expression(None)
        top_emotions = [(cue.label, 0.0)]

    sections = []
    for rank, (label, score) in enumerate(top_emotions, start=1):
        cue = cue_for_expression(label, score)
        sections.append(
            f"""
            <section class="cue-section">
              <div class="cue-title">
                <span>#{rank}</span>
                <strong>{escape(cue.display_name)} - {score * 100:.1f}%</strong>
              </div>
              <div class="headline">{escape(cue.headline)}</div>
              <div class="cue-grid">
                <div><span>Tone</span><p>{escape(cue.tone)}</p></div>
                <div><span>Action</span><p>{escape(cue.action)}</p></div>
              </div>
              <div class="suggestion">{escape(cue.suggested_response)}</div>
            </section>
            """
        )
    return "<div class='cue-list'>" + "".join(sections) + "</div>"


def status_text(status: str) -> str:
    if status == "ok":
        return "Averaged from realtime frames"
    if status == "no_face":
        return "No centered face detected"
    if status.startswith("error:"):
        return status
    return "Waiting for camera"


def init_session_state() -> None:
    st.session_state.setdefault("frozen_state", None)
    st.session_state.setdefault("last_auto_saved_update", 0.0)
    st.session_state.setdefault("last_auto_saved_wall_time", 0.0)
    st.session_state.setdefault("last_saved_message", "")


def maybe_auto_save(state, enabled: bool) -> None:
    if not enabled or state.status != "ok":
        return
    if not state.top_k or state.updated_at <= st.session_state.last_auto_saved_update:
        return
    now = time.monotonic()
    if now - st.session_state.last_auto_saved_wall_time < AUTO_SAVE_INTERVAL_SECONDS:
        return

    save_state_snapshot(state, source="auto")
    st.session_state.last_auto_saved_update = state.updated_at
    st.session_state.last_auto_saved_wall_time = now


def render_history_panel(placeholder: st.delta_generator.DeltaGenerator, limit: int = 10) -> None:
    with placeholder.container():
        with st.expander("Saved emotion history", expanded=False):
            rows = fetch_recent_snapshots(limit=limit)
            if not rows:
                st.caption("No saved snapshots yet.")
                return

            for row in rows:
                top_text = " | ".join(f"{label}: {score * 100:.1f}%" for label, score in row["top_emotions"])
                st.markdown(
                    f"""
                    <div class="history-row">
                      <div><strong>#{row['id']}</strong> {escape(row['created_at'])} ({escape(row['source'])})</div>
                      <div>{escape(top_text)}</div>
                      <div class="history-muted">samples={row['sample_count']} | latency={row['latency_ms']:.0f} ms | {escape(row['device'])}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                for cue in row["cue_sections"]:
                    st.caption(f"#{cue['rank']} {cue['display_name']}: {cue['suggested_response']}")


@st.cache_data(show_spinner=False)
def svg_data_uri(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/svg+xml;base64,{encoded}"


def inject_css() -> None:
    st.markdown(
        """
        <style>
        :root {
          --ink: #17212b;
          --muted: #5c6975;
          --line: #d8e0e7;
          --panel: #ffffff;
          --bg: #f6f8fb;
          --teal: #0e766f;
          --coral: #e6584f;
          --gold: #f3b545;
        }
        .stApp {
          background: var(--bg);
          color: var(--ink);
        }
        h1 {
          margin: 0 0 0.2rem 0;
          color: var(--ink);
          font-size: 2.1rem;
          letter-spacing: 0;
        }
        .subtitle {
          margin: 0 0 1rem 0;
          color: var(--muted);
          font-size: 0.98rem;
        }
        .section-label {
          color: var(--muted);
          font-size: 0.78rem;
          font-weight: 700;
          letter-spacing: 0.08em;
          margin: 0 0 0.45rem 0;
          text-transform: uppercase;
        }
        .freeze-banner {
          background: #fff8e7;
          border: 1px solid #f4dfaa;
          border-radius: 8px;
          color: #3e3320;
          font-size: 0.9rem;
          margin: 0 0 10px;
          padding: 10px 12px;
        }
        .coach-panel {
          background: var(--panel);
          border: 1px solid var(--line);
          border-radius: 8px;
          padding: 20px;
          min-height: 520px;
          box-shadow: 0 12px 30px rgba(22, 38, 51, 0.07);
        }
        .robot-row {
          align-items: center;
          display: flex;
          gap: 16px;
          margin-bottom: 18px;
        }
        .robot-avatar {
          border: 1px solid var(--line);
          border-radius: 8px;
          height: 132px;
          width: 132px;
        }
        .eyebrow {
          color: var(--teal);
          font-size: 0.78rem;
          font-weight: 700;
          letter-spacing: 0.04em;
          text-transform: uppercase;
        }
        .emotion {
          color: var(--ink);
          font-size: 1.26rem;
          font-weight: 800;
          line-height: 1.15;
          margin-top: 4px;
        }
        .emotion-mix {
          display: flex;
          flex-wrap: wrap;
          gap: 8px;
          margin-top: 10px;
        }
        .emotion-pill {
          align-items: center;
          background: #eef8f7;
          border: 1px solid #c9e6e2;
          border-radius: 999px;
          color: var(--ink);
          display: inline-flex;
          font-size: 0.92rem;
          font-weight: 750;
          gap: 8px;
          line-height: 1;
          padding: 8px 10px;
          white-space: nowrap;
        }
        .emotion-pill small {
          color: var(--teal);
          font-size: 0.74rem;
          font-weight: 850;
        }
        .emotion-pill strong {
          color: var(--coral);
          font-size: 0.9rem;
        }
        .emotion-pill.empty {
          background: #f4f7fa;
          border-color: var(--line);
          color: var(--muted);
        }
        .confidence {
          color: var(--muted);
          font-size: 0.9rem;
          margin-top: 6px;
        }
        .cue-list {
          display: grid;
          gap: 18px;
        }
        .cue-section {
          border-top: 1px solid var(--line);
          padding-top: 16px;
        }
        .cue-title {
          align-items: center;
          display: flex;
          gap: 10px;
          margin-bottom: 10px;
        }
        .cue-title span {
          align-items: center;
          background: var(--teal);
          border-radius: 999px;
          color: #ffffff;
          display: inline-flex;
          font-size: 0.78rem;
          font-weight: 850;
          height: 26px;
          justify-content: center;
          width: 34px;
        }
        .cue-title strong {
          color: var(--ink);
          font-size: 1rem;
          line-height: 1.25;
        }
        .headline {
          border-left: 5px solid var(--coral);
          color: var(--ink);
          font-size: 1rem;
          font-weight: 750;
          line-height: 1.35;
          margin: 10px 0 14px;
          padding-left: 14px;
        }
        .cue-grid {
          display: grid;
          gap: 12px;
          grid-template-columns: repeat(2, minmax(0, 1fr));
          margin-bottom: 14px;
        }
        .cue-grid div {
          padding: 0;
        }
        .cue-grid span {
          color: var(--teal);
          display: block;
          font-size: 0.75rem;
          font-weight: 800;
          margin-bottom: 6px;
          text-transform: uppercase;
        }
        .cue-grid p {
          color: var(--ink);
          font-size: 0.92rem;
          line-height: 1.42;
          margin: 0;
        }
        .suggestion {
          background: #fff8e7;
          border: 1px solid #f4dfaa;
          border-radius: 8px;
          color: #3e3320;
          font-size: 0.98rem;
          line-height: 1.45;
          padding: 14px;
        }
        .history-row {
          border-bottom: 1px solid var(--line);
          padding: 10px 0;
        }
        .history-muted {
          color: var(--muted);
          font-size: 0.84rem;
          margin-top: 3px;
        }
        video {
          border: 1px solid var(--line);
          border-radius: 8px;
          box-shadow: 0 12px 30px rgba(22, 38, 51, 0.07);
        }
        [data-testid="stSidebar"] {
          background: #ffffff;
          border-right: 1px solid var(--line);
        }
        @media (max-width: 900px) {
          .coach-panel {
            min-height: 0;
          }
          .cue-grid {
            grid-template-columns: 1fr;
          }
          .robot-avatar {
            height: 104px;
            width: 104px;
          }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
