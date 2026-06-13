"""
TrafficSense - Real-Time Traffic Density Dashboard

An interactive Streamlit app that:
- Lets the user upload a traffic video
- Runs YOLOv8 vehicle detection + counting + density classification
- Shows a live preview while processing
- Displays an analytics dashboard (charts, metrics, density gauge)
- Lets the user download the annotated video and the CSV log
"""

import os
import subprocess
import tempfile

# Fix: set YOLO config dir to writable location on Streamlit Cloud
os.environ["YOLO_CONFIG_DIR"] = "/tmp/Ultralytics"

import cv2
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from detector import TrafficDetector

st.set_page_config(page_title="TrafficSense", page_icon="🚦", layout="wide")

# ---------------------------------------------------------------- styling
st.markdown("""
<style>
.hero {
    background: linear-gradient(135deg, #1f2937 0%, #374151 100%);
    padding: 2rem 2.2rem;
    border-radius: 16px;
    margin-bottom: 1.8rem;
}
.hero h1 {
    color: #ffffff; font-size: 2.3rem; font-weight: 800;
    margin: 0 0 0.4rem 0; letter-spacing: -0.5px;
}
.hero p {
    color: #d1d5db; font-size: 1.02rem; margin: 0; max-width: 800px;
}
.hero .badges { margin-top: 0.9rem; }
.hero .badge {
    display: inline-block; background: rgba(255,255,255,0.12);
    color: #e5e7eb; padding: 4px 12px; border-radius: 999px;
    font-size: 0.8rem; margin-right: 8px; font-weight: 600;
}
.density-pill {
    display: inline-block; padding: 6px 18px; border-radius: 999px;
    font-weight: 700; font-size: 1.1rem;
}
.density-LOW { background-color: #d4f7dc; color: #1a7f37; }
.density-MEDIUM { background-color: #fff3cd; color: #946200; }
.density-HIGH { background-color: #fdd; color: #b3261e; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <h1>🚦 TrafficSense</h1>
  <p>Real-time vehicle detection &amp; traffic density estimation using YOLOv8 —
  built to support adaptive signal control on Bengaluru's fixed-timer junctions.</p>
  <div class="badges">
    <span class="badge">YOLOv8</span>
    <span class="badge">OpenCV</span>
    <span class="badge">Real-time tracking</span>
    <span class="badge">Live analytics</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    video_file = st.file_uploader("Upload traffic video", type=["mp4", "avi", "mov", "mkv"])
    conf = st.slider("Detection confidence threshold", 0.1, 0.9, 0.4, 0.05)
    line_pos = st.slider("Counting line position (% of width)", 0.1, 0.9, 0.5, 0.05)
    max_frames = st.slider("Frames to process (demo limit)", 50, 600, 200, 50,
                            help="Lower = faster preview on CPU. Use process_video.py for full videos.")
    run_btn = st.button("▶️  Run detection", type="primary", width='stretch')

    st.markdown("---")
    st.markdown("**Density thresholds**")
    st.markdown("🟢 Low — 0 to 5 vehicles\n\n🟡 Medium — 6 to 15 vehicles\n\n🔴 High — 16+ vehicles")

    st.markdown("---")
    st.caption("Built with YOLOv8, OpenCV, Streamlit & Plotly")

# ---------------------------------------------------------------- state
if "results_df" not in st.session_state:
    st.session_state.results_df = None
if "output_video" not in st.session_state:
    st.session_state.output_video = None

# ---------------------------------------------------------------- processing
if run_btn:
    if video_file is None:
        st.warning("Please upload a video first.")
    else:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(video_file.read())
        input_path = tfile.name

        raw_output_path = os.path.join(tempfile.gettempdir(), "traffic_raw.avi")
        output_path = os.path.join(tempfile.gettempdir(), "traffic_output.mp4")

        with st.spinner("Loading YOLOv8 model..."):
            detector = TrafficDetector(conf_threshold=conf)

        cap = cv2.VideoCapture(input_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        line_x = int(width * line_pos)

        # write with a codec OpenCV always supports, then re-encode with
        # ffmpeg to H.264 so it plays in the browser <video> element
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        out = cv2.VideoWriter(raw_output_path, fourcc, fps, (width, height))

        frame_placeholder = st.empty()
        progress_bar = st.progress(0, text="Processing video...")
        log_rows = []

        frame_idx = 0
        while frame_idx < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            annotated, stats = detector.process_frame(frame, line_x=line_x)
            out.write(annotated)

            if frame_idx % 5 == 0:
                frame_placeholder.image(
                    cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                    channels="RGB", width='stretch',
                )

            log_rows.append({
                "time_sec": round(frame_idx / fps, 2),
                "current_count": stats["current_count"],
                "total_crossed": stats["total_count"],
                "density": stats["density"],
                "car": stats["car"],
                "motorcycle": stats["motorcycle"],
                "bus": stats["bus"],
                "truck": stats["truck"],
            })

            frame_idx += 1
            progress_bar.progress(frame_idx / max_frames,
                                   text=f"Processing video... frame {frame_idx}/{max_frames}")

        cap.release()
        out.release()
        try:
            os.unlink(input_path)
        except PermissionError:
            pass
        progress_bar.empty()
        frame_placeholder.empty()

        # re-encode to H.264 mp4 so it plays in the browser
        with st.spinner("Finalizing video..."):
            ffmpeg_ok = (
                subprocess.run(
                    ["ffmpeg", "-y", "-i", raw_output_path,
                     "-c:v", "libx264", "-pix_fmt", "yuv420p",
                     "-movflags", "+faststart", output_path],
                    capture_output=True,
                ).returncode == 0
            )
            if not ffmpeg_ok:
                # ffmpeg not available - fall back to the raw (downloadable
                # but maybe not browser-playable) file
                output_path = raw_output_path

        st.session_state.results_df = pd.DataFrame(log_rows)
        st.session_state.output_video = output_path
        st.success(f"Processed {frame_idx} frames ✅")

# ---------------------------------------------------------------- dashboard
df = st.session_state.results_df

if df is not None and len(df) > 0:
    latest = df.iloc[-1]

    # ---- top metric row
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Current vehicles", int(latest["current_count"]))
    c2.metric("Total crossed", int(latest["total_crossed"]))
    c3.metric("Peak vehicles (any frame)", int(df["current_count"].max()))
    with c4:
        st.markdown("**Current density**")
        st.markdown(
            f'<span class="density-pill density-{latest["density"]}">{latest["density"]}</span>',
            unsafe_allow_html=True,
        )

    st.markdown("### 📊 Live analytics")

    # ---- row 1: count over time + density gauge
    col1, col2 = st.columns([2, 1])

    with col1:
        fig = px.line(df, x="time_sec", y="current_count", markers=True,
                       title="Vehicle count over time",
                       labels={"time_sec": "Time (s)", "current_count": "Vehicles in frame"})
        fig.add_hline(y=5, line_dash="dot", line_color="green",
                      annotation_text="Low/Medium boundary")
        fig.add_hline(y=15, line_dash="dot", line_color="red",
                      annotation_text="Medium/High boundary")
        fig.update_layout(height=340, margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig4 = go.Figure(go.Indicator(
            mode="gauge+number",
            value=int(latest["current_count"]),
            title={"text": "Current density"},
            gauge={
                "axis": {"range": [0, max(20, df["current_count"].max())]},
                "steps": [
                    {"range": [0, 5], "color": "#d4f7dc"},
                    {"range": [5, 15], "color": "#fff3cd"},
                    {"range": [15, max(20, df["current_count"].max())], "color": "#fdd"},
                ],
                "bar": {"color": "#333"},
            },
        ))
        fig4.update_layout(height=340, margin=dict(l=20, r=20, t=50, b=10))
        st.plotly_chart(fig4, use_container_width=True)

    # ---- row 2: vehicle type pie + density distribution pie
    col3, col4 = st.columns(2)

    with col3:
        type_totals = df[["car", "motorcycle", "bus", "truck"]].sum()
        fig2 = px.pie(values=type_totals.values, names=type_totals.index,
                      title="Vehicle type breakdown",
                      hole=0.45,
                      color=type_totals.index)
        fig2.update_traces(textinfo="percent+label")
        fig2.update_layout(height=340, margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(fig2, use_container_width=True)

    with col4:
        density_counts = df["density"].value_counts().reindex(["LOW", "MEDIUM", "HIGH"]).fillna(0)
        fig3 = px.pie(values=density_counts.values, names=density_counts.index,
                      title="Time spent in each density level",
                      hole=0.45,
                      color=density_counts.index,
                      color_discrete_map={"LOW": "#1a7f37", "MEDIUM": "#946200", "HIGH": "#b3261e"})
        fig3.update_traces(textinfo="percent+label")
        fig3.update_layout(height=340, margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(fig3, use_container_width=True)

    # ---- row 3: stacked vehicle type over time
    fig5 = px.area(df, x="time_sec", y=["car", "motorcycle", "bus", "truck"],
                    title="Vehicle type composition over time",
                    labels={"time_sec": "Time (s)", "value": "Count", "variable": "Type"})
    fig5.update_layout(height=320, margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig5, use_container_width=True)

    st.markdown("### 🎥 Video & data")
    tab1, tab2 = st.tabs(["📹 Annotated video", "📄 Raw data"])

    # ---- video tab
    with tab1:
        if st.session_state.output_video and os.path.exists(st.session_state.output_video):
            st.video(st.session_state.output_video)
            with open(st.session_state.output_video, "rb") as f:
                st.download_button("⬇️ Download annotated video", f,
                                    file_name="traffic_output.mp4", width='stretch')

    # ---- raw data tab
    with tab2:
        st.dataframe(df, use_container_width=True)
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download CSV log", csv_bytes,
                            file_name="results.csv", mime="text/csv",
                            width='stretch')

else:
    st.info("👆 Upload a traffic video in the sidebar and click **Run detection** to get started.")
    st.markdown("""
### How it works
1. Upload CCTV-style traffic footage
2. YOLOv8 detects and classifies vehicles (car / motorcycle / bus / truck)
3. A virtual counting line + lightweight tracker counts vehicles exactly once
4. Each frame's vehicle count is classified as 🟢 Low, 🟡 Medium, or 🔴 High
5. View live analytics and download the annotated video + CSV log
""")
