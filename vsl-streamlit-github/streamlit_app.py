import os, io, json, time, tempfile, base64, pathlib, subprocess
from typing import List, Tuple, Optional
import numpy as np
import cv2
import streamlit as st

# Optional realtime webcam
try:
    from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoProcessorBase
    import av
    HAS_WEBRTC = True
except Exception:
    HAS_WEBRTC = False

def has_ffmpeg() -> bool:
    try:
        out = subprocess.run(["ffmpeg","-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False, text=True)
        return out.returncode == 0
    except Exception:
        return False

FFMPEG_OK = has_ffmpeg()

MODEL_PATH = os.getenv("MODEL_PATH", "best.pt")
MODEL_URL  = os.getenv("MODEL_URL", "")
if (not pathlib.Path(MODEL_PATH).exists()) and MODEL_URL:
    try:
        subprocess.run(["wget","-q",MODEL_URL,"-O",MODEL_PATH], check=True)
    except Exception:
        pass

from ultralytics import YOLO

st.set_page_config(page_title="VSL YOLO — Web (Streamlit)", layout="wide")
st.title("VSL YOLO — Ảnh • Webcam • Video→Sub")

@st.cache_resource
def load_model(path: str):
    return YOLO(path)

def load_labels_map(path: str = "labels_map.json"):
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}

def apply_label_map(label: Optional[str], labels_map: dict):
    if label is None:
        return None
    return labels_map.get(label, label)

def majority_smooth(seq: List[Optional[str]], k: int = 5) -> List[Optional[str]]:
    out = []
    for i in range(len(seq)):
        w = seq[max(0, i - k): i + 1]
        vals = [x for x in w if x is not None]
        out.append(max(set(vals), key=vals.count) if vals else None)
    return out

def sec2ts(x: float) -> str:
    h = int(x // 3600)
    m = int((x % 3600) // 60)
    s = int(x % 60)
    ms = int((x - int(x)) * 1000)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"

def to_srt(spans: List[Tuple[int, float, float, str]]) -> str:
    lines = []
    for i, (idx, stt, ed, lb) in enumerate(spans, 1):
        lines.append(f"{i}")
        lines.append(f"{sec2ts(stt)} --> {sec2ts(ed)}")
        lines.append(lb if lb else "")
        lines.append("")
    return "\n".join(lines)

def burn_sub_ffmpeg(input_mp4: str, srt_path: str, output_mp4: str):
    cmd = [
        "ffmpeg","-y","-i", input_mp4,
        "-vf", f"subtitles={srt_path}",
        "-c:v","libx264","-c:a","copy","-movflags","+faststart", output_mp4
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

labels_map = load_labels_map("labels_map.json")

st.sidebar.header("Settings")
MODEL_PATH = st.sidebar.text_input("Model path (.pt)", MODEL_PATH)
conf = st.sidebar.slider("Confidence", 0.0, 1.0, 0.25, 0.01)
imgsz = st.sidebar.selectbox("imgsz", [320, 416, 480, 512, 640, 736, 768, 960, 1024], index=4)
sample_fps = st.sidebar.slider("Video sample FPS", 1, 15, 8)

with st.spinner("Loading YOLO model..."):
    model = load_model(MODEL_PATH)
st.success("Model loaded")

with st.expander("Model labels"):
    names = getattr(model, "names", {})
    if isinstance(names, dict):
        st.write([names[k] for k in sorted(names.keys())])
    else:
        st.write(names)

tab_img, tab_cam, tab_vid = st.tabs(["📷 Ảnh", "🎥 Webcam (real-time)", "🎞️ Video → phụ đề / video có sub"])

with tab_img:
    st.subheader("Nhận diện ảnh")
    f = st.file_uploader("Upload ảnh", type=["jpg", "jpeg", "png"])
    if f is not None:
        file_bytes = np.frombuffer(f.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if img is None:
            st.error("Không đọc được ảnh.")
        else:
            res = model.predict(img, conf=conf, imgsz=imgsz, verbose=False)[0]
            annotated = res.plot()
            annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            st.image(annotated, caption="Kết quả", use_column_width=True)

with tab_cam:
    st.subheader("Webcam realtime + sub")
    try:
        from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoProcessorBase
        import av
        class YOLOProcessor(VideoProcessorBase):
            def __init__(self):
                self.conf = conf; self.imgsz = imgsz
                self.history = []; self.labels_map = labels_map
            def recv(self, frame):
                img = frame.to_ndarray(format="bgr24")
                res = model.predict(img, conf=self.conf, imgsz=self.imgsz, verbose=False)[0]
                best = None
                if res.boxes is not None and len(res.boxes)>0:
                    confs = res.boxes.conf.cpu().numpy()
                    i = int(np.argmax(confs))
                    c = int(res.boxes.cls[i].item())
                    best = res.names[c]
                self.history.append(best)
                if len(self.history) > 12: self.history.pop(0)
                smoothed = majority_smooth(self.history, k=4)[-1] if self.history else None
                text = apply_label_map(smoothed, self.labels_map) or ""
                annotated = res.plot()
                h, w = annotated.shape[:2]
                cv2.rectangle(annotated, (0, h-40), (w, h), (0,0,0), -1)
                cv2.putText(annotated, text, (10, h-12), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2, cv2.LINE_AA)
                return av.VideoFrame.from_ndarray(annotated, format="bgr24")
        rtc_cfg = {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
        ctx = webrtc_streamer(key="vsl-webrtc",
                              mode=WebRtcMode.SENDRECV,
                              rtc_configuration=rtc_cfg,
                              media_stream_constraints={"video": True, "audio": False},
                              video_processor_factory=YOLOProcessor)
        if ctx.video_processor:
            ctx.video_processor.conf = conf
            ctx.video_processor.imgsz = imgsz
    except Exception as e:
        st.warning("Cần `streamlit-webrtc` để dùng webcam. Lỗi: "+str(e))

with tab_vid:
    st.subheader("Upload video → tạo phụ đề (.SRT) và video có sub (nếu có ffmpeg)")
    vf = st.file_uploader("Chọn video (.mp4, .mov, .avi)", type=["mp4", "mov", "avi"])
    make_burn = st.checkbox("Xuất video có sub (ffmpeg)", value=False, disabled=(not FFMPEG_OK))
    if not FFMPEG_OK:
        st.info("⚠️ Môi trường hiện tại không có ffmpeg → chỉ tạo SRT. Trên Hugging Face Spaces, thêm 'ffmpeg' vào packages.txt.")
    go = st.button("Bắt đầu xử lý")

    if go and vf is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(vf.read()); tfile.close()
        tmp_video = tfile.name

        cap = cv2.VideoCapture(tmp_video)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        step = max(1, int(fps // sample_fps))
        st.info(f"FPS gốc: {fps:.1f} | Lấy mẫu ~{sample_fps} fps (mỗi {step} khung)")
        progress = st.progress(0)

        labels_timeline: List[Tuple[float, Optional[str]]] = []
        frame_idx = 0
        while True:
            ok, frame = cap.read()
            if not ok: break
            if frame_idx % step == 0:
                res = model.predict(frame, conf=conf, imgsz=imgsz, verbose=False)[0]
                best_label = None
                if res.boxes is not None and len(res.boxes) > 0:
                    confs = res.boxes.conf.cpu().numpy()
                    i = int(np.argmax(confs))
                    c = int(res.boxes.cls[i].item())
                    best_label = res.names[c]
                tsec = frame_idx / fps
                labels_timeline.append((tsec, best_label))
                if total > 0:
                    progress.progress(min(1.0, frame_idx/total))
            frame_idx += 1
        cap.release()

        smooth_labels = majority_smooth([lb for _, lb in labels_timeline], k=4)
        spans = []
        cur = None; start_t = None
        for (t, _), lb in zip(labels_timeline, smooth_labels):
            if lb != cur:
                if cur is not None and start_t is not None:
                    spans.append((len(spans), start_t, t, apply_label_map(cur, labels_map) or ""))
                cur = lb; start_t = t
        if cur is not None and start_t is not None:
            last_t = labels_timeline[-1][0]
            spans.append((len(spans), start_t, last_t, apply_label_map(cur, labels_map) or ""))

        srt_text = to_srt(spans)
        srt_path = tmp_video.replace(".mp4", ".srt")
        with open(srt_path, "w", encoding="utf-8") as f:
            f.write(srt_text)

        st.success(f"Xử lý xong. Tạo {len(spans)} đoạn phụ đề.")
        st.download_button("Tải phụ đề (.srt)",
                           data=srt_text.encode("utf-8"),
                           file_name=(vf.name.rsplit('.',1)[0] if vf.name else 'output') + ".srt",
                           mime="text/plain")

        if make_burn and FFMPEG_OK:
            burned_path = tmp_video.replace(".mp4", "_with_sub.mp4")
            try:
                burn_sub_ffmpeg(tmp_video, srt_path, burned_path)
                with open(burned_path, "rb") as f:
                    st.download_button("Tải video có sub (.mp4)", f, file_name="output_with_sub.mp4", mime="video/mp4")
            except Exception as e:
                st.error("Burn phụ đề thất bại: " + str(e))

# Footer
st.caption("💡 Dùng labels_map.json để ánh xạ gloss → Tiếng Việt. Thiết lập MODEL_URL / MODEL_PATH trong Secrets để auto tải model.")