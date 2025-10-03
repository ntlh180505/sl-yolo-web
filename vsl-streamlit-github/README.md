# VSL YOLO Web (Streamlit) — GitHub → Deploy

Hai tính năng chính:
- Webcam realtime + sub (smoothing)
- Upload video → tạo .SRT và (nếu có ffmpeg) video có sub

## Repo files
- `streamlit_app.py` — ứng dụng Streamlit
- `requirements.txt` — thư viện Python
- `packages.txt` — cài ffmpeg (dùng cho Hugging Face Spaces)
- `labels_map.json` — ánh xạ gloss → tiếng Việt
- `.gitignore` — tránh commit model
- `README.md` — hướng dẫn

## Deploy options
### A) Streamlit Community Cloud
1. Push repo này lên GitHub.
2. New app → chọn repo & branch → file: `streamlit_app.py`.
3. **Secrets**:
```
MODEL_URL = https://.../your_model.pt
MODEL_PATH = best.pt
```
4. Deploy (nếu không có ffmpeg, app vẫn xuất `.srt`; muốn video có sub → dùng Spaces).

### B) Hugging Face Spaces (khuyên dùng để có video burn sub)
1. Tạo Space (SDK: Streamlit), kết nối GitHub hoặc upload file.
2. Chọn Hardware: CPU/T4.
3. **Secrets**:
```
MODEL_URL = https://.../your_model.pt
MODEL_PATH = best.pt
```
4. Spaces sẽ cài `ffmpeg` dựa trên `packages.txt` → nút “video có sub” sẽ khả dụng.

## Local
```bash
pip install -r requirements.txt
export MODEL_URL="https://.../your_model.pt"
export MODEL_PATH="best.pt"
streamlit run streamlit_app.py
```

## Tips
- Model nhỏ (yolov8n/s), imgsz 512–640.
- Smoothing 0.5–1.0s cho webcam.
- Video dài → sample fps 8–10.
