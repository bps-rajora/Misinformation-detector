# 🛡️ Shodh AI: Misinformation & Deepfake Detector

Shodh AI is a unified security dashboard designed to detect digital manipulation in Video, Audio, and Images, and verify spoken content against global news sources.

## 🚀 Features
- **Visual Deepfake Detection**: Uses an EfficientNetV2-S model (Fine-tuned on FaceForensics++) to identify face manipulation.
- **Audio Authenticity Scan**: Detects synthetic/AI-generated voices using Audio Spectrogram Transformers.
- **Speech-to-Text Transcription**: Powered by OpenAI's Whisper for high-accuracy script extraction.
- **News Verifier**: Cross-references transcripts with Google News RSS to calculate a truth/correlation score.
- **Unified Interface**: One-click analysis for all media formats.

## 🛠️ Installation

### 1. Requirements
Ensure you have Python 3.9+ installed.

### 2. Setup Virtual Environment
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Running the Dashboard
```bash
streamlit run shodh_dashboard.py
```

## 📦 Project Structure
- `shodh_dashboard.py`: Main unified Streamlit application.
- `scanner_app.py`: Desktop-version auto-scanner (PyQt6).
- `weight.pth`: Trained weights for the deepfake detection model.
- `blaze_face_short_range.tflite`: MediaPipe model for fast face detection.

## 🛡️ Ethics & Disclaimer
This tool is intended for research and educational purposes. Always verify AI results with official sources.
