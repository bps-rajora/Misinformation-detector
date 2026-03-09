import streamlit as st
import feedparser
import urllib.parse
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os
import tempfile
import gc

# --- CONFIGURATION ---
ST_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_HALF = torch.cuda.is_available()  # float16 only on GPU (slower on CPU)

# --- LAZY MODEL LOADERS (load only when needed, not all at once) ---

@st.cache_resource
def load_deepfake_model():
    """Load the EfficientNetV2-S classifier in float16 to save memory."""
    try:
        import timm
        model = timm.create_model('tf_efficientnetv2_s', num_classes=5)
        model.classifier = nn.Sequential(
            nn.Linear(model.classifier.in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 5)
        )
        weights_path = os.path.join(os.path.dirname(__file__), "weight.pth")
        if os.path.exists(weights_path):
            state_dict = torch.load(weights_path, map_location=ST_DEVICE)
            new_state_dict = {k.replace("base_model.", ""): v for k, v in state_dict.items()}
            model.load_state_dict(new_state_dict)
            if USE_HALF:
                model.half()
            model.eval().to(ST_DEVICE)
            return model, True
        return None, False
    except Exception as e:
        return None, False

@st.cache_resource
def load_whisper_model():
    """Load Whisper tiny model — uses ~40MB vs 150MB for base."""
    try:
        import whisper
        return whisper.load_model("tiny", device=ST_DEVICE)
    except Exception:
        return None

@st.cache_resource
def load_sentence_transformer():
    """Load a lightweight sentence model for fact-checking."""
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_resource
def get_face_detector():
    """Initialize MediaPipe Face Detector."""
    model_path = os.path.join(os.path.dirname(__file__), "blaze_face_short_range.tflite")
    if not os.path.exists(model_path):
        import urllib.request
        url = "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
        urllib.request.urlretrieve(url, model_path)
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.FaceDetectorOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        min_detection_confidence=0.5
    )
    return vision.FaceDetector.create_from_options(options)

# --- UTILITY FUNCTIONS ---

def verify_claim(claim, sentence_model):
    """Search Google News RSS and compare semantic similarity."""
    from sentence_transformers import util
    encoded_query = urllib.parse.quote(claim[:200])
    url = f"https://news.google.com/rss/search?q={encoded_query}"
    feed = feedparser.parse(url)
    articles = [entry.title for entry in feed.entries[:5]]
    if not articles:
        return 0, []
    claim_map = sentence_model.encode([claim] + articles)
    scores = util.cos_sim(claim_map[0], claim_map[1:])
    return float(torch.max(scores)), articles

def preprocess_face(face_img):
    """Preprocess a face crop for EfficientNetV2-S inference."""
    preprocess = transforms.Compose([
        transforms.Resize(384),
        transforms.CenterCrop(384),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    pil_img = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
    t = preprocess(pil_img).unsqueeze(0).to(ST_DEVICE)
    return t.half() if USE_HALF else t

def analyze_visual(frame, model, face_detector):
    """Detect faces in a frame and classify each as real/fake."""
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    results = face_detector.detect(mp_image)
    if not results.detections:
        return None

    scores = []
    for d in results.detections:
        bbox = d.bounding_box
        x, y, w, h = int(bbox.origin_x), int(bbox.origin_y), int(bbox.width), int(bbox.height)
        pad = int(min(w, h) * 0.3)
        face = frame[max(0, y - pad):y + h + pad, max(0, x - pad):x + w + pad]
        if face.size == 0:
            continue
        input_t = preprocess_face(face)
        with torch.no_grad():
            probs = F.softmax(model(input_t).float(), dim=1)  # Cast back to float32 for softmax
            scores.append(probs[0][0].item())  # Class 0 = Real
        del input_t  # Free memory immediately
    gc.collect()
    return np.mean(scores) if scores else 0.5

# --- STREAMLIT UI ---

st.set_page_config(page_title="Shodh AI - Unified Verifier", layout="wide", page_icon="🛡️")
st.title("🛡️ Shodh AI: All-in-One Verifier")
st.markdown("Upload any **Video**, **Audio** to scan for Deepfakes and Fact-Check claims.")

# Unified Uploader
uploaded_file = st.file_uploader(
    "Drop your media here",
    type=["mp4", "wav", "mp3", "avi", "mkv", "webm", "ogg", "flac"]
)

# Only load models on demand, show status in sidebar
with st.sidebar:
    st.header("⚙️ System Status")
    st.caption("Models load on first use to save memory.")
    st.divider()
    st.caption("Built with ❤️ by Shodh AI")

if uploaded_file:
    file_type = uploaded_file.type.split('/')[0]
    ext = uploaded_file.name.split('.')[-1].lower()

    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    st.divider()
    col_pre, col_res = st.columns([1, 1])

    with col_pre:
        st.subheader("📂 Media Preview")
        if file_type == 'image':
            st.image(uploaded_file, use_container_width=True)
        elif file_type == 'video':
            st.video(uploaded_file)
        elif file_type == 'audio':
            st.audio(uploaded_file)

    with col_res:
        st.subheader("🔬 AI Analysis Report")

        if st.button("🚀 Run Full Security Scan", type="primary"):
            vis_score, aud_score, transcript = None, None, None

            # 1. Visual Analysis (Image or Video) — loads deepfake model on demand
            if file_type in ['image', 'video']:
                with st.spinner("🔍 Loading Visual AI & Analyzing..."):
                    df_model, df_ok = load_deepfake_model()
                    if df_ok:
                        face_det = get_face_detector()
                        if file_type == 'image':
                            img = cv2.imread(tmp_path)
                            if img is not None:
                                vis_score = analyze_visual(img, df_model, face_det)
                        else:
                            cap = cv2.VideoCapture(tmp_path)
                            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                            frame_scores = []
                            sample_count = min(3, total)  # 3 frames is enough for a quick scan
                            progress = st.progress(0, text="Scanning video frames...")
                            indices = [int(i) for i in np.linspace(0, total - 1, sample_count)]
                            for idx, i in enumerate(indices):
                                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                                ret, frame = cap.read()
                                if ret:
                                    s = analyze_visual(frame, df_model, face_det)
                                    if s is not None:
                                        frame_scores.append(s)
                                progress.progress((idx + 1) / len(indices), text=f"Frame {idx+1}/{len(indices)}")
                            cap.release()
                            progress.empty()
                            vis_score = np.mean(frame_scores) if frame_scores else None
                    gc.collect()

            # 2. Audio & Speech (Video or Audio) — loads whisper on demand
            if file_type in ['video', 'audio']:
                with st.spinner("🎧 Processing Audio & Speech..."):
                    audio_path = tmp_path
                    if file_type == 'video':
                        try:
                            from moviepy import VideoFileClip
                            with VideoFileClip(tmp_path) as video:
                                if video.audio:
                                    audio_path = tmp_path.rsplit('.', 1)[0] + ".wav"
                                    video.audio.write_audiofile(audio_path, logger=None)
                                else:
                                    audio_path = None
                        except Exception:
                            audio_path = None

                    if audio_path and os.path.exists(audio_path):
                        try:
                            import librosa
                            audio_arr, _ = librosa.load(audio_path, sr=16000)

                            # Speech-to-text (Whisper tiny)
                            whisper_model = load_whisper_model()
                            if whisper_model:
                                transcript = whisper_model.transcribe(audio_arr)["text"]
                            del audio_arr
                            gc.collect()
                        except Exception:
                            pass

            # Save results
            st.session_state['v_score'] = vis_score
            st.session_state['a_score'] = aud_score
            st.session_state['script'] = transcript
            st.session_state.pop('fact_score', None)
            st.session_state.pop('fact_articles', None)

        # --- Display Results ---
        if 'v_score' in st.session_state and st.session_state['v_score'] is not None:
            st.metric("Visual Trust Score", f"{st.session_state['v_score']:.2%}")
            if st.session_state['v_score'] < 0.5:
                st.error("🚩 Visual Manipulation Detected!")
            else:
                st.success("✅ Real-Source Video Match")

        if 'script' in st.session_state and st.session_state['script']:
            st.subheader("📝 Speech-to-Text Verification")
            edited_script = st.text_area(
                "Verify and edit the detected speech before fact-checking:",
                value=st.session_state['script'],
                height=150,
                help="The AI transcription may have errors. Edit before running fact-check."
            )

            if st.button("🔍 Fact-Check this Script"):
                with st.spinner("Searching Global News Sources..."):
                    sentence_model = load_sentence_transformer()
                    score, articles = verify_claim(edited_script, sentence_model)
                    st.session_state['fact_score'] = score
                    st.session_state['fact_articles'] = articles
                    st.session_state['script'] = edited_script

        if 'fact_score' in st.session_state:
            st.metric("News Correlation", f"{st.session_state['fact_score']:.2%}")
            if st.session_state['fact_score'] > 0.7:
                st.success("✅ High correlation with verified news sources.")
            elif st.session_state['fact_score'] > 0.3:
                st.warning("⚠️ Partial match found. Verify manually.")
            else:
                st.error("🚩 No reliable news match found. Possible misinformation.")
            if st.session_state.get('fact_articles'):
                st.write("**Related Articles:**")
                for a in st.session_state['fact_articles']:
                    st.write(f"- {a}")

    try:
        os.unlink(tmp_path)
    except Exception:
        pass

st.divider()
st.subheader("🔍 Quick Claim Fact-Checker")
manual_claim = st.text_input("Type a claim to verify manually:")
if manual_claim:
    with st.spinner("Checking..."):
        sentence_model = load_sentence_transformer()
        s, a = verify_claim(manual_claim, sentence_model)
        st.metric("Truth Score", f"{s:.2%}")
        if a:
            st.write("**Related Articles:**")
            for item in a:
                st.write(f"• {item}")
        else:
            st.info("No matching news articles found.")
