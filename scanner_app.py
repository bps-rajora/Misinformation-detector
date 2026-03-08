"""
Auto-Tracking Deepfake Scanner (EfficientNetV2-S Edition)
A transparent overlay that automatically detects and tracks faces on screen,
then analyzes them using an EfficientNetV2-S architecture.
"""

import sys
import os
import warnings
warnings.filterwarnings("ignore")

# IMPORTANT: Import torch BEFORE PyQt6 to avoid DLL conflict on Windows
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from transformers import pipeline # Fallback

import cv2
import numpy as np
from mss import mss
from PIL import Image
from PyQt6.QtWidgets import QApplication, QMainWindow, QLabel
from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal, QPoint
from PyQt6.QtGui import QPainter, QColor, QPen, QFont

# MediaPipe tasks API
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class GlobalScannerThread(QThread):
    """Background thread for global screen face detection."""
    
    face_detected = pyqtSignal(int, int, int, int)  # x, y, w, h in screen coords
    no_face = pyqtSignal()
    
    def __init__(self):
        super().__init__()
        self.running = True
        self.scale_factor = 0.5  # Downscale for faster processing
        
        # Initialize MediaPipe Face Detection
        base_options = python.BaseOptions(
            model_asset_path=self._get_face_detector_model()
        )
        options = vision.FaceDetectorOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            min_detection_confidence=0.5
        )
        self.face_detector = vision.FaceDetector.create_from_options(options)
        
    def _get_face_detector_model(self):
        """Download and cache the face detector model."""
        import urllib.request
        model_path = os.path.join(os.path.dirname(__file__), "blaze_face_short_range.tflite")
        if not os.path.exists(model_path):
            print("Downloading face detector model...")
            url = "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
            urllib.request.urlretrieve(url, model_path)
        return model_path
        
    def run(self):
        with mss() as sct:
            monitor = sct.monitors[1]
            while self.running:
                try:
                    screenshot = sct.grab(monitor)
                    img = np.array(screenshot)
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
                    
                    height, width = img_rgb.shape[:2]
                    small_img = cv2.resize(img_rgb, (int(width * self.scale_factor), int(height * self.scale_factor)))
                    
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=small_img)
                    results = self.face_detector.detect(mp_image)
                    
                    if results.detections:
                        detection = results.detections[0]
                        bbox = detection.bounding_box
                        
                        x = int(bbox.origin_x / self.scale_factor)
                        y = int(bbox.origin_y / self.scale_factor)
                        w = int(bbox.width / self.scale_factor)
                        h = int(bbox.height / self.scale_factor)
                        
                        x += monitor["left"]
                        y += monitor["top"]
                        
                        padding = int(min(w, h) * 0.3)
                        x = max(0, x - padding)
                        y = max(0, y - padding)
                        w = w + padding * 2
                        h = h + padding * 2
                        
                        self.face_detected.emit(x, y, w, h)
                    else:
                        self.no_face.emit()
                except Exception as e:
                    pass # Silently ignore grab errors
                self.msleep(200)

    def stop(self):
        self.running = False
        self.wait()


class AutoScanner(QMainWindow):
    """Main application window."""
    
    STATE_SEARCHING = "SEARCHING"
    STATE_LOCKED = "LOCKED"
    
    def __init__(self):
        super().__init__()
        
        self.window_size = 300
        self.border_width = 4
        self.corner_length = 30
        
        self.current_x = 0
        self.current_y = 0
        self.target_x = 0
        self.target_y = 0
        self.state = self.STATE_SEARCHING
        self.trust_score = 0
        self.manual_mode = False
        self.dragging = False
        self.drag_offset = QPoint()
        
        self.color_searching = QColor(128, 128, 128)
        self.color_real = QColor(0, 255, 0)
        self.color_fake = QColor(255, 0, 0)
        self.current_color = self.color_searching
        
        self.init_models()
        self.init_ui()
        
        self.scanner_thread = GlobalScannerThread()
        self.scanner_thread.face_detected.connect(self.on_face_detected)
        self.scanner_thread.no_face.connect(self.on_no_face)
        self.scanner_thread.start()
        
        self.tracking_timer = QTimer()
        self.tracking_timer.timeout.connect(self.update_position)
        self.tracking_timer.start(16)
        
        self.analysis_timer = QTimer()
        self.analysis_timer.timeout.connect(self.analyze_face)
        self.analysis_timer.start(500)
        
    def init_models(self):
        """Initialize EfficientNetV2-S + MediaPipe."""
        print("⚡ Loading AI Models...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # --- 1. EfficientNetV2-S Setup ---
        try:
            # Load the Architecture
            self.model = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.DEFAULT)
            
            # Modify Head for Binary Classification (Real vs Fake)
            # Default is 1000 classes; we change it to 2.
            in_features = self.model.classifier[1].in_features
            self.model.classifier[1] = nn.Linear(in_features, 2)
            
            # --- CUSTOM WEIGHTS LOADING ---
            weights_path = os.path.join(os.path.dirname(__file__), "weight.pth")
            if os.path.exists(weights_path):
                self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
                print(f"✓ Loaded Custom Weights: {weights_path}")
            else:
                print("⚠️ No custom weights found. Using ImageNet weights (For Demo/Structure Only).")
            
            self.model.to(self.device)
            self.model.eval()

            # Preprocessing required by EfficientNet
            self.preprocess = transforms.Compose([
                transforms.Resize(384),
                transforms.CenterCrop(384),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            self.use_efficientnet = True
            print("✓ EfficientNetV2-S Loaded")

        except Exception as e:
            print(f"❌ EfficientNet Failed: {e}")
            print("   -> Fallback to HF Transformer...")
            self.use_efficientnet = False
            # Fallback to the dima806 model if PyTorch fails or is missing weights
            try:
                self.fallback_pipe = pipeline("image-classification", model="dima806/deepfake_vs_real_image_detection")
            except:
                self.fallback_pipe = None

        # --- 2. MediaPipe FaceMesh Setup ---
        self._init_face_mesh()
        
    def _init_face_mesh(self):
        try:
            model_path = os.path.join(os.path.dirname(__file__), "face_landmarker.task")
            if not os.path.exists(model_path):
                import urllib.request
                url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
                urllib.request.urlretrieve(url, model_path)
            
            base_options = python.BaseOptions(model_asset_path=model_path)
            options = vision.FaceLandmarkerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.IMAGE,
                num_faces=1,
                min_face_detection_confidence=0.5
            )
            self.face_landmarker = vision.FaceLandmarker.create_from_options(options)
        except Exception as e:
            print(f"Face Mesh Error: {e}")
            self.face_landmarker = None

    def init_ui(self):
        # UI Setup (Transparent Window)
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint | Qt.WindowType.Tool)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        
        screen = QApplication.primaryScreen().geometry()
        self.current_x = (screen.width() - self.window_size) // 2
        self.current_y = (screen.height() - self.window_size) // 2
        self.target_x = self.current_x
        self.target_y = self.current_y
        self.setGeometry(self.current_x, self.current_y, self.window_size, self.window_size)
        
        self.status_label = QLabel("SEARCHING...", self)
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setStyleSheet("color: white; background-color: rgba(0,0,0,150); border-radius: 5px; font-weight: bold;")
        self.status_label.setFixedWidth(150)
        self.status_label.move((self.window_size - 150) // 2, self.window_size - 35)
        
        self.mode_label = QLabel("AUTO", self)
        self.mode_label.setStyleSheet("color: #00ff00; background-color: rgba(0,0,0,150); border-radius: 3px; font-weight: bold;")
        self.mode_label.move(10, 10)

    def analyze_face(self):
        if self.state != self.STATE_LOCKED: return
        
        try:
            # 1. Capture Face
            with mss() as sct:
                monitor = {"left": self.x(), "top": self.y(), "width": self.window_size, "height": self.window_size}
                screenshot = sct.grab(monitor)
            img = Image.frombytes("RGB", screenshot.size, screenshot.bgra, "raw", "BGRX")

            # 2. Inference (EfficientNet OR Fallback)
            real_score = 0.5
            
            if self.use_efficientnet:
                # EfficientNet Path
                input_tensor = self.preprocess(img).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    logits = self.model(input_tensor)
                    probs = F.softmax(logits, dim=1)
                    # Assuming Class 1 is Real, Class 0 is Fake. Swap if trained differently.
                    fake_prob = probs[0][0].item()
                    real_prob = probs[0][1].item()
                    real_score = real_prob
            elif self.fallback_pipe:
                # Hugging Face Path
                results = self.fallback_pipe(img)
                for res in results:
                    if "real" in res["label"].lower(): real_score = res["score"]
                    elif "fake" in res["label"].lower(): real_score = 1.0 - res["score"]

            self.trust_score = int(real_score * 100)

            # 3. Geometry Check (MediaPipe)
            if self.face_landmarker:
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.array(img))
                mesh = self.face_landmarker.detect(mp_img)
                if mesh.face_landmarks:
                    geo_conf = self.check_landmark_consistency(mesh.face_landmarks[0])
                    # Combine: 70% Visual, 30% Geometry
                    self.trust_score = int((self.trust_score * 0.7) + (geo_conf * 0.3))

            # 4. Update UI
            if self.trust_score >= 50:
                self.current_color = self.color_real
                self.status_label.setText(f"REAL: {self.trust_score}%")
            else:
                self.current_color = self.color_fake
                self.status_label.setText(f"FAKE: {100 - self.trust_score}%")
            self.update()
            
        except Exception as e:
            print(f"Analysis Error: {e}")

    def check_landmark_consistency(self, landmarks):
        if len(landmarks) < 468: return 50
        nose = landmarks[1]; left_eye = landmarks[33]; right_eye = landmarks[263]
        eye_dist = abs(right_eye.x - left_eye.x)
        nose_h = abs((left_eye.y + right_eye.y)/2 - nose.y)
        if eye_dist > 0:
            ratio = nose_h / eye_dist
            if 0.2 < ratio < 0.6: return 80
        return 40

    # --- Boilerplate UI Events ---
    def paintEvent(self, event):
        painter = QPainter(self); painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        pen = QPen(self.current_color, self.border_width); painter.setPen(pen)
        w = self.window_size; h = self.window_size; c = self.corner_length; b = self.border_width // 2
        # Draw Corners
        painter.drawLine(b, b, b+c, b); painter.drawLine(b, b, b, b+c) # TL
        painter.drawLine(w-b-c, b, w-b, b); painter.drawLine(w-b, b, w-b, b+c) # TR
        painter.drawLine(b, h-b, b+c, h-b); painter.drawLine(b, h-b-c, b, h-b) # BL
        painter.drawLine(w-b-c, h-b, w-b, h-b); painter.drawLine(w-b, h-b-c, w-b, h-b) # BR
        # Crosshair
        center = w // 2; cs = 10
        painter.drawLine(center-cs, center, center+cs, center)
        painter.drawLine(center, center-cs, center, center+cs)

    def on_face_detected(self, x, y, w, h):
        if self.manual_mode: return
        self.target_x = x + (w - self.window_size)//2
        self.target_y = y + (h - self.window_size)//2
        self.state = self.STATE_LOCKED

    def on_no_face(self):
        if self.manual_mode: return
        self.state = self.STATE_SEARCHING; self.trust_score = 0; self.current_color = self.color_searching
        self.status_label.setText("SEARCHING..."); self.update()

    def update_position(self):
        if self.manual_mode or self.dragging: return
        self.current_x = int(self.current_x * 0.7 + self.target_x * 0.3)
        self.current_y = int(self.current_y * 0.7 + self.target_y * 0.3)
        self.move(self.current_x, self.current_y)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_M:
            self.manual_mode = not self.manual_mode
            self.mode_label.setText("MANUAL" if self.manual_mode else "AUTO")
            self.mode_label.setStyleSheet(f"color: {'#ffaa00' if self.manual_mode else '#00ff00'}; background-color: rgba(0,0,0,150); border-radius: 3px;")
        elif event.key() == Qt.Key.Key_Escape: self.close()
        elif event.key() in [Qt.Key.Key_Plus, Qt.Key.Key_Equal]:
            self.window_size = min(500, self.window_size + 20); self.resize(self.window_size, self.window_size)
        elif event.key() == Qt.Key.Key_Minus:
            self.window_size = max(150, self.window_size - 20); self.resize(self.window_size, self.window_size)
        self.status_label.move((self.window_size - 150) // 2, self.window_size - 35)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self.manual_mode:
            self.dragging = True; self.drag_offset = event.pos()
    def mouseMoveEvent(self, event):
        if self.dragging and self.manual_mode:
            self.move(event.globalPosition().toPoint() - self.drag_offset)
    def mouseReleaseEvent(self, event): self.dragging = False
    def closeEvent(self, event):
        self.scanner_thread.stop(); self.tracking_timer.stop(); self.analysis_timer.stop(); event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    font = QFont("Segoe UI", 10); app.setFont(font)
    scanner = AutoScanner()
    scanner.show()
    sys.exit(app.exec())