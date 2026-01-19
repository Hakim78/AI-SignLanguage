"""
S-TRM ASL Recognition - Hugging Face Spaces Demo
Real-time American Sign Language alphabet recognition using webcam.

Author: Hakim
Model: S-TRM (Stateful Tiny Recursive Model)
"""
import gradio as gr
import numpy as np
import torch
import torch.nn as nn
import joblib
import json
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import urllib.request
import os

# CONFIGURATION

MODEL_PATH = "models/trm_best.pt"
CONFIG_PATH = "models/trm_config.json"
SCALER_PATH = "models/trm_scaler.joblib"

# Classes dynamiques (mots/gestes)
DYNAMIC_CLASSES = {"J", "Z", "HELLO", "THANKYOU", "SORRY", "YES", "NO"}

# MODEL DEFINITION (S-TRM) - Exact architecture from trm_model.py

class TinyBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TRM(nn.Module):
    def __init__(
        self,
        input_dim: int = 63,
        num_classes: int = 24,
        latent_dim: int = 64,
        hidden_dim: int = 128,
        n_latent: int = 6,
        T_deep: int = 3,
        deep_supervision: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.n_latent = n_latent
        self.T_deep = T_deep
        self.deep_supervision = deep_supervision
        self.dropout = dropout

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.y_init = nn.Linear(hidden_dim, num_classes)
        self.z_init = nn.Linear(hidden_dim, latent_dim)
        self.state_gate = nn.Sequential(
            nn.Linear(latent_dim * 2, latent_dim),
            nn.Sigmoid(),
        )
        self.z_update = TinyBlock(
            in_dim=hidden_dim + num_classes + latent_dim,
            out_dim=latent_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
        self.y_update = TinyBlock(
            in_dim=num_classes + latent_dim,
            out_dim=num_classes,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

    def blend_state(self, prev_z: torch.Tensor, new_z: torch.Tensor) -> torch.Tensor:
        gate_input = torch.cat([prev_z, new_z], dim=-1)
        gate = self.state_gate(gate_input)
        return gate * new_z + (1 - gate) * prev_z

    def latent_recursion(self, x_emb, y, z, n=None):
        if n is None:
            n = self.n_latent
        y_history = []
        for _ in range(n):
            z_input = torch.cat([x_emb, y, z], dim=-1)
            z = z + self.z_update(z_input)
            y_input = torch.cat([y, z], dim=-1)
            y = y + self.y_update(y_input)
            y_history.append(y)
        return y, z, y_history

    def deep_recursion(self, x_emb, y, z, n=None, T=None):
        if n is None:
            n = self.n_latent
        if T is None:
            T = self.T_deep
        all_y_history = []
        if T > 1:
            with torch.no_grad():
                for _ in range(T - 1):
                    y, z, y_hist = self.latent_recursion(x_emb, y, z, n)
                    if self.deep_supervision:
                        all_y_history.extend([h.detach() for h in y_hist])
        y, z, y_hist = self.latent_recursion(x_emb, y, z, n)
        all_y_history.extend(y_hist)
        return y, z, all_y_history

    def forward(self, x, prev_state=None, return_all_outputs=None, return_state=False):
        if return_all_outputs is None:
            return_all_outputs = self.deep_supervision and self.training
        x_emb = self.input_proj(x)
        y = self.y_init(x_emb)
        z = self.z_init(x_emb)
        if prev_state is not None:
            y_prev, z_prev = prev_state
            z = self.blend_state(z_prev, z)
            y = 0.3 * y_prev + 0.7 * y
        y_final, z_final, all_y = self.deep_recursion(x_emb, y, z)
        state = (y_final.detach(), z_final.detach())
        if return_all_outputs and return_state:
            return y_final, all_y, state
        elif return_all_outputs:
            return y_final, all_y
        elif return_state:
            return y_final, state
        return y_final


# LOAD MODEL

print("[INFO] Loading model...")

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config = json.load(f)

CLASS_NAMES = config["class_names"]
NUM_CLASSES = config["num_classes"]
INPUT_DIM = config.get("input_dim", 63)

model = TRM(
    input_dim=INPUT_DIM,
    num_classes=NUM_CLASSES,
    latent_dim=config.get("latent_dim", 64),
    hidden_dim=config.get("hidden_dim", 128),
    n_latent=config.get("n_latent", 6),
    T_deep=config.get("T_deep", 3),
    dropout=config.get("dropout", 0.1)
)

model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu", weights_only=True))
model.eval()

scaler = joblib.load(SCALER_PATH)

STATIC_INDICES = [i for i, name in enumerate(CLASS_NAMES) if name not in DYNAMIC_CLASSES]
DYNAMIC_INDICES = [i for i, name in enumerate(CLASS_NAMES) if name in DYNAMIC_CLASSES]

# MediaPipe Tasks API
HAND_LANDMARKER_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
HAND_LANDMARKER_MODEL_PATH = "hand_landmarker.task"

if not os.path.exists(HAND_LANDMARKER_MODEL_PATH):
    print("[INFO] Downloading hand landmarker model...")
    urllib.request.urlretrieve(HAND_LANDMARKER_MODEL_URL, HAND_LANDMARKER_MODEL_PATH)

base_options = python.BaseOptions(model_asset_path=HAND_LANDMARKER_MODEL_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
hand_landmarker = vision.HandLandmarker.create_from_options(options)

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17)
]

print(f"[INFO] Model loaded. Classes: {CLASS_NAMES}")


def draw_landmarks(image, hand_landmarks):
    """Draw hand landmarks on image."""
    h, w = image.shape[:2]
    for start_idx, end_idx in HAND_CONNECTIONS:
        start = hand_landmarks[start_idx]
        end = hand_landmarks[end_idx]
        pt1 = (int(start.x * w), int(start.y * h))
        pt2 = (int(end.x * w), int(end.y * h))
        cv2.line(image, pt1, pt2, (0, 255, 0), 2)
    for lm in hand_landmarks:
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(image, (cx, cy), 5, (0, 255, 0), -1)
    return image


def predict(image, mode):
    """Process image and return prediction."""
    if image is None:
        return None, "Waiting for image...", {}

    # Ensure RGB format
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

    # Ensure contiguous array
    image = np.ascontiguousarray(image)

    # MediaPipe detection
    try:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
        results = hand_landmarker.detect(mp_image)
    except Exception as e:
        return image, f"Error: {str(e)}", {}

    annotated = image.copy()

    if not results.hand_landmarks:
        cv2.putText(annotated, "No hand detected", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        return annotated, "No hand detected", {}

    # Get landmarks
    hand_landmarks = results.hand_landmarks[0]
    annotated = draw_landmarks(annotated, hand_landmarks)

    # Extract features
    coords = []
    for lm in hand_landmarks:
        coords.extend([lm.x, lm.y, lm.z])

    features = np.array(coords, dtype=np.float32).reshape(1, -1)
    features_normalized = scaler.transform(features)

    # Inference
    with torch.no_grad():
        x = torch.tensor(features_normalized, dtype=torch.float32)
        logits = model(x)
        probs = torch.softmax(logits, dim=-1).numpy()[0]

    # Apply mode mask
    masked_probs = probs.copy()
    if mode == "SPELLING":
        for idx in DYNAMIC_INDICES:
            masked_probs[idx] = 0.0
    else:
        for idx in STATIC_INDICES:
            masked_probs[idx] = 0.0

    if masked_probs.sum() > 0:
        masked_probs /= masked_probs.sum()

    # Get predictions
    top_indices = np.argsort(masked_probs)[::-1][:5]
    top_results = {CLASS_NAMES[i]: float(masked_probs[i]) for i in top_indices}
    predicted_class = CLASS_NAMES[top_indices[0]]
    confidence = masked_probs[top_indices[0]]

    # Draw prediction
    color = (255, 200, 0) if mode == "SPELLING" else (0, 140, 255)
    cv2.putText(annotated, f"Mode: {mode}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.putText(annotated, f"{predicted_class} ({confidence:.0%})", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    return annotated, f"{predicted_class} ({confidence:.0%})", top_results


# GRADIO INTERFACE

css = """
.main-title {text-align: center; margin-bottom: 20px;}
.prediction-box {font-size: 24px; font-weight: bold; text-align: center; padding: 20px;}
"""

with gr.Blocks(title="S-TRM ASL Recognition") as demo:
    gr.Markdown("""
    # S-TRM ASL Sign Language Recognition

    Real-time American Sign Language recognition using **S-TRM (Stateful Tiny Recursive Model)**.

    **Instructions:**
    1. Click the webcam button to start your camera
    2. Show an ASL sign to the camera
    3. The model will recognize the sign in real-time
    """)

    with gr.Row():
        with gr.Column(scale=1):
            mode = gr.Radio(
                choices=["SPELLING", "WORD"],
                value="SPELLING",
                label="Recognition Mode",
                info="SPELLING = Letters A-Y | WORD = Gestures (HELLO, YES, NO, etc.)"
            )
            webcam = gr.Image(
                sources=["webcam"],
                type="numpy",
                label="Show your hand sign here",
                streaming=True,
                webcam_options=gr.WebcamOptions(mirror=True)
            )

        with gr.Column(scale=1):
            output_image = gr.Image(label="Detection Result", type="numpy")
            prediction = gr.Textbox(label="Prediction")
            confidence_chart = gr.Label(label="Top 5 Predictions", num_top_classes=5)

    webcam.stream(
        fn=predict,
        inputs=[webcam, mode],
        outputs=[output_image, prediction, confidence_chart],
        time_limit=30,
        stream_every=0.1
    )

    gr.Markdown("""
    ---
    ### Technical Details
    - **Model**: S-TRM (~79K parameters) - Recursive latent reasoning architecture
    - **Input**: 21 hand landmarks (63 dimensions) from MediaPipe
    - **Classes**: 31 signs (A-Z letters + HELLO, YES, NO, SORRY, THANKYOU)

    *Developed for IPSSI - 2025*
    """)

if __name__ == "__main__":
    demo.launch()
