# src/demo_trm.py
"""
Real-time demo for S-TRM (Stateful Tiny Recursive Model) ASL recognition.

Features:
    - Stateful recurrence: Model state persists between frames
    - Top-3 confidence visualization with bar graph
    - Hand landmark visualization with convex hull

Usage:
    uv run python -m src.demo_trm

Requirements:
    - Trained S-TRM model (run train_trm.py first)
    - Webcam
"""
from __future__ import annotations
import argparse
import json
import time
from collections import deque, Counter
from pathlib import Path
from typing import Tuple, Optional

import cv2
import numpy as np
import mediapipe as mp
import torch
import joblib

from trm_model import TRM


# Helpers
def put_text(img, text, org, scale=0.7, thickness=2, color=(255, 255, 255)):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def draw_confidence_bars(
    img: np.ndarray,
    probs: np.ndarray,
    class_names: list,
    top_k: int = 3,
    x: int = 10,
    y: int = 130,
    bar_width: int = 150,
    bar_height: int = 20,
    spacing: int = 5,
) -> None:
    """
    Draw horizontal bar chart showing top-K prediction confidences.

    Args:
        img: Image to draw on
        probs: Probability array [num_classes]
        class_names: List of class names
        top_k: Number of top predictions to show
        x, y: Top-left corner of the chart
        bar_width: Maximum width of bars (100% confidence)
        bar_height: Height of each bar
        spacing: Vertical spacing between bars
    """
    # Get top-K indices
    top_indices = np.argsort(probs)[::-1][:top_k]

    # Colors for top-3 (green, yellow, orange)
    colors = [
        (0, 255, 0),    # Green for top-1
        (0, 255, 255),  # Yellow for top-2
        (0, 165, 255),  # Orange for top-3
    ]

    for i, idx in enumerate(top_indices):
        prob = probs[idx]
        name = class_names[idx] if idx < len(class_names) else f"C{idx}"
        color = colors[i] if i < len(colors) else (200, 200, 200)

        # Bar position
        bar_y = y + i * (bar_height + spacing)

        # Background bar (gray)
        cv2.rectangle(
            img,
            (x, bar_y),
            (x + bar_width, bar_y + bar_height),
            (60, 60, 60),
            -1
        )

        # Foreground bar (colored by confidence)
        filled_width = int(bar_width * prob)
        if filled_width > 0:
            cv2.rectangle(
                img,
                (x, bar_y),
                (x + filled_width, bar_y + bar_height),
                color,
                -1
            )

        # Border
        cv2.rectangle(
            img,
            (x, bar_y),
            (x + bar_width, bar_y + bar_height),
            (255, 255, 255),
            1
        )

        # Label and percentage
        label = f"{name}: {prob:.0%}"
        cv2.putText(
            img, label,
            (x + bar_width + 10, bar_y + bar_height - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5, (255, 255, 255), 1, cv2.LINE_AA
        )


def draw_state_indicator(
    img: np.ndarray,
    has_state: bool,
    x: int = 10,
    y: int = 240,
) -> None:
    """Draw indicator showing if model is using stateful recurrence."""
    color = (0, 255, 0) if has_state else (128, 128, 128)
    status = "STATE: Active" if has_state else "STATE: Reset"
    cv2.circle(img, (x + 5, y), 5, color, -1)
    cv2.putText(
        img, status,
        (x + 15, y + 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4, color, 1, cv2.LINE_AA
    )


def draw_landmarks(image, hand_landmarks, connections=None):
    """Draw hand landmarks on image."""
    h, w = image.shape[:2]
    if connections is None:
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),
            (0, 5), (5, 6), (6, 7), (7, 8),
            (0, 9), (9, 10), (10, 11), (11, 12),
            (0, 13), (13, 14), (14, 15), (15, 16),
            (0, 17), (17, 18), (18, 19), (19, 20),
            (5, 9), (9, 13), (13, 17),
        ]

    pts = []
    for lm in hand_landmarks.landmark:
        x, y = int(lm.x * w), int(lm.y * h)
        pts.append((x, y))
        cv2.circle(image, (x, y), 4, (0, 255, 0), -1, cv2.LINE_AA)

    for i, j in connections:
        if i < len(pts) and j < len(pts):
            cv2.line(image, pts[i], pts[j], (0, 200, 255), 2, cv2.LINE_AA)


def extract_raw_landmarks(hand_landmarks) -> np.ndarray:
    """Extract raw 63-dim features from MediaPipe hand landmarks."""
    features = []
    for lm in hand_landmarks.landmark:
        features.extend([lm.x, lm.y, lm.z])
    return np.array(features, dtype=np.float32)


def extract_processed_features(lms_xy: np.ndarray, handedness: str | None) -> np.ndarray:
    """Extract 102-dim processed features using landmark_features.py logic."""
    from landmark_features import build_feature_vector
    return build_feature_vector(lms_xy, handedness)


def convex_hull_mask(shape_wh, pts_xy, dilate_px=30, feather=21):
    """Build a convex hull mask around the hand."""
    W, H = shape_wh
    mask = np.zeros((H, W), np.uint8)
    if not pts_xy:
        return mask, None
    hull = cv2.convexHull(np.array(pts_xy, dtype=np.int32))
    cv2.fillConvexPoly(mask, hull, 255)
    if dilate_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_px, dilate_px))
        mask = cv2.dilate(mask, k, 1)
    if feather > 0:
        mask = cv2.GaussianBlur(mask, (feather | 1, feather | 1), 0)
    return mask, hull


def focus_effect(frame, mask, darken=0.2, inside_boost=1.05):
    """Darken background and boost hand region."""
    m = (mask.astype(np.float32) / 255.0)[:, :, None]
    outside = (frame.astype(np.float32) * (1.0 - darken)).astype(np.uint8)
    inside = np.clip(frame.astype(np.float32) * inside_boost, 0, 255).astype(np.uint8)
    return (inside * m + outside * (1.0 - m)).astype(np.uint8)


def main(args):
    # Load model config
    config_path = Path("outputs/models/trm_config.json")
    model_path = Path("outputs/models/trm_best.pt")
    scaler_path = Path("outputs/models/trm_scaler.joblib")

    if not model_path.exists():
        raise FileNotFoundError(
            f"S-TRM model not found at {model_path}. "
            "Please train the model first with: uv run python -m src.train_trm"
        )

    # Load config
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    class_names = config["class_names"]
    input_dim = config["input_dim"]
    num_classes = config["num_classes"]
    dropout = config.get("dropout", 0.1)

    print(f"[S-TRM] Input dim: {input_dim}, Classes: {num_classes}")
    print(f"[S-TRM] Classes: {class_names}")
    print(f"[S-TRM] Stateful mode: {'Enabled' if args.stateful else 'Disabled'}")

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TRM(
        input_dim=input_dim,
        num_classes=num_classes,
        latent_dim=config["latent_dim"],
        hidden_dim=config["hidden_dim"],
        n_latent=config["n_latent"],
        T_deep=config["T_deep"],
        deep_supervision=False,
        dropout=dropout,
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Load scaler
    scaler = joblib.load(scaler_path)

    print(f"[S-TRM] Model loaded. Device: {device}")

    # Webcam
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)

    # MediaPipe Hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        model_complexity=1,
        min_detection_confidence=args.det_conf,
        min_tracking_confidence=args.trk_conf,
    )

    # State
    mirror = True
    ring = deque(maxlen=args.stable_frames)
    last_letter_time = 0.0
    word_chars = []
    phrase_tokens = []
    nohand_ms = 0.0
    prev_t = time.monotonic()

    # S-TRM stateful recurrence state
    model_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    state_active = False
    frames_with_hand = 0
    state_warmup_frames = 3  # Frames needed before state becomes "active"

    # Probability history for smoothing
    prob_history = deque(maxlen=args.prob_smooth_frames)
    current_probs = np.zeros(num_classes, dtype=np.float32)

    help_line = "Q quit | M mirror | R reset state | BKSP del | ENTER commit | SPACE space"

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if mirror:
                frame = cv2.flip(frame, 1)

            H, W = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            t = time.monotonic()
            dt = t - prev_t
            prev_t = t

            result = hands.process(rgb)

            pred_txt = "-"
            p_txt = 0.0

            if result.multi_hand_landmarks:
                nohand_ms = 0.0
                hand_lms = result.multi_hand_landmarks[0]

                # Get handedness
                handedness = None
                try:
                    handedness = result.multi_handedness[0].classification[0].label
                except Exception:
                    pass

                # Extract features based on input_dim
                if input_dim == 63:
                    # Raw landmarks
                    features = extract_raw_landmarks(hand_lms)
                else:
                    # Processed features (102 dims)
                    lms_xy = np.array([[lm.x, lm.y] for lm in hand_lms.landmark], dtype=np.float32)
                    features = extract_processed_features(lms_xy, handedness)

                # Normalize
                features = scaler.transform(features.reshape(1, -1))

                # Stateful inference
                with torch.no_grad():
                    x = torch.tensor(features, dtype=torch.float32, device=device)

                    # Use stateful recurrence if enabled
                    if args.stateful:
                        logits, model_state = model(
                            x,
                            prev_state=model_state,
                            return_all_outputs=False,
                            return_state=True
                        )
                        frames_with_hand += 1
                        state_active = frames_with_hand >= state_warmup_frames
                    else:
                        logits = model(x, return_all_outputs=False)

                    probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]

                # Temporal smoothing of probabilities
                prob_history.append(probs)
                if len(prob_history) > 1:
                    current_probs = np.mean(prob_history, axis=0)
                else:
                    current_probs = probs

                idx = int(current_probs.argmax())
                p = float(current_probs[idx])
                pred = class_names[idx]
                pred_txt, p_txt = pred, p

                if p >= args.min_prob:
                    ring.append(pred)

                # Visual effects
                lms_xy = np.array([[lm.x, lm.y] for lm in hand_lms.landmark], dtype=np.float32)
                xs = (lms_xy[:, 0] * W).astype(int)
                ys = (lms_xy[:, 1] * H).astype(int)
                pts = list(zip(xs, ys))

                mask, hull = convex_hull_mask((W, H), pts, dilate_px=30, feather=21)
                frame = focus_effect(frame, mask, darken=args.darken, inside_boost=1.05)

                if hull is not None:
                    cv2.polylines(frame, [hull], True, (0, 255, 255), 2, cv2.LINE_AA)

                draw_landmarks(frame, hand_lms)

            else:
                nohand_ms += dt * 1000
                # Reset state after no hand for a while
                if nohand_ms > args.state_reset_ms:
                    model_state = None
                    state_active = False
                    frames_with_hand = 0
                    prob_history.clear()
                    current_probs = np.zeros(num_classes, dtype=np.float32)

            # Stabilization + cooldown
            now_ms = time.monotonic() * 1000
            if len(ring) == ring.maxlen and (now_ms - last_letter_time) >= args.letter_cooldown_ms:
                maj = Counter(ring).most_common(1)[0][0]
                if not word_chars or word_chars[-1] != maj:
                    word_chars.append(maj)
                    last_letter_time = now_ms
                ring.clear()

            # End word by pause
            if nohand_ms >= args.word_pause_ms and word_chars:
                phrase_tokens.append("".join(word_chars))
                word_chars.clear()
                nohand_ms = 0.0

            # HUD
            fps = 1.0 / max(1e-6, dt)
            put_text(frame, f"FPS {fps:.1f}", (10, 20), 0.6, 1)
            put_text(frame, f"S-TRM Pred: {pred_txt} ({p_txt:.2f})", (10, 45))
            put_text(frame, f"Word: {''.join(word_chars)}", (10, 75))
            put_text(frame, f"Phrase: {' '.join(phrase_tokens)[:70]}", (10, 105), 0.6, 2)

            # Draw confidence bar graph for top-3 predictions
            if current_probs.sum() > 0:
                draw_confidence_bars(frame, current_probs, class_names, top_k=3, x=10, y=130)

            # Draw state indicator
            if args.stateful:
                draw_state_indicator(frame, state_active, x=10, y=220)

            put_text(frame, help_line, (10, H - 10), 0.5, 1)

            cv2.imshow("S-TRM ASL Recognition", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key in (ord("m"), ord("M")):
                mirror = not mirror
            elif key in (ord("r"), ord("R")):
                # Reset model state
                model_state = None
                state_active = False
                frames_with_hand = 0
                prob_history.clear()
                current_probs = np.zeros(num_classes, dtype=np.float32)
                print("[S-TRM] State reset")
            elif key == 8 and word_chars:  # Backspace
                word_chars.pop()
            elif key == 13:  # Enter
                if word_chars:
                    phrase_tokens.append("".join(word_chars))
                    word_chars.clear()
            elif key == 32:  # Space
                if word_chars:
                    phrase_tokens.append("".join(word_chars))
                    word_chars.clear()
                phrase_tokens.append("")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        hands.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="S-TRM (Stateful TRM) ASL Recognition Demo")

    # Detection
    parser.add_argument("--det_conf", type=float, default=0.6,
                        help="MediaPipe hand detection confidence")
    parser.add_argument("--trk_conf", type=float, default=0.6,
                        help="MediaPipe hand tracking confidence")

    # Stateful mode
    parser.add_argument("--stateful", action="store_true", default=True,
                        help="Enable stateful recurrence (state passed between frames)")
    parser.add_argument("--no-stateful", dest="stateful", action="store_false",
                        help="Disable stateful recurrence")
    parser.add_argument("--state_reset_ms", type=int, default=500,
                        help="Reset state after no hand detected for this many ms")
    parser.add_argument("--prob_smooth_frames", type=int, default=3,
                        help="Number of frames to smooth probabilities over")

    # Stabilization
    parser.add_argument("--min_prob", type=float, default=0.80,
                        help="Minimum probability threshold for letter detection")
    parser.add_argument("--stable_frames", type=int, default=10,
                        help="Number of stable frames before committing letter")
    parser.add_argument("--letter_cooldown_ms", type=int, default=1000,
                        help="Cooldown between letter detections")
    parser.add_argument("--word_pause_ms", type=int, default=1500,
                        help="Pause duration to end word")

    # Visual
    parser.add_argument("--darken", type=float, default=0.20,
                        help="Background darkening amount")

    args = parser.parse_args()
    main(args)
