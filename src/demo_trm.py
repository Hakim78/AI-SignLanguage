# src/demo_trm.py
"""
S-TRM Production Demo: Strict Mode Switching System
====================================================

Architecture avec séparation stricte des modes de reconnaissance.
L'utilisateur contrôle manuellement le mode pour éviter les hallucinations.

Modes:
- SPELLING MODE (Touche L): Reconnaissance des lettres statiques uniquement (A-Y)
- WORD MODE (Touche W): Reconnaissance des gestes dynamiques uniquement (HELLO, YES, NO...)

Principe:
Le modèle performe mal quand il mélange lettres et mots dans la même passe d'inférence.
Solution: HARD MASK des probabilités selon le mode sélectionné.

Data Pipeline:
- Input: Coordonnées MediaPipe BRUTES (21 landmarks × 3 coords = 63 dims)
- Normalisation: StandardScaler (identique à l'entraînement)
- AUCUNE transformation géométrique (pas de centrage, pas de rotation)

Auteur: S-TRM Team - IPSSI MIA4
Version: 3.0 Production (Strict Mode Switch)
"""
from __future__ import annotations
import argparse
import json
import time
from collections import deque
from pathlib import Path
from typing import Optional, Tuple, List

import cv2
import numpy as np
import mediapipe as mp
import torch
import joblib

from trm_model import TRM


# =============================================================================
# CONFIGURATION - CLASSES DYNAMIQUES (MOTS/GESTES)
# =============================================================================

# Ces classes nécessitent du mouvement et sont reconnues en WORD_MODE
DYNAMIC_CLASSES = {"J", "Z", "HELLO", "THANKYOU", "SORRY", "YES", "NO"}

# Toutes les autres classes sont des lettres statiques (SPELLING_MODE)


# =============================================================================
# CONFIGURATION - STABILITÉ ET LISSAGE
# =============================================================================

# Taille du buffer de lissage temporel (frames)
SMOOTHING_BUFFER_SIZE = 8

# Frames requises pour valider une prédiction
STABILITY_FRAMES_SPELLING = 10  # Lettres: besoin de stabilité
STABILITY_FRAMES_WORD = 6       # Mots: validation plus rapide

# Cooldown après validation (évite répétitions)
COOLDOWN_FRAMES = 12


# =============================================================================
# CLASSES UTILITAIRES
# =============================================================================

class ProbabilityBuffer:
    """
    Buffer de lissage temporel des probabilités.
    Moyenne glissante sur N frames pour stabiliser les prédictions.
    """

    def __init__(self, buffer_size: int = SMOOTHING_BUFFER_SIZE, num_classes: int = 24):
        self.buffer_size = buffer_size
        self.num_classes = num_classes
        self.prob_history: deque = deque(maxlen=buffer_size)

    def reset(self):
        """Reset du buffer."""
        self.prob_history.clear()

    def update(self, probs: np.ndarray) -> np.ndarray:
        """Ajoute les probabilités et retourne la moyenne lissée."""
        self.prob_history.append(probs.copy())

        if len(self.prob_history) == 0:
            return probs

        stacked = np.stack(list(self.prob_history), axis=0)
        return np.mean(stacked, axis=0)

    @property
    def fill_ratio(self) -> float:
        """Ratio de remplissage du buffer (0 à 1)."""
        return len(self.prob_history) / self.buffer_size


class StabilityTracker:
    """
    Traqueur de stabilité des prédictions.
    Valide une prédiction après N frames consécutives avec confiance suffisante.
    """

    def __init__(self, cooldown: int = COOLDOWN_FRAMES):
        self.cooldown = cooldown
        self.last_class: Optional[str] = None
        self.consecutive_count = 0
        self.cooldown_counter = 0

    def reset(self):
        """Reset du tracker."""
        self.last_class = None
        self.consecutive_count = 0
        self.cooldown_counter = 0

    def update(
        self,
        predicted_class: str,
        confidence: float,
        min_confidence: float,
        required_frames: int
    ) -> Optional[str]:
        """
        Met à jour et retourne la classe validée si applicable.
        """
        # Gestion cooldown
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            return None

        # Confiance insuffisante
        if confidence < min_confidence:
            self.consecutive_count = 0
            return None

        # Comptage consécutif
        if predicted_class == self.last_class:
            self.consecutive_count += 1
        else:
            self.last_class = predicted_class
            self.consecutive_count = 1

        # Validation
        if self.consecutive_count >= required_frames:
            self.cooldown_counter = self.cooldown
            self.consecutive_count = 0
            return predicted_class

        return None

    @property
    def progress(self) -> float:
        """Progression vers la validation (0 à 1)."""
        if self.last_class is None:
            return 0.0
        return min(1.0, self.consecutive_count / STABILITY_FRAMES_SPELLING)


# =============================================================================
# EXTRACTION DE FEATURES - RAW INPUT (CRITICAL)
# =============================================================================

def extract_raw_features(hand_landmarks) -> np.ndarray:
    """
    Extrait les features BRUTES des landmarks MediaPipe.

    CRITICAL: Doit reproduire EXACTEMENT le pipeline d'entraînement.
    - 21 landmarks × 3 coordonnées (x, y, z) = 63 dimensions
    - Coordonnées normalisées [0, 1] par MediaPipe
    - AUCUNE transformation (pas de centrage, pas de rotation)

    Args:
        hand_landmarks: MediaPipe hand landmarks

    Returns:
        np.ndarray: Features brutes [63]
    """
    raw_coords = []
    for lm in hand_landmarks.landmark:
        raw_coords.extend([lm.x, lm.y, lm.z])
    return np.array(raw_coords, dtype=np.float32)


# =============================================================================
# INTERFACE UTILISATEUR - HUD AVEC MODE STRICT
# =============================================================================

def draw_strict_mode_hud(
    frame: np.ndarray,
    current_mode: str,  # "SPELLING" ou "WORD"
    top_predictions: List[Tuple[str, float]],
    stability_progress: float,
    buffer_fill: float,
    word_buffer: str,
    sentence: str,
    fps: float,
    hand_detected: bool
):
    """
    Dessine l'interface avec indication claire du mode actif.
    """
    H, W = frame.shape[:2]

    # Couleurs selon le mode
    if current_mode == "SPELLING":
        mode_color = (255, 200, 0)    # Cyan/Jaune
        mode_text = "SPELLING MODE"
        mode_subtitle = "Letters Only (A-Y)"
    else:  # WORD
        mode_color = (0, 140, 255)    # Orange
        mode_text = "WORD MODE"
        mode_subtitle = "Gestures Only (HELLO, YES, NO...)"

    status_color = mode_color if hand_detected else (80, 80, 80)

    # =========================================================================
    # 1. BANNIÈRE DE MODE (Très visible en haut)
    # =========================================================================
    banner_height = 80
    cv2.rectangle(frame, (0, 0), (W, banner_height), (20, 20, 20), -1)

    # Mode principal (GRAND)
    cv2.putText(frame, f"[ {mode_text} ]", (W // 2 - 180, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 1.3, mode_color, 3)

    # Sous-titre
    cv2.putText(frame, mode_subtitle, (W // 2 - 140, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

    # FPS (coin droit)
    cv2.putText(frame, f"FPS: {int(fps)}", (W - 100, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)

    # Indicateur de statut main
    hand_status = "HAND OK" if hand_detected else "NO HAND"
    hand_color = (0, 255, 0) if hand_detected else (100, 100, 100)
    cv2.putText(frame, hand_status, (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, hand_color, 2)

    # =========================================================================
    # 2. PANNEAU DE PRÉDICTION (Gauche)
    # =========================================================================
    panel_x, panel_y = 15, 95
    panel_w, panel_h = 300, 180

    # Fond semi-transparent
    overlay = frame.copy()
    cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h),
                  (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

    # Bordure colorée selon le mode
    cv2.rectangle(frame, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h),
                  status_color, 2)

    # Titre
    cv2.putText(frame, "DETECTION", (panel_x + 10, panel_y + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)

    # Top-3 prédictions
    for i, (cls, prob) in enumerate(top_predictions[:3]):
        y_offset = panel_y + 55 + i * 45

        # Nom de la classe
        text_color = (255, 255, 255) if i == 0 else (130, 130, 130)
        font_scale = 1.0 if i == 0 else 0.6
        thickness = 2 if i == 0 else 1
        cv2.putText(frame, cls, (panel_x + 15, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness)

        # Barre de confiance
        bar_x = panel_x + 110
        bar_w = 150
        bar_h = 16 if i == 0 else 10
        bar_y = y_offset - bar_h + 3

        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
                      (40, 40, 40), -1)

        fill_w = int(bar_w * prob)
        if prob > 0.7:
            fill_color = (0, 255, 0)
        elif prob > 0.4:
            fill_color = (0, 255, 255)
        else:
            fill_color = (80, 80, 80)

        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h),
                      fill_color, -1)

        # Pourcentage
        cv2.putText(frame, f"{prob:.0%}", (bar_x + bar_w + 8, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

    # =========================================================================
    # 3. INDICATEURS DE PROGRESSION (Sous le panneau)
    # =========================================================================
    prog_y = panel_y + panel_h + 20

    # Buffer fill
    cv2.putText(frame, "BUFFER:", (panel_x, prog_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)

    for i in range(8):
        dot_x = panel_x + 65 + i * 14
        dot_filled = i < int(buffer_fill * 8)
        dot_color = mode_color if dot_filled else (40, 40, 40)
        cv2.circle(frame, (dot_x, prog_y - 5), 5, dot_color, -1)

    # Lock progress
    prog_y += 25
    cv2.putText(frame, "LOCK:", (panel_x, prog_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)

    lock_bar_x = panel_x + 65
    lock_bar_w = 112
    cv2.rectangle(frame, (lock_bar_x, prog_y - 12), (lock_bar_x + lock_bar_w, prog_y),
                  (40, 40, 40), -1)

    lock_fill = int(lock_bar_w * stability_progress)
    lock_color = (0, 255, 255) if stability_progress < 1.0 else (0, 255, 0)
    cv2.rectangle(frame, (lock_bar_x, prog_y - 12), (lock_bar_x + lock_fill, prog_y),
                  lock_color, -1)

    # =========================================================================
    # 4. ZONE DE SAISIE (Bas)
    # =========================================================================
    text_panel_y = H - 100
    cv2.rectangle(frame, (0, text_panel_y), (W, H), (15, 15, 15), -1)
    cv2.line(frame, (0, text_panel_y), (W, text_panel_y), mode_color, 3)

    # Buffer de lettres
    cv2.putText(frame, "TYPING:", (20, text_panel_y + 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)

    cursor = "|" if int(time.time() * 2) % 2 == 0 else ""
    cv2.putText(frame, word_buffer + cursor, (100, text_panel_y + 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, mode_color, 2)

    # Phrase complète
    cv2.putText(frame, "OUTPUT:", (20, text_panel_y + 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)

    display_sentence = sentence if len(sentence) < 55 else "..." + sentence[-52:]
    cv2.putText(frame, display_sentence, (100, text_panel_y + 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    # =========================================================================
    # 5. CONTRÔLES (Bas)
    # =========================================================================
    help_y = H - 10
    cv2.putText(frame, "[L] Letters  [W] Words  [SPACE] Validate  [BKSP] Delete  [R] Reset  [Q] Quit",
                (15, help_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (70, 70, 70), 1)


def draw_hand_overlay(frame: np.ndarray, landmarks, mode: str):
    """Dessine les landmarks avec couleur selon le mode."""
    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands

    if mode == "SPELLING":
        landmark_color = (255, 200, 0)    # Cyan
        connection_color = (200, 150, 0)
    else:
        landmark_color = (0, 140, 255)    # Orange
        connection_color = (0, 100, 200)

    landmark_style = mp_drawing.DrawingSpec(
        color=landmark_color, thickness=2, circle_radius=4
    )
    connection_style = mp_drawing.DrawingSpec(
        color=connection_color, thickness=2
    )

    mp_drawing.draw_landmarks(
        frame, landmarks, mp_hands.HAND_CONNECTIONS,
        landmark_style, connection_style
    )


# =============================================================================
# FONCTION PRINCIPALE
# =============================================================================

def main(args):
    """
    Boucle principale avec système de modes stricts.
    """
    print("=" * 65)
    print("  S-TRM STRICT MODE SWITCHING SYSTEM v3.0")
    print("  Press [L] for Letters | Press [W] for Words")
    print("=" * 65)

    # =========================================================================
    # CHARGEMENT DES RESSOURCES
    # =========================================================================
    model_path = Path("outputs/models/trm_best.pt")
    config_path = Path("outputs/models/trm_config.json")
    scaler_path = Path("outputs/models/trm_scaler.joblib")

    for path, name in [(model_path, "Model"), (config_path, "Config"), (scaler_path, "Scaler")]:
        if not path.exists():
            print(f"[ERROR] {name} not found: {path}")
            return

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    class_names = config["class_names"]
    num_classes = config["num_classes"]
    input_dim = config.get("input_dim", 63)

    print(f"\n[CONFIG] Classes: {num_classes} -> {class_names}")

    # Indices pour le masquage strict
    static_indices = [i for i, name in enumerate(class_names) if name not in DYNAMIC_CLASSES]
    dynamic_indices = [i for i, name in enumerate(class_names) if name in DYNAMIC_CLASSES]

    static_names = [class_names[i] for i in static_indices]
    dynamic_names = [class_names[i] for i in dynamic_indices]

    print(f"[CONFIG] Static letters ({len(static_indices)}): {static_names}")
    print(f"[CONFIG] Dynamic words ({len(dynamic_indices)}): {dynamic_names}")

    # Chargement scaler et modèle
    scaler = joblib.load(scaler_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = TRM(
        input_dim=input_dim,
        num_classes=num_classes,
        latent_dim=config.get("latent_dim", 64),
        hidden_dim=config.get("hidden_dim", 128),
        n_latent=config.get("n_latent", 6),
        T_deep=config.get("T_deep", 3),
        dropout=config.get("dropout", 0.1)
    ).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    print(f"[CONFIG] Device: {device}")
    print(f"[CONFIG] Input pipeline: RAW (63 dims, no centering)")

    # =========================================================================
    # INITIALISATION
    # =========================================================================
    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=args.detection_confidence,
        min_tracking_confidence=args.tracking_confidence
    )

    # État du système
    current_mode = "SPELLING"  # Mode par défaut
    prob_buffer = ProbabilityBuffer(buffer_size=SMOOTHING_BUFFER_SIZE, num_classes=num_classes)
    stability_tracker = StabilityTracker()
    model_state = None

    # Saisie
    word_buffer: List[str] = []
    sentence_tokens: List[str] = []

    # FPS
    prev_time = time.time()
    fps_history = deque(maxlen=30)

    print("\n[READY] System initialized.")
    print("[INFO] Default mode: SPELLING (Letters)")
    print("[INFO] Press [W] to switch to WORD mode\n")

    # =========================================================================
    # BOUCLE PRINCIPALE
    # =========================================================================
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # FPS
        curr_time = time.time()
        dt = curr_time - prev_time
        prev_time = curr_time
        fps_history.append(1.0 / dt if dt > 0 else 30)
        fps = np.mean(fps_history)

        # Miroir
        frame = cv2.flip(frame, 1)
        H, W = frame.shape[:2]

        # MediaPipe
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        # État par défaut
        hand_detected = False
        top_predictions = [("---", 0.0), ("---", 0.0), ("---", 0.0)]

        if results.multi_hand_landmarks:
            hand_detected = True
            landmarks = results.multi_hand_landmarks[0]

            # =================================================================
            # ÉTAPE 1: EXTRACTION RAW FEATURES
            # =================================================================
            raw_features = extract_raw_features(landmarks)
            features_normalized = scaler.transform(raw_features.reshape(1, -1))

            # =================================================================
            # ÉTAPE 2: INFÉRENCE S-TRM
            # =================================================================
            with torch.no_grad():
                x = torch.tensor(features_normalized, dtype=torch.float32, device=device)

                if args.stateful:
                    logits, model_state = model(x, prev_state=model_state, return_state=True)
                else:
                    logits = model(x)

                raw_probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]

            # =================================================================
            # ÉTAPE 3: HARD MASK SELON LE MODE (CRITICAL)
            # =================================================================
            masked_probs = raw_probs.copy()

            if current_mode == "SPELLING":
                # SPELLING MODE: Masquer tous les mots dynamiques
                for idx in dynamic_indices:
                    masked_probs[idx] = 0.0
            else:
                # WORD MODE: Masquer toutes les lettres statiques
                for idx in static_indices:
                    masked_probs[idx] = 0.0

            # Renormalisation
            prob_sum = masked_probs.sum()
            if prob_sum > 0:
                masked_probs /= prob_sum
            else:
                # Fallback si tout est masqué
                masked_probs = np.ones(num_classes) / num_classes

            # =================================================================
            # ÉTAPE 4: LISSAGE TEMPOREL
            # =================================================================
            smoothed_probs = prob_buffer.update(masked_probs)

            # =================================================================
            # ÉTAPE 5: TOP-3 PRÉDICTIONS
            # =================================================================
            top_indices = np.argsort(smoothed_probs)[::-1][:3]
            top_predictions = [(class_names[i], smoothed_probs[i]) for i in top_indices]

            # =================================================================
            # ÉTAPE 6: VALIDATION
            # =================================================================
            top_class, top_prob = top_predictions[0]
            required_frames = STABILITY_FRAMES_WORD if current_mode == "WORD" else STABILITY_FRAMES_SPELLING

            validated_class = stability_tracker.update(
                predicted_class=top_class,
                confidence=top_prob,
                min_confidence=args.min_confidence,
                required_frames=required_frames
            )

            # =================================================================
            # ÉTAPE 7: ACTION SUR VALIDATION
            # =================================================================
            if validated_class is not None:
                if current_mode == "WORD":
                    # Mot complet -> ajouter directement à la phrase
                    sentence_tokens.append(f"[{validated_class}]")
                    # Flash visuel
                    cv2.rectangle(frame, (0, 0), (W, H), (0, 200, 255), 10)
                else:
                    # Lettre -> ajouter au buffer
                    word_buffer.append(validated_class)
                    # Flash léger
                    cv2.rectangle(frame, (0, 0), (W, H), (255, 200, 0), 5)

            # Dessiner la main
            draw_hand_overlay(frame, landmarks, current_mode)

        else:
            # Pas de main - Reset partiel
            prob_buffer.reset()
            stability_tracker.reset()
            model_state = None

        # =====================================================================
        # AFFICHAGE HUD
        # =====================================================================
        word_str = "".join(word_buffer)
        sentence_str = " ".join(sentence_tokens)

        draw_strict_mode_hud(
            frame=frame,
            current_mode=current_mode,
            top_predictions=top_predictions,
            stability_progress=stability_tracker.progress,
            buffer_fill=prob_buffer.fill_ratio,
            word_buffer=word_str,
            sentence=sentence_str,
            fps=fps,
            hand_detected=hand_detected
        )

        cv2.imshow("S-TRM Strict Mode System", frame)

        # =====================================================================
        # GESTION CLAVIER
        # =====================================================================
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q') or key == 27:  # Quit
            break

        elif key == ord('l'):  # Switch to SPELLING mode
            if current_mode != "SPELLING":
                current_mode = "SPELLING"
                prob_buffer.reset()
                stability_tracker.reset()
                print("[MODE] Switched to SPELLING MODE (Letters)")

        elif key == ord('w'):  # Switch to WORD mode
            if current_mode != "WORD":
                current_mode = "WORD"
                prob_buffer.reset()
                stability_tracker.reset()
                print("[MODE] Switched to WORD MODE (Gestures)")

        elif key == 8:  # Backspace
            if word_buffer:
                word_buffer.pop()

        elif key == 32:  # Space - valider mot
            if word_buffer:
                sentence_tokens.append("".join(word_buffer))
                word_buffer = []

        elif key == 13:  # Enter
            if word_buffer:
                sentence_tokens.append("".join(word_buffer))
                word_buffer = []

        elif key == ord('r'):  # Reset tout
            word_buffer = []
            sentence_tokens = []
            prob_buffer.reset()
            stability_tracker.reset()
            model_state = None
            print("[RESET] All cleared")

        elif key == ord('c'):  # Clear sentence
            sentence_tokens = []
            print("[CLEAR] Sentence cleared")

    # =========================================================================
    # CLEANUP
    # =========================================================================
    cap.release()
    cv2.destroyAllWindows()
    hands.close()

    final_output = " ".join(sentence_tokens)
    if word_buffer:
        final_output += " " + "".join(word_buffer)

    print("\n" + "=" * 65)
    print("SESSION ENDED")
    print("=" * 65)
    print(f"Final output: {final_output}")
    print("=" * 65)


# =============================================================================
# POINT D'ENTRÉE
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="S-TRM Strict Mode Switching System",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Caméra
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)

    # MediaPipe
    parser.add_argument("--detection_confidence", type=float, default=0.7)
    parser.add_argument("--tracking_confidence", type=float, default=0.5)

    # Modèle
    parser.add_argument("--stateful", action="store_true", default=True)
    parser.add_argument("--no-stateful", dest="stateful", action="store_false")

    # Seuils
    parser.add_argument("--min_confidence", type=float, default=0.50)

    args = parser.parse_args()
    main(args)
