# S-TRM: Stateful Tiny Recursive Model for ASL Recognition

**Version 2.0** - Architecture neuronale avancée pour la reconnaissance de l'alphabet ASL en temps réel.

Ce projet implémente le **S-TRM (Stateful Tiny Recursive Model)**, une architecture légère et récursive optimisée pour les appareils edge (lunettes AR, mobile) avec support de l'inférence stateful pour les vidéos.

---

## Table des matières

1. [Nouveautés v2](#nouveautés-v2)
2. [Architecture S-TRM](#architecture-s-trm)
3. [Installation](#installation)
4. [Structure du projet](#structure-du-projet)
5. [Entraînement local](#entraînement-local)
6. [Entraînement sur Kaggle](#entraînement-sur-kaggle)
7. [Démo temps réel](#démo-temps-réel)
8. [Intégration Hugging Face Hub](#intégration-hugging-face-hub)
9. [API du modèle](#api-du-modèle)
10. [Comparaison avec v1](#comparaison-avec-v1)

---

## Nouveautés v2

| Fonctionnalité | v1 (MLP) | v2 (S-TRM) |
|----------------|----------|------------|
| Architecture | MLP scikit-learn | Réseau récursif PyTorch |
| Mémoire temporelle | Non | Oui (état entre frames) |
| Lettres dynamiques (J, Z) | Non supportées | Supportées |
| Deep Supervision | Non | Oui (loss pondérée) |
| Entraînement cloud | Non | Kaggle + HF Hub |
| Paramètres | ~50K | ~79K |
| BPTT (séquences) | Non | Oui |

---

## Architecture S-TRM

Le S-TRM est basé sur le papier "Tiny Recursive Model" avec des extensions pour l'inférence stateful.

```
┌─────────────────────────────────────────────────────────────────┐
│                        S-TRM Architecture                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Frame t-1 ──► [prev_state: (y, z)] ──────────────────┐         │
│                                                        │         │
│  Frame t ──► [Input Projection] ──► y₀                │         │
│                      │                                 │         │
│                      ▼                                 ▼         │
│              ┌───────────────┐                 ┌─────────────┐  │
│              │ Latent Block  │ ◄───────────── │ State Gate  │  │
│              │   n=6 iters   │                 │  (sigmoid)  │  │
│              └───────┬───────┘                 └─────────────┘  │
│                      │                                          │
│                      ▼                                          │
│              ┌───────────────┐                                  │
│              │  Deep Block   │ ──► Intermediate logits (T=3)   │
│              │   T=3 iters   │                                  │
│              └───────┬───────┘                                  │
│                      │                                          │
│                      ▼                                          │
│              [Output Layer] ──► Final logits (24-31 classes)   │
│                      │                                          │
│                      ▼                                          │
│              [new_state: (y, z)] ──► Frame t+1                 │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Composants clés

| Composant | Dimension | Description |
|-----------|-----------|-------------|
| Input Projection | 63 → 64 | Encode les 21 landmarks (×3 coords) |
| Latent Block | 64 → 64 | Raisonnement récursif (n=6 itérations) |
| Deep Block | 64 → 128 → 64 | Itérations profondes (T=3) avec skip connections |
| State Gate | 64 → 64 | Mélange appris entre état précédent et nouveau |
| Output Layer | 64 → num_classes | Classification finale |

### Deep Supervision

La loss est calculée à chaque itération T avec pondération décroissante :

```
L_total = Σ(t=1→T) λ^(T-t) × CrossEntropy(logits_t, target)
```

Avec `λ = 0.9` par défaut, les itérations finales ont plus de poids.

---

## Installation

### Prérequis

- Python 3.10+
- CUDA (optionnel, pour GPU)
- Webcam (pour la démo)

### Installation des dépendances

```bash
# Cloner le repository
git clone https://github.com/Hakim78/AI-SignLanguage.git
cd AI-SignLanguage

# Installer les dépendances
pip install torch torchvision mediapipe==0.10.14 opencv-python
pip install scikit-learn joblib rich matplotlib seaborn

# Pour l'entraînement cloud
pip install huggingface_hub gitpython
```

### Avec uv (recommandé)

```bash
uv venv
uv sync
uv add torch mediapipe==0.10.14 opencv-python scikit-learn rich
```

---

## Structure du projet

```
AI-SignLanguage/
├── src/
│   ├── trm_model.py          # Architecture S-TRM (PyTorch)
│   ├── train_trm.py          # Entraînement local
│   ├── train_kaggle.py       # Entraînement Kaggle + HF Hub
│   ├── demo_trm.py           # Démo temps réel avec état
│   ├── process_videos.py     # Conversion vidéos → CSV landmarks
│   ├── landmark_features.py  # Extraction features (102 dims)
│   └── ...
├── outputs/
│   └── models/
│       ├── trm_best.pt       # Poids du modèle (Le Cerveau)
│       ├── trm_config.json   # Configuration (La Carte d'identité)
│       └── trm_scaler.joblib # Normaliseur (Le Traducteur)
├── data/
│   └── landmarks/
│       └── features.csv      # Dataset landmarks
├── README.md                 # Documentation v1 (MLP)
└── README_v2.md              # Documentation v2 (S-TRM) ← Vous êtes ici
```

---

## Entraînement local

### 1. Préparer les données

```bash
# Extraire les landmarks depuis ASL Alphabet
python -m src.extract_asl_alphabet_landmarks \
  --root data/asl_alphabet/asl_alphabet_train/asl_alphabet_train \
  --out_csv data/landmarks/features.csv \
  --max_per_class 400
```

### 2. Entraîner le S-TRM

```bash
# Mode single-frame (rapide)
python src/train_trm.py \
  --csv data/landmarks/features.csv \
  --epochs 100 \
  --batch_size 64 \
  --lr 1e-3

# Mode séquence avec BPTT (pour lettres dynamiques J, Z)
python src/train_trm.py \
  --csv data/landmarks/features.csv \
  --sequence_mode \
  --seq_len 16 \
  --epochs 100
```

### Arguments principaux

| Argument | Défaut | Description |
|----------|--------|-------------|
| `--csv` | requis | Chemin vers le CSV de features |
| `--epochs` | 100 | Nombre d'époques |
| `--batch_size` | 64 | Taille des batchs |
| `--lr` | 1e-3 | Learning rate |
| `--sequence_mode` | false | Activer BPTT |
| `--seq_len` | 16 | Longueur des séquences |
| `--n_latent` | 6 | Itérations latentes |
| `--T_deep` | 3 | Itérations profondes |
| `--dropout` | 0.1 | Probabilité dropout |

---

## Entraînement sur Kaggle

### 1. Configuration des secrets Kaggle

Dans votre notebook Kaggle : **Settings → Add-ons → Secrets**

| Secret | Description |
|--------|-------------|
| `HF_TOKEN` | Token Hugging Face (write permission) |
| `HF_REPO_ID` | ID du repo HF (ex: `username/asl-strm`) |
| `GITHUB_TOKEN` | (Optionnel) Token GitHub pour sync |

### 2. Notebook Kaggle

```python
# Cellule 1 - Installation
!pip install huggingface_hub gitpython rich

# Clone du repository
!git clone https://github.com/Hakim78/AI-SignLanguage.git
%cd AI-SignLanguage

# Configuration secrets
import os
from kaggle_secrets import UserSecretsClient
secrets = UserSecretsClient()
os.environ["HF_TOKEN"] = secrets.get_secret("HF_TOKEN")
os.environ["HF_REPO_ID"] = secrets.get_secret("HF_REPO_ID")

# Cellule 2 - Entraînement
!python src/train_kaggle.py \
  --csv /kaggle/input/your-dataset/features.csv \
  --epochs 100 \
  --sequence_mode \
  --hf_private
```

### 3. Conversion de vidéos en dataset

Si vous avez des vidéos ASL, convertissez-les en CSV :

```python
!python src/process_videos.py \
  --input_dir /kaggle/input/asl-videos \
  --output_csv /kaggle/working/landmarks.csv \
  --fps 15
```

---

## Démo temps réel

### Lancer la démo

```bash
# Mode stateful (recommandé pour vidéo)
python src/demo_trm.py --stateful

# Mode sans état (chaque frame indépendante)
python src/demo_trm.py
```

### Contrôles clavier

| Touche | Action |
|--------|--------|
| `Q` / `Esc` | Quitter |
| `M` | Miroir caméra |
| `R` | Reset de l'état (mode stateful) |
| `S` | Toggle affichage squelette |

### Options avancées

```bash
python src/demo_trm.py \
  --stateful \
  --camera 0 \
  --width 1280 \
  --height 720 \
  --confidence 0.7 \
  --smoothing 0.3
```

---

## Intégration Hugging Face Hub

Le modèle entraîné est automatiquement uploadé sur Hugging Face Hub.

### Télécharger un modèle pré-entraîné

```python
from huggingface_hub import hf_hub_download
import torch
import json

# Télécharger les fichiers
model_path = hf_hub_download(repo_id="your-username/asl-strm", filename="pytorch_model.pt")
config_path = hf_hub_download(repo_id="your-username/asl-strm", filename="config.json")

# Charger le modèle
from src.trm_model import TRM

with open(config_path) as f:
    config = json.load(f)

model = TRM(
    input_dim=config["input_dim"],
    num_classes=config["num_classes"],
    latent_dim=config["latent_dim"],
    hidden_dim=config["hidden_dim"],
    n_latent=config["n_latent"],
    T_deep=config["T_deep"],
)
model.load_state_dict(torch.load(model_path, map_location="cpu"))
model.eval()
```

### Inférence stateful

```python
state = None
for landmarks in video_frames:
    x = torch.tensor(landmarks).unsqueeze(0)
    logits, state = model(x, prev_state=state, return_state=True)
    prediction = logits.argmax(dim=-1).item()
    print(f"Prediction: {class_names[prediction]}")
```

---

## API du modèle

### TRM Forward

```python
def forward(
    self,
    x: torch.Tensor,                    # [B, input_dim]
    prev_state: Optional[Tuple] = None, # (y_prev, z_prev) pour stateful
    return_all_outputs: bool = False,   # Retourner tous les logits intermédiaires
    return_state: bool = False,         # Retourner le nouvel état
) -> Union[torch.Tensor, Tuple]:
    """
    Returns:
        - logits: [B, num_classes] ou list de [B, num_classes] si return_all_outputs
        - state: (y, z) si return_state=True
    """
```

### DeepSupervisionLoss

```python
loss_fn = DeepSupervisionLoss(
    num_classes=24,
    weight_decay=0.9,      # Pondération des itérations
    label_smoothing=0.1,   # Régularisation
)

# Usage
all_logits = model(x, return_all_outputs=True)  # Liste de T tenseurs
loss = loss_fn(all_logits, targets)
```

---

## Comparaison avec v1

| Aspect | v1 (README.md) | v2 (README_v2.md) |
|--------|----------------|-------------------|
| **Modèle** | MLP scikit-learn | S-TRM PyTorch |
| **Input** | 102 features invariantes | 63 landmarks bruts |
| **Entraînement** | Local uniquement | Local + Kaggle |
| **Lettres dynamiques** | Non (J, Z exclus) | Oui (BPTT) |
| **Mémoire temporelle** | Post-traitement | Architecture native |
| **Déploiement cloud** | Non | Hugging Face Hub |
| **GPU** | Non | Supporté |

### Quand utiliser v1 vs v2 ?

- **v1 (MLP)** : Déploiement CPU ultra-léger, lettres statiques uniquement
- **v2 (S-TRM)** : Reconnaissance complète incluant J/Z, inférence vidéo, déploiement cloud

---

## Fichiers de modèle

Après entraînement, vous obtenez 3 fichiers essentiels :

| Fichier | Rôle | Taille |
|---------|------|--------|
| `trm_best.pt` | Poids du réseau (Le Cerveau) | ~320 KB |
| `trm_config.json` | Architecture + classes (La Carte d'identité) | ~1 KB |
| `trm_scaler.joblib` | Normalisation StandardScaler (Le Traducteur) | ~5 KB |

**Emplacement** : `outputs/models/`

---

## Métriques attendues

Sur le dataset ASL Alphabet (24+ classes) :

| Métrique | Single-frame | Sequence (BPTT) |
|----------|--------------|-----------------|
| Accuracy | ~95-97% | ~97-99% |
| F1-Score (macro) | ~94-96% | ~96-98% |
| Latence (CPU) | ~2-5 ms | ~2-5 ms |
| Latence (GPU) | <1 ms | <1 ms |

---

## Licence

MIT License

---

## Crédits

- **MediaPipe Hands** : Google (détection 21 landmarks)
- **Tiny Recursive Model** : Inspiration architecturale
- **Datasets** : ASL Alphabet (Kaggle), Sign Language MNIST (Kaggle)
- **Développement** : Projet IPSSI MIA4 - Août 2025

---

## Citation

```bibtex
@software{strm_asl_2025,
  title = {S-TRM: Stateful Tiny Recursive Model for ASL Recognition},
  author = {Hakim et al.},
  year = {2025},
  url = {https://github.com/Hakim78/AI-SignLanguage}
}
```
