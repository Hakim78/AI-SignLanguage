# src/train_kaggle.py
"""
Kaggle Cloud Training Script for S-TRM with Hugging Face Hub Integration.

This script adapts train_trm.py for execution on Kaggle Kernels with:
- Automatic model upload to Hugging Face Hub after each best epoch
- GitHub sync for experiment tracking
- Kaggle-specific path handling

================================================================================
SETUP INSTRUCTIONS FOR KAGGLE
================================================================================

1. KAGGLE SECRETS SETUP:
   Go to Kaggle Notebook Settings -> Add-ons -> Secrets
   Add the following secrets:

   - HF_TOKEN: Your Hugging Face access token (write permission required)
     Get it from: https://huggingface.co/settings/tokens

   - GITHUB_TOKEN: (Optional) Personal access token for GitHub sync
     Get it from: https://github.com/settings/tokens

   - HF_REPO_ID: (Optional) Your HF repository ID (e.g., "username/model-name")
     If not set, defaults to "your-username/asl-strm-model"

2. KAGGLE NOTEBOOK CODE:
   Copy this to the first cell of your Kaggle notebook:

   ```python
   # Install dependencies
   !pip install huggingface_hub gitpython rich

   # Clone repository
   !git clone https://github.com/Hakim78/AI-SignLanguage.git
   %cd AI-SignLanguage

   # Set up secrets (Kaggle will inject these from your Secrets)
   import os
   from kaggle_secrets import UserSecretsClient
   secrets = UserSecretsClient()
   os.environ["HF_TOKEN"] = secrets.get_secret("HF_TOKEN")
   # os.environ["GITHUB_TOKEN"] = secrets.get_secret("GITHUB_TOKEN")  # Optional

   # Run training
   !python src/train_kaggle.py --csv /kaggle/input/your-dataset/features.csv --epochs 100
   ```

3. DATASET SETUP:
   - Upload your landmarks CSV to a Kaggle Dataset
   - Or use the process_videos.py script to create one from video datasets

================================================================================
HUGGING FACE HUB FEATURES
================================================================================

- Auto-uploads best model checkpoint (trm_best.pt) to HF Hub
- Uploads config.json with model architecture and training params
- Uploads training metrics (metrics.json) for experiment tracking
- Creates model card with training information

================================================================================
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

# Rich for nice console output
try:
    from rich import print
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn
    from rich.table import Table
    console = Console()
except ImportError:
    console = None
    print("[Warning] Rich not installed. Using basic print.")

# Hugging Face Hub
try:
    from huggingface_hub import HfApi, Repository, create_repo, upload_file, upload_folder
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("[Warning] huggingface_hub not installed. HF upload disabled.")

# Import from local modules
sys.path.insert(0, str(Path(__file__).parent))
from trm_model import TRM, DeepSupervisionLoss, create_trm_model
from train_trm import (
    LandmarkDataset,
    SequenceDataset,
    load_csv_data,
    train_epoch,
    train_epoch_sequence,
    eval_epoch,
)


class HuggingFaceCallback:
    """
    Callback for uploading models and metrics to Hugging Face Hub.

    Uploads:
    - Best model checkpoint (trm_best.pt)
    - Model config (trm_config.json)
    - Training metrics (metrics.json)
    - Scaler (trm_scaler.joblib)
    """

    def __init__(
        self,
        repo_id: str,
        token: str,
        local_dir: Path,
        private: bool = False,
    ):
        """
        Initialize HF callback.

        Args:
            repo_id: HF repository ID (e.g., "username/model-name")
            token: HF access token with write permission
            local_dir: Local directory containing model files
            private: Whether to create a private repository
        """
        self.repo_id = repo_id
        self.token = token
        self.local_dir = Path(local_dir)
        self.private = private
        self.api = HfApi(token=token)
        self.metrics_history = []

        # Create or get repository
        try:
            create_repo(
                repo_id=repo_id,
                token=token,
                private=private,
                repo_type="model",
                exist_ok=True,
            )
            print(f"[green]HF Repository ready:[/green] https://huggingface.co/{repo_id}")
        except Exception as e:
            print(f"[yellow]HF Repo creation warning: {e}[/yellow]")

    def on_epoch_end(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
        is_best: bool,
    ):
        """Called at the end of each epoch."""
        # Record metrics
        self.metrics_history.append({
            "epoch": epoch,
            "timestamp": datetime.now().isoformat(),
            "train_loss": train_metrics.get("loss", 0),
            "train_acc": train_metrics.get("accuracy", 0),
            "val_loss": val_metrics.get("loss", 0),
            "val_acc": val_metrics.get("accuracy", 0),
            "val_f1": val_metrics.get("f1_macro", 0),
            "is_best": is_best,
        })

        # Save metrics locally
        metrics_path = self.local_dir / "metrics.json"
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(self.metrics_history, f, indent=2)

        # Upload if best epoch
        if is_best:
            self._upload_best_model(epoch)

    def _upload_best_model(self, epoch: int):
        """Upload best model checkpoint to HF Hub."""
        try:
            files_to_upload = [
                ("trm_best.pt", "pytorch_model.pt"),
                ("trm_config.json", "config.json"),
                ("trm_scaler.joblib", "scaler.joblib"),
                ("metrics.json", "metrics.json"),
            ]

            for local_name, remote_name in files_to_upload:
                local_path = self.local_dir / local_name
                if local_path.exists():
                    upload_file(
                        path_or_fileobj=str(local_path),
                        path_in_repo=remote_name,
                        repo_id=self.repo_id,
                        token=self.token,
                        commit_message=f"Upload {remote_name} (epoch {epoch})",
                    )

            # Create/update model card
            self._create_model_card(epoch)

            print(f"[green]Uploaded best model (epoch {epoch}) to HF Hub[/green]")

        except Exception as e:
            print(f"[red]HF upload error: {e}[/red]")

    def _create_model_card(self, epoch: int):
        """Create a README.md model card for the HF repository."""
        # Load config if available
        config_path = self.local_dir / "trm_config.json"
        config = {}
        if config_path.exists():
            with open(config_path, "r") as f:
                config = json.load(f)

        # Get best metrics
        best_metrics = {}
        for m in reversed(self.metrics_history):
            if m.get("is_best"):
                best_metrics = m
                break

        model_card = f"""---
tags:
- sign-language
- asl
- hand-gesture-recognition
- mediapipe
- pytorch
license: mit
datasets:
- asl-alphabet
metrics:
- accuracy
- f1
---

# S-TRM: Stateful Tiny Recursive Model for ASL Recognition

A lightweight, edge-optimized model for American Sign Language (ASL) alphabet recognition
using hand landmarks from MediaPipe.

## Model Description

This model uses the **Stateful Tiny Recursive Model (S-TRM)** architecture, which:
- Processes 63-dimensional hand landmark features (21 points × 3 coords)
- Uses recursive reasoning with latent state for improved accuracy
- Supports stateful inference for real-time video processing
- Is optimized for edge devices (AR glasses, mobile)

## Architecture

| Parameter | Value |
|-----------|-------|
| Input Dimension | {config.get('input_dim', 63)} |
| Number of Classes | {config.get('num_classes', 24)} |
| Latent Dimension | {config.get('latent_dim', 64)} |
| Hidden Dimension | {config.get('hidden_dim', 128)} |
| Latent Iterations (n) | {config.get('n_latent', 6)} |
| Deep Iterations (T) | {config.get('T_deep', 3)} |
| Dropout | {config.get('dropout', 0.1)} |
| Parameters | ~79K |

## Training

- **Best Epoch**: {epoch}
- **Validation Accuracy**: {best_metrics.get('val_acc', 0):.2%}
- **Validation F1 (macro)**: {best_metrics.get('val_f1', 0):.2%}
- **Training Date**: {datetime.now().strftime('%Y-%m-%d')}

## Usage

```python
import torch
from huggingface_hub import hf_hub_download

# Download model
model_path = hf_hub_download(repo_id="{self.repo_id}", filename="pytorch_model.pt")
config_path = hf_hub_download(repo_id="{self.repo_id}", filename="config.json")

# Load config
import json
with open(config_path) as f:
    config = json.load(f)

# Initialize model
from trm_model import TRM
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

# Inference with stateful recurrence
state = None
for landmarks in video_frames:
    x = torch.tensor(landmarks).unsqueeze(0)
    logits, state = model(x, prev_state=state, return_state=True)
    prediction = logits.argmax(dim=-1)
```

## Classes

{', '.join(config.get('class_names', ['A-Z (except J, Z)']))}

## License

MIT

## Citation

```bibtex
@software{{strm_asl,
  title = {{S-TRM: Stateful Tiny Recursive Model for ASL Recognition}},
  year = {{2025}},
  url = {{https://huggingface.co/{self.repo_id}}}
}}
```
"""

        # Upload model card
        try:
            upload_file(
                path_or_fileobj=model_card.encode("utf-8"),
                path_in_repo="README.md",
                repo_id=self.repo_id,
                token=self.token,
                commit_message=f"Update model card (epoch {epoch})",
            )
        except Exception as e:
            print(f"[yellow]Model card upload warning: {e}[/yellow]")

    def on_training_end(self, final_metrics: Dict[str, Any]):
        """Called at the end of training."""
        # Final upload with all metrics
        self.metrics_history.append({
            "event": "training_complete",
            "timestamp": datetime.now().isoformat(),
            **final_metrics,
        })

        metrics_path = self.local_dir / "metrics.json"
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(self.metrics_history, f, indent=2)

        try:
            upload_file(
                path_or_fileobj=str(metrics_path),
                path_in_repo="metrics.json",
                repo_id=self.repo_id,
                token=self.token,
                commit_message="Final training metrics",
            )
            print(f"[green]Training complete! Model available at:[/green] https://huggingface.co/{self.repo_id}")
        except Exception as e:
            print(f"[red]Final upload error: {e}[/red]")


def sync_to_github(
    repo_url: str,
    token: str,
    metrics_path: Path,
    commit_message: str = "Update training metrics",
):
    """
    Sync training metrics to GitHub repository.

    Args:
        repo_url: GitHub repository URL
        token: GitHub personal access token
        metrics_path: Path to metrics.json file
        commit_message: Commit message
    """
    try:
        from git import Repo
        import tempfile

        # Clone repo to temp directory
        with tempfile.TemporaryDirectory() as tmpdir:
            # Add token to URL for authentication
            auth_url = repo_url.replace("https://", f"https://{token}@")
            repo = Repo.clone_from(auth_url, tmpdir)

            # Copy metrics file
            dest = Path(tmpdir) / "outputs" / "logs" / "kaggle_metrics.json"
            dest.parent.mkdir(parents=True, exist_ok=True)

            import shutil
            shutil.copy(metrics_path, dest)

            # Commit and push
            repo.index.add([str(dest.relative_to(tmpdir))])
            repo.index.commit(commit_message)
            repo.remote("origin").push()

            print(f"[green]Synced metrics to GitHub[/green]")

    except Exception as e:
        print(f"[yellow]GitHub sync warning: {e}[/yellow]")


def main(args):
    """Main training function adapted for Kaggle."""
    # Setup paths
    out_models = Path(args.output_dir) / "models"
    out_logs = Path(args.output_dir) / "logs"
    out_figs = Path(args.output_dir) / "figures"
    for p in [out_models, out_logs, out_figs]:
        p.mkdir(parents=True, exist_ok=True)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[bold green]Device:[/bold green] {device}")
    print(f"[bold green]Mode:[/bold green] {'Sequence (BPTT)' if args.sequence_mode else 'Single-frame'}")

    # Get HF token from environment
    hf_token = os.environ.get("HF_TOKEN", args.hf_token)
    hf_repo_id = os.environ.get("HF_REPO_ID", args.hf_repo_id)

    # Initialize HF callback if available
    hf_callback = None
    if HF_AVAILABLE and hf_token and hf_repo_id:
        hf_callback = HuggingFaceCallback(
            repo_id=hf_repo_id,
            token=hf_token,
            local_dir=out_models,
            private=args.hf_private,
        )
    elif hf_token:
        print("[yellow]HF_REPO_ID not set. Skipping HF upload.[/yellow]")

    # Load data
    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    features, labels, class_names, input_dim = load_csv_data(
        csv_path,
        use_raw_landmarks=args.raw_landmarks,
    )
    num_classes = len(class_names)

    # Normalize features
    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels,
        test_size=args.test_size,
        stratify=labels,
        random_state=42,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train,
        test_size=0.1,
        stratify=y_train,
        random_state=42,
    )

    print(f"[cyan]Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}[/cyan]")

    # Create datasets and loaders
    if args.sequence_mode:
        train_ds = SequenceDataset(X_train, y_train, seq_len=args.seq_len, stride=args.seq_stride)
        val_ds_single = LandmarkDataset(X_val, y_val)
        test_ds_single = LandmarkDataset(X_test, y_test)
    else:
        train_ds = LandmarkDataset(X_train, y_train)
        val_ds_single = train_ds.__class__(X_val, y_val)
        test_ds_single = train_ds.__class__(X_test, y_test)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader_single = DataLoader(val_ds_single, batch_size=args.batch_size, shuffle=False)
    test_loader_single = DataLoader(test_ds_single, batch_size=args.batch_size, shuffle=False)

    # Create model
    model = create_trm_model(
        input_dim=input_dim,
        num_classes=num_classes,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        n_latent=args.n_latent,
        T_deep=args.T_deep,
        dropout=args.dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[bold]Model parameters:[/bold] {n_params:,}")

    # Loss and optimizer
    loss_fn = DeepSupervisionLoss(
        num_classes=num_classes,
        weight_decay=args.loss_weight_decay,
        label_smoothing=args.label_smoothing,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.lr * 0.01,
    )

    # Training loop
    best_val_acc = 0.0
    best_epoch = 0
    patience_counter = 0
    train_fn = train_epoch_sequence if args.sequence_mode else train_epoch

    print("\n[bold magenta]Starting training...[/bold magenta]\n")

    training_start = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        # Train
        train_metrics = train_fn(model, train_loader, loss_fn, optimizer, device)

        # Validate
        val_metrics = eval_epoch(model, val_loader_single, loss_fn, device, num_classes=num_classes)

        scheduler.step()

        epoch_time = time.time() - epoch_start

        # Log
        is_best = val_metrics["accuracy"] > best_val_acc
        print(
            f"[bold]Epoch {epoch:3d}[/bold] ({epoch_time:.1f}s) | "
            f"Train Loss: {train_metrics['loss']:.4f} Acc: {train_metrics['accuracy']:.2%} | "
            f"Val Loss: {val_metrics['loss']:.4f} Acc: {val_metrics['accuracy']:.2%} F1: {val_metrics['f1_macro']:.2%}"
            + (" [green]*BEST*[/green]" if is_best else "")
        )

        # Save best model
        if is_best:
            best_val_acc = val_metrics["accuracy"]
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), out_models / "trm_best.pt")

            # Save config
            config = {
                "input_dim": input_dim,
                "num_classes": num_classes,
                "class_names": class_names,
                "latent_dim": args.latent_dim,
                "hidden_dim": args.hidden_dim,
                "n_latent": args.n_latent,
                "T_deep": args.T_deep,
                "dropout": args.dropout,
                "sequence_mode": args.sequence_mode,
                "seq_len": args.seq_len if args.sequence_mode else None,
                "best_val_acc": best_val_acc,
            }
            with open(out_models / "trm_config.json", "w", encoding="utf-8") as f:
                json.dump(config, f, indent=2, ensure_ascii=False)

            # Save scaler
            import joblib
            joblib.dump(scaler, out_models / "trm_scaler.joblib")

        else:
            patience_counter += 1

        # HF callback
        if hf_callback:
            hf_callback.on_epoch_end(epoch, train_metrics, val_metrics, is_best)

        # Early stopping
        if patience_counter >= args.patience:
            print(f"\n[yellow]Early stopping at epoch {epoch}[/yellow]")
            break

    training_time = time.time() - training_start

    # Final evaluation
    model.load_state_dict(torch.load(out_models / "trm_best.pt", map_location=device))
    test_metrics = eval_epoch(
        model, test_loader_single, loss_fn, device,
        num_classes=num_classes, measure_latency=True
    )

    # Print results
    print("\n" + "=" * 60)
    print(f"[bold green]Training completed in {training_time/60:.1f} minutes[/bold green]")
    print(f"[bold green]Best Validation Accuracy:[/bold green] {best_val_acc:.2%} (epoch {best_epoch})")
    print(f"[bold green]Test Accuracy:[/bold green] {test_metrics['accuracy']:.2%}")
    print(f"[bold green]Test F1-Score (macro):[/bold green] {test_metrics['f1_macro']:.2%}")
    if "latency_ms" in test_metrics:
        print(f"[bold green]Inference Latency:[/bold green] {test_metrics['latency_ms']:.2f} ms/sample")
    print("=" * 60)

    # Final metrics
    final_metrics = {
        "best_val_acc": best_val_acc,
        "best_epoch": best_epoch,
        "test_acc": test_metrics["accuracy"],
        "test_f1_macro": test_metrics["f1_macro"],
        "test_latency_ms": test_metrics.get("latency_ms"),
        "training_time_minutes": training_time / 60,
        "total_epochs": epoch,
    }

    # Update config with final metrics
    config["test_acc"] = test_metrics["accuracy"]
    config["test_f1_macro"] = test_metrics["f1_macro"]
    config["test_latency_ms"] = test_metrics.get("latency_ms")
    with open(out_models / "trm_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    # HF final upload
    if hf_callback:
        hf_callback.on_training_end(final_metrics)

    # GitHub sync (optional)
    github_token = os.environ.get("GITHUB_TOKEN")
    if github_token and args.github_repo:
        sync_to_github(
            repo_url=args.github_repo,
            token=github_token,
            metrics_path=out_models / "metrics.json",
            commit_message=f"Kaggle training: acc={test_metrics['accuracy']:.2%}",
        )

    print(f"\n[bold green]Model saved to:[/bold green] {out_models / 'trm_best.pt'}")

    return final_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train S-TRM on Kaggle with HuggingFace Hub integration"
    )

    # Data
    parser.add_argument("--csv", type=str, required=True,
                        help="Path to CSV file with landmarks/features")
    parser.add_argument("--output_dir", type=str, default="/kaggle/working/outputs",
                        help="Output directory for models and logs")
    parser.add_argument("--raw_landmarks", action="store_true", default=False,
                        help="Use raw 63-dim landmarks instead of processed features")
    parser.add_argument("--test_size", type=float, default=0.15,
                        help="Test set ratio")

    # Sequence mode (BPTT)
    parser.add_argument("--sequence_mode", action="store_true", default=False,
                        help="Enable sequence training with BPTT")
    parser.add_argument("--seq_len", type=int, default=16,
                        help="Sequence length for BPTT training")
    parser.add_argument("--seq_stride", type=int, default=4,
                        help="Stride between sequences")

    # Model architecture
    parser.add_argument("--latent_dim", type=int, default=64,
                        help="Latent state dimension")
    parser.add_argument("--hidden_dim", type=int, default=128,
                        help="Hidden layer dimension")
    parser.add_argument("--n_latent", type=int, default=6,
                        help="Latent recursion iterations")
    parser.add_argument("--T_deep", type=int, default=3,
                        help="Deep recursion iterations")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout probability")

    # Training
    parser.add_argument("--epochs", type=int, default=100,
                        help="Maximum training epochs")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help="AdamW weight decay")
    parser.add_argument("--patience", type=int, default=20,
                        help="Early stopping patience")
    parser.add_argument("--num_workers", type=int, default=2,
                        help="DataLoader workers")

    # Deep Supervision
    parser.add_argument("--loss_weight_decay", type=float, default=0.9,
                        help="Weight decay for deep supervision")
    parser.add_argument("--label_smoothing", type=float, default=0.1,
                        help="Label smoothing")

    # Hugging Face Hub
    parser.add_argument("--hf_token", type=str, default=None,
                        help="HuggingFace token (or set HF_TOKEN env var)")
    parser.add_argument("--hf_repo_id", type=str, default=None,
                        help="HuggingFace repo ID (or set HF_REPO_ID env var)")
    parser.add_argument("--hf_private", action="store_true", default=False,
                        help="Make HF repository private")

    # GitHub sync
    parser.add_argument("--github_repo", type=str,
                        default="https://github.com/Hakim78/AI-SignLanguage.git",
                        help="GitHub repository URL for metrics sync")

    args = parser.parse_args()
    main(args)
