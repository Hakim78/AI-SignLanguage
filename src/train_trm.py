# src/train_trm.py
"""
Training script for S-TRM (Stateful Tiny Recursive Model) on ASL Landmarks.

Usage:
    # Single-frame training (static signs):
    uv run python -m src.train_trm --csv data/landmarks/features.csv --epochs 100

    # Sequence training with BPTT (dynamic signs like J, Z):
    uv run python -m src.train_trm --csv data/landmarks/sequences.csv --sequence_mode --seq_len 16

The script supports two input modes:
1. Raw landmarks (63 dims): 21 points * 3 coords (x, y, z)
2. Processed features (102 dims): From landmark_features.py

For best results with TRM, raw landmarks (63 dims) are recommended.

Sequence mode enables BPTT (Backpropagation Through Time) for learning
temporal patterns in dynamic signs.
"""
from __future__ import annotations
import argparse
import json
import csv
import time
from pathlib import Path
from typing import Tuple, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_recall_fscore_support
from rich import print
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn
from rich.table import Table
from rich.console import Console

from trm_model import TRM, DeepSupervisionLoss, create_trm_model

console = Console()


class LandmarkDataset(Dataset):
    """
    Dataset for ASL landmarks (single-frame mode).

    Supports loading from CSV with either:
    - Raw landmarks (columns: x0,y0,z0,...,x20,y20,z20,label) -> 63 dims
    - Processed features (columns: f0,...,f101,label) -> 102 dims
    """
    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
    ):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.labels[idx]


class SequenceDataset(Dataset):
    """
    Dataset for ASL landmark sequences (BPTT mode).

    Creates sequences of consecutive frames for training with
    Backpropagation Through Time (BPTT).

    Input shape: [Seq_Len, input_dim]
    Output shape: [Seq_Len] (labels for each frame)
    """
    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        seq_len: int = 16,
        stride: int = 1,
    ):
        """
        Args:
            features: [N, input_dim] array of landmarks
            labels: [N] array of labels
            seq_len: Sequence length for BPTT
            stride: Stride between sequences (1 = overlapping, seq_len = non-overlapping)
        """
        self.features = features
        self.labels = labels
        self.seq_len = seq_len
        self.stride = stride

        # Calculate valid sequence start indices
        self.indices = []
        for i in range(0, len(features) - seq_len + 1, stride):
            # Only include sequences where all frames have the same label
            # (for consistent training on static signs)
            seq_labels = labels[i:i + seq_len]
            if len(set(seq_labels)) == 1:
                self.indices.append(i)

        print(f"[cyan]SequenceDataset: {len(self.indices)} sequences of length {seq_len}[/cyan]")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        start = self.indices[idx]
        end = start + self.seq_len

        x = torch.tensor(self.features[start:end], dtype=torch.float32)  # [seq_len, input_dim]
        y = torch.tensor(self.labels[start:end], dtype=torch.long)  # [seq_len]

        return x, y


def load_csv_data(
    csv_path: Path,
    use_raw_landmarks: bool = False,
) -> Tuple[np.ndarray, np.ndarray, List[str], int]:
    """
    Load data from CSV file.

    Args:
        csv_path: Path to CSV file
        use_raw_landmarks: If True, expect raw x,y,z columns. Otherwise, expect f0-fN columns.

    Returns:
        features: numpy array [N, input_dim]
        labels_encoded: numpy array [N] (integer labels)
        class_names: list of class names
        input_dim: feature dimension
    """
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for r in reader:
            rows.append(r)

    if not rows:
        raise ValueError(f"CSV file is empty: {csv_path}")

    # Detect input format
    if "x0" in fieldnames or "landmark_0_x" in fieldnames:
        # Raw landmarks format
        print("[cyan]Detected raw landmarks format[/cyan]")
        # Try to extract 21 points * 3 coords = 63 dims
        features = []
        for row in rows:
            feat = []
            for i in range(21):
                for coord in ["x", "y", "z"]:
                    key = f"x{i}" if coord == "x" else (f"y{i}" if coord == "y" else f"z{i}")
                    alt_key = f"landmark_{i}_{coord}"
                    if key in row:
                        feat.append(float(row[key]))
                    elif alt_key in row:
                        feat.append(float(row[alt_key]))
                    else:
                        # If z is missing, use 0
                        feat.append(0.0)
            features.append(feat)
        features = np.array(features, dtype=np.float32)
        input_dim = 63

    elif "f0" in fieldnames:
        # Processed features format (from landmark_features.py or extract scripts)
        print("[cyan]Detected processed features format[/cyan]")
        # Count feature columns
        feat_cols = [k for k in fieldnames if k.startswith("f") and k[1:].isdigit()]
        n_feats = len(feat_cols)
        print(f"[cyan]Found {n_feats} feature columns[/cyan]")

        features = []
        for row in rows:
            feat = [float(row[f"f{i}"]) for i in range(n_feats)]
            features.append(feat)
        features = np.array(features, dtype=np.float32)
        input_dim = n_feats

    else:
        raise ValueError(f"Unknown CSV format. Columns: {fieldnames[:10]}...")

    # Extract labels
    label_col = "label" if "label" in fieldnames else "class"
    labels_str = [row[label_col] for row in rows]

    # Encode labels
    le = LabelEncoder()
    labels_encoded = le.fit_transform(labels_str)
    class_names = list(le.classes_)

    print(f"[green]Loaded {len(features)} samples, {input_dim} features, {len(class_names)} classes[/green]")
    print(f"[green]Classes: {class_names}[/green]")

    return features, labels_encoded, class_names, input_dim


def train_epoch(
    model: TRM,
    loader: DataLoader,
    loss_fn: DeepSupervisionLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: torch.cuda.amp.GradScaler = None,
) -> dict:
    """Train for one epoch (single-frame mode)."""
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    n_batches = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad(set_to_none=True)

        # Forward with deep supervision
        final_out, all_outputs = model(x, return_all_outputs=True)
        loss, metrics = loss_fn(final_out, all_outputs, y)

        # Backward
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += metrics["total_loss"]
        total_acc += metrics["accuracy"]
        n_batches += 1

    return {
        "loss": total_loss / n_batches,
        "accuracy": total_acc / n_batches,
    }


def train_epoch_sequence(
    model: TRM,
    loader: DataLoader,
    loss_fn: DeepSupervisionLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> dict:
    """
    Train for one epoch with BPTT (sequence mode).

    Processes sequences frame-by-frame with stateful recurrence,
    computing loss at each timestep and backpropagating through time.
    """
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    n_batches = 0

    for x_seq, y_seq in loader:
        # x_seq: [B, Seq_Len, input_dim]
        # y_seq: [B, Seq_Len]
        x_seq, y_seq = x_seq.to(device), y_seq.to(device)
        B, T, D = x_seq.shape

        optimizer.zero_grad(set_to_none=True)

        # Forward through sequence with stateful recurrence
        outputs, states, all_intermediates = model.forward_sequence(x_seq, return_all_outputs=True)
        # outputs: [B, T, num_classes]

        # Compute loss across all timesteps
        batch_loss = torch.tensor(0.0, device=device)
        batch_correct = 0
        batch_total = 0

        for t in range(T):
            final_t = outputs[:, t, :]  # [B, num_classes]
            target_t = y_seq[:, t]  # [B]
            intermediates_t = all_intermediates[t]

            loss_t, metrics_t = loss_fn(final_t, intermediates_t, target_t)
            batch_loss = batch_loss + loss_t

            batch_correct += (final_t.argmax(dim=-1) == target_t).sum().item()
            batch_total += B

        # Average loss over timesteps
        batch_loss = batch_loss / T

        # Backward (BPTT)
        batch_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += batch_loss.item()
        total_acc += batch_correct / batch_total
        n_batches += 1

    return {
        "loss": total_loss / n_batches,
        "accuracy": total_acc / n_batches,
    }


@torch.no_grad()
def eval_epoch(
    model: TRM,
    loader: DataLoader,
    loss_fn: DeepSupervisionLoss,
    device: torch.device,
    num_classes: int = None,
    measure_latency: bool = False,
) -> dict:
    """
    Evaluate for one epoch with extended metrics.

    Returns:
        dict with: loss, accuracy, predictions, labels, f1_macro, f1_per_class,
                   per_class_accuracy, latency_ms (if measure_latency=True)
    """
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    n_batches = 0

    all_preds = []
    all_labels = []
    latencies = []

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        # Measure inference latency
        if measure_latency:
            if device.type == "cuda":
                torch.cuda.synchronize()
            t_start = time.perf_counter()

        # Forward (eval mode - still get all outputs for consistent loss)
        final_out, all_outputs = model(x, return_all_outputs=True)

        if measure_latency:
            if device.type == "cuda":
                torch.cuda.synchronize()
            t_end = time.perf_counter()
            # Latency per sample in ms
            latencies.append((t_end - t_start) * 1000 / x.size(0))

        loss, metrics = loss_fn(final_out, all_outputs, y)

        total_loss += metrics["total_loss"]
        total_acc += metrics["accuracy"]
        n_batches += 1

        all_preds.extend(final_out.argmax(dim=-1).cpu().tolist())
        all_labels.extend(y.cpu().tolist())

    # Compute extended metrics
    all_preds_np = np.array(all_preds)
    all_labels_np = np.array(all_labels)

    # F1-score (macro and per-class)
    f1_macro = f1_score(all_labels_np, all_preds_np, average="macro", zero_division=0)
    f1_per_class = f1_score(all_labels_np, all_preds_np, average=None, zero_division=0)

    # Per-class accuracy
    per_class_acc = []
    if num_classes is not None:
        for c in range(num_classes):
            mask = all_labels_np == c
            if mask.sum() > 0:
                acc_c = (all_preds_np[mask] == c).mean()
                per_class_acc.append(float(acc_c))
            else:
                per_class_acc.append(0.0)

    result = {
        "loss": total_loss / n_batches,
        "accuracy": total_acc / n_batches,
        "predictions": all_preds,
        "labels": all_labels,
        "f1_macro": f1_macro,
        "f1_per_class": f1_per_class.tolist(),
        "per_class_accuracy": per_class_acc,
    }

    if measure_latency and latencies:
        result["latency_ms"] = np.mean(latencies)
        result["latency_std_ms"] = np.std(latencies)

    return result


def main(args):
    # Setup paths
    out_models = Path("outputs/models")
    out_logs = Path("outputs/logs")
    out_figs = Path("outputs/figures")
    for p in [out_models, out_logs, out_figs]:
        p.mkdir(parents=True, exist_ok=True)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[bold green]Device:[/bold green] {device}")
    print(f"[bold green]Mode:[/bold green] {'Sequence (BPTT)' if args.sequence_mode else 'Single-frame'}")

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
        # Sequence mode for BPTT training
        train_ds = SequenceDataset(X_train, y_train, seq_len=args.seq_len, stride=args.seq_stride)
        val_ds = SequenceDataset(X_val, y_val, seq_len=args.seq_len, stride=args.seq_len)
        test_ds = SequenceDataset(X_test, y_test, seq_len=args.seq_len, stride=args.seq_len)
    else:
        # Single-frame mode
        train_ds = LandmarkDataset(X_train, y_train)
        val_ds = LandmarkDataset(X_val, y_val)
        test_ds = LandmarkDataset(X_test, y_test)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    # For evaluation, always use single-frame loaders for consistent metrics
    if args.sequence_mode:
        val_ds_single = LandmarkDataset(X_val, y_val)
        test_ds_single = LandmarkDataset(X_test, y_test)
    else:
        val_ds_single = val_ds
        test_ds_single = test_ds

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

    # Select training function based on mode
    train_fn = train_epoch_sequence if args.sequence_mode else train_epoch

    print("\n[bold magenta]Starting training...[/bold magenta]\n")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("[cyan]Training", total=args.epochs)

        for epoch in range(1, args.epochs + 1):
            # Train (sequence or single-frame mode)
            if args.sequence_mode:
                train_metrics = train_fn(model, train_loader, loss_fn, optimizer, device)
            else:
                train_metrics = train_fn(model, train_loader, loss_fn, optimizer, device)

            # Validate (always use single-frame for consistent metrics)
            val_metrics = eval_epoch(model, val_loader_single, loss_fn, device, num_classes=num_classes)

            scheduler.step()

            # Log
            progress.update(task, advance=1)
            console.print(
                f"[bold]Epoch {epoch:3d}[/bold] | "
                f"Train Loss: {train_metrics['loss']:.4f} Acc: {train_metrics['accuracy']:.2%} | "
                f"Val Loss: {val_metrics['loss']:.4f} Acc: {val_metrics['accuracy']:.2%} F1: {val_metrics['f1_macro']:.2%}"
            )

            # Save best model
            if val_metrics["accuracy"] > best_val_acc:
                best_val_acc = val_metrics["accuracy"]
                best_epoch = epoch
                patience_counter = 0
                torch.save(model.state_dict(), out_models / "trm_best.pt")
            else:
                patience_counter += 1

            # Early stopping
            if patience_counter >= args.patience:
                console.print(f"\n[yellow]Early stopping at epoch {epoch}[/yellow]")
                break

    # Load best model and evaluate on test set with full metrics
    model.load_state_dict(torch.load(out_models / "trm_best.pt", map_location=device))
    test_metrics = eval_epoch(
        model, test_loader_single, loss_fn, device,
        num_classes=num_classes, measure_latency=True
    )

    # Print results
    console.print("\n" + "=" * 60)
    console.print(f"[bold green]Best Validation Accuracy:[/bold green] {best_val_acc:.2%} (epoch {best_epoch})")
    console.print(f"[bold green]Test Accuracy:[/bold green] {test_metrics['accuracy']:.2%}")
    console.print(f"[bold green]Test F1-Score (macro):[/bold green] {test_metrics['f1_macro']:.2%}")
    if "latency_ms" in test_metrics:
        console.print(f"[bold green]Inference Latency:[/bold green] {test_metrics['latency_ms']:.2f} ms/sample (±{test_metrics['latency_std_ms']:.2f})")
    console.print("=" * 60)

    # Per-class accuracy table
    if test_metrics["per_class_accuracy"]:
        table = Table(title="Per-Class Metrics")
        table.add_column("Class", style="cyan")
        table.add_column("Accuracy", justify="right")
        table.add_column("F1-Score", justify="right")

        for i, name in enumerate(class_names):
            acc = test_metrics["per_class_accuracy"][i] if i < len(test_metrics["per_class_accuracy"]) else 0
            f1 = test_metrics["f1_per_class"][i] if i < len(test_metrics["f1_per_class"]) else 0
            table.add_row(name, f"{acc:.2%}", f"{f1:.2%}")

        console.print(table)

    # Save class names and config
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
        "test_acc": test_metrics["accuracy"],
        "test_f1_macro": test_metrics["f1_macro"],
        "test_latency_ms": test_metrics.get("latency_ms"),
    }
    with open(out_models / "trm_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    # Save scaler
    import joblib
    joblib.dump(scaler, out_models / "trm_scaler.joblib")

    console.print(f"\n[bold green]Model saved to:[/bold green] {out_models / 'trm_best.pt'}")
    console.print(f"[bold green]Config saved to:[/bold green] {out_models / 'trm_config.json'}")

    # Per-class accuracy
    from sklearn.metrics import classification_report, confusion_matrix
    report = classification_report(
        test_metrics["labels"],
        test_metrics["predictions"],
        target_names=class_names,
        digits=4,
    )
    console.print("\n[bold cyan]Classification Report (Test):[/bold cyan]")
    console.print(report)

    # Save confusion matrix
    from src.utils_metrics import plot_confusion_matrix, plot_pr_curve

    y_test_arr = np.array(test_metrics["labels"])
    y_pred_arr = np.array(test_metrics["predictions"])

    plot_confusion_matrix(
        y_test_arr,
        y_pred_arr,
        class_names,
        str(out_figs / "trm_confusion_matrix.png"),
    )
    console.print(f"[green]Confusion matrix saved to:[/green] {out_figs / 'trm_confusion_matrix.png'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train S-TRM (Stateful TRM) model on ASL landmarks")

    # Data
    parser.add_argument("--csv", type=str, default="data/landmarks/features.csv",
                        help="Path to CSV file with landmarks/features")
    parser.add_argument("--raw_landmarks", action="store_true", default=False,
                        help="Use raw 63-dim landmarks instead of processed features")
    parser.add_argument("--test_size", type=float, default=0.15,
                        help="Test set ratio")

    # Sequence mode (BPTT)
    parser.add_argument("--sequence_mode", action="store_true", default=False,
                        help="Enable sequence training with BPTT for dynamic signs")
    parser.add_argument("--seq_len", type=int, default=16,
                        help="Sequence length for BPTT training")
    parser.add_argument("--seq_stride", type=int, default=4,
                        help="Stride between sequences (overlap control)")

    # Model architecture (TRM paper defaults)
    parser.add_argument("--latent_dim", type=int, default=64,
                        help="Latent state dimension")
    parser.add_argument("--hidden_dim", type=int, default=128,
                        help="Hidden layer dimension in TinyBlocks")
    parser.add_argument("--n_latent", type=int, default=6,
                        help="Latent recursion iterations (paper: 6)")
    parser.add_argument("--T_deep", type=int, default=3,
                        help="Deep recursion iterations (paper: 3)")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout probability for regularization (0.1-0.3)")

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

    # Deep Supervision
    parser.add_argument("--loss_weight_decay", type=float, default=0.9,
                        help="Weight decay for deep supervision (later outputs weighted more)")
    parser.add_argument("--label_smoothing", type=float, default=0.1,
                        help="Label smoothing for cross-entropy")

    args = parser.parse_args()
    main(args)
