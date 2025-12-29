# src/process_videos.py
"""
Video to TRM Sequence CSV Converter.

Converts raw video datasets (SignAlphaSet, WLASL, MS-ASL) into the TRM Sequence format
compatible with SequenceDataset in train_trm.py.

Features:
- Extracts 63 raw landmarks (21 points * 3 coords: x, y, z) per frame using MediaPipe Hands
- Handles occlusion with zero-padding or linear interpolation
- Outputs CSV compatible with S-TRM sequence training (BPTT)

Usage:
    # Process a folder of videos organized by class
    uv run python -m src.process_videos --input data/videos --output data/landmarks/sequences.csv

    # With interpolation for missing frames
    uv run python -m src.process_videos --input data/videos --output data/sequences.csv --interpolate

    # Kaggle usage
    python process_videos.py --input /kaggle/input/wlasl/videos --output /kaggle/working/sequences.csv

Directory Structure Expected:
    data/videos/
        A/
            video1.mp4
            video2.mp4
        B/
            video1.mp4
        ...

Output CSV Format:
    frame_idx,video_id,x0,y0,z0,x1,y1,z1,...,x20,y20,z20,label
    0,video_A_001,0.45,0.32,0.01,...,A
    1,video_A_001,0.46,0.33,0.01,...,A
    ...

Dependencies:
    pip install mediapipe opencv-python numpy tqdm rich
"""
from __future__ import annotations
import argparse
import csv
import os
from pathlib import Path
from typing import List, Tuple, Optional
import warnings

import cv2
import numpy as np
import mediapipe as mp
from tqdm import tqdm
from rich import print
from rich.console import Console
from rich.table import Table

console = Console()

# Constants
NUM_LANDMARKS = 21
COORDS_PER_LANDMARK = 3  # x, y, z
FEATURE_DIM = NUM_LANDMARKS * COORDS_PER_LANDMARK  # 63


def extract_landmarks_from_frame(
    frame: np.ndarray,
    hands: mp.solutions.hands.Hands,
) -> Optional[np.ndarray]:
    """
    Extract 63-dim raw landmarks from a single frame.

    Args:
        frame: BGR image frame
        hands: MediaPipe Hands instance

    Returns:
        numpy array of shape (63,) with [x0,y0,z0,...,x20,y20,z20] or None if no hand detected
    """
    # Convert BGR to RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        # Take the first detected hand
        hand_landmarks = result.multi_hand_landmarks[0]
        features = []
        for lm in hand_landmarks.landmark:
            features.extend([lm.x, lm.y, lm.z])
        return np.array(features, dtype=np.float32)

    return None


def interpolate_missing_frames(
    landmarks_sequence: List[Optional[np.ndarray]],
) -> List[np.ndarray]:
    """
    Interpolate missing frames using linear interpolation.

    For gaps between valid frames, linearly interpolates.
    For leading/trailing None values, uses nearest valid frame.

    Args:
        landmarks_sequence: List of landmark arrays, with None for missing frames

    Returns:
        List of landmark arrays with all Nones filled
    """
    n = len(landmarks_sequence)
    if n == 0:
        return []

    # Find indices of valid (non-None) frames
    valid_indices = [i for i, lm in enumerate(landmarks_sequence) if lm is not None]

    if len(valid_indices) == 0:
        # All frames are None - return zeros
        return [np.zeros(FEATURE_DIM, dtype=np.float32) for _ in range(n)]

    result = [None] * n

    # Copy valid frames
    for idx in valid_indices:
        result[idx] = landmarks_sequence[idx].copy()

    # Fill leading Nones with first valid frame
    first_valid = valid_indices[0]
    for i in range(first_valid):
        result[i] = result[first_valid].copy()

    # Fill trailing Nones with last valid frame
    last_valid = valid_indices[-1]
    for i in range(last_valid + 1, n):
        result[i] = result[last_valid].copy()

    # Interpolate gaps between valid frames
    for i in range(len(valid_indices) - 1):
        start_idx = valid_indices[i]
        end_idx = valid_indices[i + 1]

        if end_idx - start_idx > 1:
            # There's a gap to interpolate
            start_lm = result[start_idx]
            end_lm = result[end_idx]
            gap_size = end_idx - start_idx

            for j in range(1, gap_size):
                alpha = j / gap_size
                interpolated = (1 - alpha) * start_lm + alpha * end_lm
                result[start_idx + j] = interpolated.astype(np.float32)

    return result


def pad_missing_frames(
    landmarks_sequence: List[Optional[np.ndarray]],
) -> List[np.ndarray]:
    """
    Pad missing frames with zeros.

    Args:
        landmarks_sequence: List of landmark arrays, with None for missing frames

    Returns:
        List of landmark arrays with Nones replaced by zeros
    """
    result = []
    for lm in landmarks_sequence:
        if lm is not None:
            result.append(lm)
        else:
            result.append(np.zeros(FEATURE_DIM, dtype=np.float32))
    return result


def process_video(
    video_path: Path,
    hands: mp.solutions.hands.Hands,
    max_frames: int = None,
    target_fps: int = None,
) -> Tuple[List[np.ndarray], dict]:
    """
    Process a single video and extract landmarks for all frames.

    Args:
        video_path: Path to video file
        hands: MediaPipe Hands instance
        max_frames: Maximum number of frames to extract (None = all)
        target_fps: Target FPS for frame sampling (None = use original)

    Returns:
        Tuple of (list of landmark arrays with None for missing, metadata dict)
    """
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    # Get video properties
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Calculate frame sampling rate
    frame_skip = 1
    if target_fps is not None and original_fps > target_fps:
        frame_skip = int(original_fps / target_fps)

    landmarks_sequence = []
    frames_processed = 0
    frames_with_hand = 0
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Skip frames if downsampling
        if frame_idx % frame_skip != 0:
            frame_idx += 1
            continue

        # Extract landmarks
        landmarks = extract_landmarks_from_frame(frame, hands)
        landmarks_sequence.append(landmarks)

        frames_processed += 1
        if landmarks is not None:
            frames_with_hand += 1

        if max_frames is not None and frames_processed >= max_frames:
            break

        frame_idx += 1

    cap.release()

    metadata = {
        "video_path": str(video_path),
        "original_fps": original_fps,
        "total_frames": total_frames,
        "frames_processed": frames_processed,
        "frames_with_hand": frames_with_hand,
        "detection_rate": frames_with_hand / max(1, frames_processed),
        "width": width,
        "height": height,
    }

    return landmarks_sequence, metadata


def process_video_folder(
    input_dir: Path,
    output_csv: Path,
    interpolate: bool = False,
    max_frames: int = None,
    target_fps: int = 15,
    min_detection_rate: float = 0.3,
    video_extensions: Tuple[str, ...] = (".mp4", ".avi", ".mov", ".webm", ".mkv"),
) -> dict:
    """
    Process all videos in a folder organized by class labels.

    Args:
        input_dir: Root directory containing class subfolders
        output_csv: Path to output CSV file
        interpolate: Whether to interpolate missing frames (vs zero-padding)
        max_frames: Maximum frames per video (None = all)
        target_fps: Target FPS for frame sampling
        min_detection_rate: Minimum hand detection rate to include video
        video_extensions: Tuple of valid video extensions

    Returns:
        Statistics dictionary
    """
    # Initialize MediaPipe Hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    # Find all class folders
    class_folders = [d for d in input_dir.iterdir() if d.is_dir()]
    if not class_folders:
        # Try to find videos directly in input_dir (flat structure)
        print("[yellow]No class subfolders found. Looking for videos directly in input folder.[/yellow]")
        class_folders = [input_dir]

    # Prepare output CSV
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    # CSV header
    header = ["frame_idx", "video_id"]
    for i in range(NUM_LANDMARKS):
        header.extend([f"x{i}", f"y{i}", f"z{i}"])
    header.append("label")

    stats = {
        "total_videos": 0,
        "processed_videos": 0,
        "skipped_videos": 0,
        "total_frames": 0,
        "classes": {},
    }

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        # Process each class folder
        for class_folder in tqdm(class_folders, desc="Processing classes"):
            if class_folder == input_dir:
                label = "unknown"
                videos = list(input_dir.glob("*"))
            else:
                label = class_folder.name
                videos = list(class_folder.iterdir())

            # Filter to video files
            videos = [v for v in videos if v.suffix.lower() in video_extensions]

            if label not in stats["classes"]:
                stats["classes"][label] = {"videos": 0, "frames": 0}

            for video_path in tqdm(videos, desc=f"Class {label}", leave=False):
                stats["total_videos"] += 1

                try:
                    # Process video
                    landmarks_seq, metadata = process_video(
                        video_path, hands, max_frames, target_fps
                    )

                    # Check detection rate
                    if metadata["detection_rate"] < min_detection_rate:
                        stats["skipped_videos"] += 1
                        continue

                    # Handle missing frames
                    if interpolate:
                        landmarks_seq = interpolate_missing_frames(landmarks_seq)
                    else:
                        landmarks_seq = pad_missing_frames(landmarks_seq)

                    # Generate video ID
                    video_id = f"{label}_{video_path.stem}"

                    # Write frames to CSV
                    for frame_idx, landmarks in enumerate(landmarks_seq):
                        row = [frame_idx, video_id]
                        row.extend(landmarks.tolist())
                        row.append(label)
                        writer.writerow(row)

                    stats["processed_videos"] += 1
                    stats["total_frames"] += len(landmarks_seq)
                    stats["classes"][label]["videos"] += 1
                    stats["classes"][label]["frames"] += len(landmarks_seq)

                except Exception as e:
                    print(f"[red]Error processing {video_path}: {e}[/red]")
                    stats["skipped_videos"] += 1

    hands.close()
    return stats


def create_sequence_csv_from_frames(
    frames_csv: Path,
    output_csv: Path,
    seq_len: int = 16,
    stride: int = 8,
) -> dict:
    """
    Convert a frame-by-frame CSV into a sequence-ready CSV.

    Groups consecutive frames from the same video into sequences of fixed length.

    Args:
        frames_csv: Input CSV with frame-by-frame data
        output_csv: Output CSV with sequences
        seq_len: Length of each sequence (T in S-TRM)
        stride: Stride between sequences

    Returns:
        Statistics dictionary
    """
    # Read all frames
    with open(frames_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        raise ValueError("Input CSV is empty")

    # Group by video_id
    videos = {}
    for row in rows:
        vid_id = row["video_id"]
        if vid_id not in videos:
            videos[vid_id] = []
        videos[vid_id].append(row)

    # Get feature columns
    feature_cols = [col for col in rows[0].keys() if col.startswith(("x", "y", "z"))]

    # Prepare output
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    # New header for sequences: seq_id, t0_x0, t0_y0, ..., t15_x20, t15_y20, t15_z20, label
    header = ["seq_id"]
    for t in range(seq_len):
        for feat in feature_cols:
            header.append(f"t{t}_{feat}")
    header.append("label")

    stats = {"sequences": 0, "videos_used": 0}

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for vid_id, frames in videos.items():
            # Sort by frame_idx
            frames.sort(key=lambda x: int(x["frame_idx"]))

            if len(frames) < seq_len:
                continue

            stats["videos_used"] += 1
            label = frames[0]["label"]

            # Create sequences with stride
            for start in range(0, len(frames) - seq_len + 1, stride):
                seq_frames = frames[start:start + seq_len]

                row = [f"{vid_id}_seq{start}"]
                for frame in seq_frames:
                    for feat in feature_cols:
                        row.append(frame[feat])
                row.append(label)

                writer.writerow(row)
                stats["sequences"] += 1

    return stats


def main(args):
    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"Input path not found: {input_path}")

    console.print(f"[bold green]Processing videos from:[/bold green] {input_path}")
    console.print(f"[bold green]Output CSV:[/bold green] {output_path}")
    console.print(f"[bold green]Interpolation:[/bold green] {'Enabled' if args.interpolate else 'Disabled (zero-padding)'}")
    console.print(f"[bold green]Target FPS:[/bold green] {args.target_fps}")
    console.print(f"[bold green]Max frames per video:[/bold green] {args.max_frames or 'All'}")

    # Process videos
    stats = process_video_folder(
        input_dir=input_path,
        output_csv=output_path,
        interpolate=args.interpolate,
        max_frames=args.max_frames,
        target_fps=args.target_fps,
        min_detection_rate=args.min_detection,
    )

    # Print statistics
    console.print("\n" + "=" * 60)
    console.print("[bold cyan]Processing Statistics[/bold cyan]")
    console.print("=" * 60)
    console.print(f"Total videos found: {stats['total_videos']}")
    console.print(f"Videos processed: {stats['processed_videos']}")
    console.print(f"Videos skipped: {stats['skipped_videos']}")
    console.print(f"Total frames extracted: {stats['total_frames']}")

    # Per-class table
    if stats["classes"]:
        table = Table(title="Per-Class Statistics")
        table.add_column("Class", style="cyan")
        table.add_column("Videos", justify="right")
        table.add_column("Frames", justify="right")

        for label, class_stats in sorted(stats["classes"].items()):
            table.add_row(label, str(class_stats["videos"]), str(class_stats["frames"]))

        console.print(table)

    console.print(f"\n[bold green]Output saved to:[/bold green] {output_path}")

    # Optionally create sequence-ready CSV
    if args.create_sequences:
        seq_output = output_path.parent / f"{output_path.stem}_sequences{output_path.suffix}"
        console.print(f"\n[bold yellow]Creating sequence CSV (T={args.seq_len})...[/bold yellow]")

        seq_stats = create_sequence_csv_from_frames(
            frames_csv=output_path,
            output_csv=seq_output,
            seq_len=args.seq_len,
            stride=args.seq_stride,
        )

        console.print(f"Sequences created: {seq_stats['sequences']}")
        console.print(f"Videos used: {seq_stats['videos_used']}")
        console.print(f"[bold green]Sequence CSV saved to:[/bold green] {seq_output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert video datasets to TRM Sequence CSV format"
    )

    # Input/Output
    parser.add_argument(
        "--input", type=str, required=True,
        help="Input directory containing class subfolders with videos"
    )
    parser.add_argument(
        "--output", type=str, default="data/landmarks/sequences.csv",
        help="Output CSV file path"
    )

    # Processing options
    parser.add_argument(
        "--interpolate", action="store_true", default=False,
        help="Interpolate missing frames (default: zero-padding)"
    )
    parser.add_argument(
        "--target_fps", type=int, default=15,
        help="Target FPS for frame sampling (default: 15)"
    )
    parser.add_argument(
        "--max_frames", type=int, default=None,
        help="Maximum frames per video (default: all)"
    )
    parser.add_argument(
        "--min_detection", type=float, default=0.3,
        help="Minimum hand detection rate to include video (default: 0.3)"
    )

    # Sequence options
    parser.add_argument(
        "--create_sequences", action="store_true", default=False,
        help="Also create a sequence-ready CSV (flattened sequences)"
    )
    parser.add_argument(
        "--seq_len", type=int, default=16,
        help="Sequence length for S-TRM (default: 16)"
    )
    parser.add_argument(
        "--seq_stride", type=int, default=8,
        help="Stride between sequences (default: 8)"
    )

    args = parser.parse_args()
    main(args)
