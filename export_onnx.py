"""
Export S-TRM model to ONNX format for client-side inference.

Usage:
    python export_onnx.py

Output:
    - web/model.onnx: The ONNX model
    - web/config.json: Model configuration (class names, scaler params)
"""
import torch
import torch.nn as nn
import json
import numpy as np
import joblib
from pathlib import Path

# Import model architecture
import sys
sys.path.insert(0, str(Path(__file__).parent / "src"))
from trm_model import TRM


def export_model_to_onnx():
    """Export the trained S-TRM model to ONNX format."""

    # Paths
    model_path = Path("outputs/models/trm_best.pt")
    config_path = Path("outputs/models/trm_config.json")
    scaler_path = Path("outputs/models/trm_scaler.joblib")
    output_dir = Path("web")
    output_dir.mkdir(exist_ok=True)

    # Load config
    print("[1/4] Loading configuration...")
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    class_names = config["class_names"]
    num_classes = config["num_classes"]
    input_dim = config.get("input_dim", 63)

    print(f"      Classes: {num_classes}")
    print(f"      Input dim: {input_dim}")

    # Load model
    print("[2/4] Loading PyTorch model...")
    model = TRM(
        input_dim=input_dim,
        num_classes=num_classes,
        latent_dim=config.get("latent_dim", 64),
        hidden_dim=config.get("hidden_dim", 128),
        n_latent=config.get("n_latent", 6),
        T_deep=config.get("T_deep", 3),
        dropout=config.get("dropout", 0.1)
    )

    state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"      Parameters: {total_params:,}")

    # Load scaler
    print("[3/4] Loading scaler...")
    scaler = joblib.load(scaler_path)
    scaler_mean = scaler.mean_.tolist()
    scaler_scale = scaler.scale_.tolist()

    # Export to ONNX
    print("[4/4] Exporting to ONNX...")

    # Create wrapper that doesn't use stateful inference
    class TRMInference(nn.Module):
        def __init__(self, trm_model):
            super().__init__()
            self.model = trm_model

        def forward(self, x):
            # x: [batch, 63] - normalized landmarks
            # Returns: [batch, num_classes] - logits
            with torch.no_grad():
                logits = self.model(x, prev_state=None, return_state=False)
            return logits

    inference_model = TRMInference(model)
    inference_model.eval()

    # Dummy input
    dummy_input = torch.randn(1, input_dim)

    # Export
    onnx_path = output_dir / "model.onnx"
    torch.onnx.export(
        inference_model,
        dummy_input,
        str(onnx_path),
        opset_version=12,
        input_names=["landmarks"],
        output_names=["logits"],
        dynamic_axes={
            "landmarks": {0: "batch_size"},
            "logits": {0: "batch_size"}
        },
        do_constant_folding=True
    )

    print(f"      Saved: {onnx_path}")

    # Save web config (includes scaler params for JS)
    web_config = {
        "class_names": class_names,
        "num_classes": num_classes,
        "input_dim": input_dim,
        "scaler": {
            "mean": scaler_mean,
            "scale": scaler_scale
        },
        "dynamic_classes": ["J", "Z", "HELLO", "THANKYOU", "SORRY", "YES", "NO"]
    }

    config_output_path = output_dir / "config.json"
    with open(config_output_path, "w", encoding="utf-8") as f:
        json.dump(web_config, f, indent=2)

    print(f"      Saved: {config_output_path}")

    # Verify export
    print("\n[Verification]")
    try:
        import onnxruntime as ort
        session = ort.InferenceSession(str(onnx_path))

        # Test inference
        test_input = np.random.randn(1, input_dim).astype(np.float32)
        outputs = session.run(None, {"landmarks": test_input})

        print(f"      ONNX input shape: {test_input.shape}")
        print(f"      ONNX output shape: {outputs[0].shape}")
        print(f"      Verification: OK")
    except ImportError:
        print("      onnxruntime not installed, skipping verification")
    except Exception as e:
        print(f"      Verification failed: {e}")

    print(f"\n[Done] Model exported to {output_dir}/")
    print(f"       - model.onnx ({onnx_path.stat().st_size / 1024:.1f} KB)")
    print(f"       - config.json")


if __name__ == "__main__":
    export_model_to_onnx()
