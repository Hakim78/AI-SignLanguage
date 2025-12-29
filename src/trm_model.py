# src/trm_model.py
"""
Stateful Tiny Recursive Model (S-TRM) for ASL Landmark Classification.

Based on the research paper "Recursive Reasoning with Tiny Networks" (2025).
Extended with Stateful Recurrence for real-time video/sequence processing.

Core concept: A single tiny network (2 layers) that recurses on input x, output y, and latent state z.
The stateful version passes z (latent memory) between frames for temporal awareness.

Architecture parameters (optimal settings from paper):
- n = 6 (latent recursion iterations)
- T = 3 (deep recursion iterations)
- Layers = 2

Features:
- Deep Supervision: Loss calculated at each recursion step
- Stateful Recurrence: State passed between video frames
- Dropout regularization for improved generalization
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional


class TinyBlock(nn.Module):
    """
    A tiny 2-layer network block used for recursive updates.
    Takes concatenated inputs and produces an output of specified dimension.
    Includes dropout for regularization.
    """
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
    """
    Stateful Tiny Recursive Model (S-TRM) for classification.

    The model recurses through:
    1. Latent recursion: Updates z based on (x, y, z), then updates y based on (y, z)
    2. Deep recursion: Runs latent recursion T times (T-1 without gradients, 1 with)
    3. Stateful recurrence: Passes state (y, z) between video frames for temporal awareness

    Args:
        input_dim: Dimension of input features (63 for raw landmarks: 21 points * 3 coords)
        num_classes: Number of output classes (24 for ASL alphabet)
        latent_dim: Dimension of latent state z
        hidden_dim: Hidden dimension in TinyBlocks
        n_latent: Number of latent recursion iterations (default: 6)
        T_deep: Number of deep recursion iterations (default: 3)
        deep_supervision: If True, return outputs from all recursion steps
        dropout: Dropout probability for regularization (default: 0.1)
    """
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

        # Input projection: x -> embedded representation
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Initial y projection (from embedded x to class logits space)
        self.y_init = nn.Linear(hidden_dim, num_classes)

        # Initial z projection (from embedded x to latent space)
        self.z_init = nn.Linear(hidden_dim, latent_dim)

        # State gate: blends previous state with new state (for stateful recurrence)
        # Input: prev_z + new_z -> gating weights
        self.state_gate = nn.Sequential(
            nn.Linear(latent_dim * 2, latent_dim),
            nn.Sigmoid(),
        )

        # TinyBlock for updating z: (x_emb, y, z) -> z_new
        # Input: hidden_dim + num_classes + latent_dim
        self.z_update = TinyBlock(
            in_dim=hidden_dim + num_classes + latent_dim,
            out_dim=latent_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

        # TinyBlock for updating y: (y, z) -> y_new
        # Input: num_classes + latent_dim
        self.y_update = TinyBlock(
            in_dim=num_classes + latent_dim,
            out_dim=num_classes,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

    def init_state(self, batch_size: int, device: torch.device = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Initialize state (y, z) for stateful recurrence.

        Args:
            batch_size: Batch size
            device: Device to create tensors on

        Returns:
            Tuple of (y_init, z_init) both as zero tensors
        """
        if device is None:
            device = next(self.parameters()).device
        y = torch.zeros(batch_size, self.num_classes, device=device)
        z = torch.zeros(batch_size, self.latent_dim, device=device)
        return y, z

    def blend_state(
        self,
        prev_z: torch.Tensor,
        new_z: torch.Tensor,
    ) -> torch.Tensor:
        """
        Blend previous latent state with new state using learned gating.

        Args:
            prev_z: Previous latent state [B, latent_dim]
            new_z: New latent state from current frame [B, latent_dim]

        Returns:
            Blended latent state [B, latent_dim]
        """
        gate_input = torch.cat([prev_z, new_z], dim=-1)
        gate = self.state_gate(gate_input)  # [B, latent_dim], values in [0, 1]
        # gate=1 means keep new_z, gate=0 means keep prev_z
        blended = gate * new_z + (1 - gate) * prev_z
        return blended

    def latent_recursion(
        self,
        x_emb: torch.Tensor,
        y: torch.Tensor,
        z: torch.Tensor,
        n: int = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """
        Latent recursion loop.

        For i in range(n):
            z = z_update(concat(x_emb, y, z))
            y = y + y_update(concat(y, z))  # Residual connection for y

        Args:
            x_emb: Embedded input [B, hidden_dim]
            y: Current output logits [B, num_classes]
            z: Current latent state [B, latent_dim]
            n: Number of iterations (default: self.n_latent)

        Returns:
            y: Updated output logits
            z: Updated latent state
            y_history: List of y at each step (for deep supervision)
        """
        if n is None:
            n = self.n_latent

        y_history = []

        for _ in range(n):
            # Update z based on (x_emb, y, z)
            z_input = torch.cat([x_emb, y, z], dim=-1)
            z = z + self.z_update(z_input)  # Residual

            # Update y based on (y, z)
            y_input = torch.cat([y, z], dim=-1)
            y = y + self.y_update(y_input)  # Residual

            y_history.append(y)

        return y, z, y_history

    def deep_recursion(
        self,
        x_emb: torch.Tensor,
        y: torch.Tensor,
        z: torch.Tensor,
        n: int = None,
        T: int = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """
        Deep recursion loop with gradient checkpointing.

        Runs latent_recursion T times:
        - First T-1 iterations: no gradients (for efficiency)
        - Last iteration: with gradients (for training)

        Args:
            x_emb: Embedded input [B, hidden_dim]
            y: Initial output logits [B, num_classes]
            z: Initial latent state [B, latent_dim]
            n: Latent recursion iterations
            T: Deep recursion iterations

        Returns:
            y: Final output logits
            z: Final latent state
            all_y_history: All y outputs from all iterations (for deep supervision)
        """
        if n is None:
            n = self.n_latent
        if T is None:
            T = self.T_deep

        all_y_history = []

        # First T-1 iterations without gradients
        if T > 1:
            with torch.no_grad():
                for _ in range(T - 1):
                    y, z, y_hist = self.latent_recursion(x_emb, y, z, n)
                    if self.deep_supervision:
                        # Detach for supervision but don't backprop through
                        all_y_history.extend([h.detach() for h in y_hist])

        # Last iteration with gradients
        y, z, y_hist = self.latent_recursion(x_emb, y, z, n)
        all_y_history.extend(y_hist)

        return y, z, all_y_history

    def forward(
        self,
        x: torch.Tensor,
        prev_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        return_all_outputs: bool = None,
        return_state: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, List[torch.Tensor]] | Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass with optional stateful recurrence.

        Args:
            x: Input landmarks [B, input_dim] (default: 63 = 21*3)
            prev_state: Optional tuple of (y_prev, z_prev) from previous frame.
                        If provided, enables stateful recurrence for temporal awareness.
            return_all_outputs: If True, return all intermediate outputs for deep supervision
            return_state: If True, also return the final state (y, z) for next frame

        Returns:
            If return_all_outputs and return_state:
                (final_logits, list_of_all_logits, (y_state, z_state))
            If return_all_outputs:
                (final_logits, list_of_all_logits)
            If return_state:
                (final_logits, (y_state, z_state))
            Else:
                final_logits
        """
        if return_all_outputs is None:
            return_all_outputs = self.deep_supervision and self.training

        # Embed input
        x_emb = self.input_proj(x)

        # Initialize y and z from embedded input
        y = self.y_init(x_emb)
        z = self.z_init(x_emb)

        # Stateful recurrence: blend with previous state if provided
        if prev_state is not None:
            y_prev, z_prev = prev_state
            # Blend latent state with previous frame's state
            z = self.blend_state(z_prev, z)
            # For y, we use a simple weighted average (can be learned too)
            y = 0.3 * y_prev + 0.7 * y

        # Deep recursion
        y_final, z_final, all_y = self.deep_recursion(x_emb, y, z)

        # Build return value
        state = (y_final.detach(), z_final.detach())

        if return_all_outputs and return_state:
            return y_final, all_y, state
        elif return_all_outputs:
            return y_final, all_y
        elif return_state:
            return y_final, state
        else:
            return y_final

    def forward_sequence(
        self,
        x_seq: torch.Tensor,
        return_all_outputs: bool = False,
    ) -> Tuple[torch.Tensor, List[torch.Tensor], List[List[torch.Tensor]]]:
        """
        Forward pass over a sequence of frames with stateful recurrence.
        Used for training on video sequences with BPTT.

        Args:
            x_seq: Input sequence [B, Seq_Len, input_dim]
            return_all_outputs: If True, return intermediate outputs for deep supervision

        Returns:
            final_outputs: Logits for each frame [B, Seq_Len, num_classes]
            states: List of (y, z) states for each frame
            all_intermediates: List of intermediate outputs per frame (for deep supervision)
        """
        B, T, D = x_seq.shape
        device = x_seq.device

        # Initialize state
        state = self.init_state(B, device)

        final_outputs = []
        states = []
        all_intermediates = []

        for t in range(T):
            x_t = x_seq[:, t, :]  # [B, input_dim]

            if return_all_outputs:
                y_out, intermediates, state = self.forward(
                    x_t, prev_state=state, return_all_outputs=True, return_state=True
                )
                all_intermediates.append(intermediates)
            else:
                y_out, state = self.forward(
                    x_t, prev_state=state, return_all_outputs=False, return_state=True
                )

            final_outputs.append(y_out)
            states.append(state)

        # Stack outputs: [B, T, num_classes]
        final_outputs = torch.stack(final_outputs, dim=1)

        return final_outputs, states, all_intermediates


class DeepSupervisionLoss(nn.Module):
    """
    Deep Supervision Loss for TRM.

    Calculates cross-entropy loss at each recursion step with optional weighting.
    Later steps can be weighted more heavily to emphasize refined predictions.
    """
    def __init__(
        self,
        num_classes: int = 24,
        weight_decay: float = 0.9,
        label_smoothing: float = 0.1,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.weight_decay = weight_decay
        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(
        self,
        final_output: torch.Tensor,
        all_outputs: List[torch.Tensor],
        targets: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Calculate deep supervision loss.

        Args:
            final_output: Final prediction logits [B, num_classes]
            all_outputs: List of all intermediate outputs
            targets: Ground truth labels [B]

        Returns:
            total_loss: Weighted sum of all losses
            metrics: Dictionary with individual losses and accuracy
        """
        # Calculate weights: later outputs get higher weights
        n_outputs = len(all_outputs)
        weights = [self.weight_decay ** (n_outputs - 1 - i) for i in range(n_outputs)]
        weight_sum = sum(weights)
        weights = [w / weight_sum for w in weights]  # Normalize

        # Calculate loss for each output
        total_loss = torch.tensor(0.0, device=final_output.device)
        losses = []

        for i, (output, weight) in enumerate(zip(all_outputs, weights)):
            loss_i = self.ce_loss(output, targets)
            losses.append(loss_i.item())
            total_loss = total_loss + weight * loss_i

        # Final output loss (extra weight on the actual prediction)
        final_loss = self.ce_loss(final_output, targets)
        total_loss = total_loss + final_loss

        # Calculate accuracy
        with torch.no_grad():
            preds = final_output.argmax(dim=-1)
            accuracy = (preds == targets).float().mean().item()

        metrics = {
            "total_loss": total_loss.item(),
            "final_loss": final_loss.item(),
            "accuracy": accuracy,
            "intermediate_losses": losses,
        }

        return total_loss, metrics


def create_trm_model(
    input_dim: int = 63,
    num_classes: int = 24,
    latent_dim: int = 64,
    hidden_dim: int = 128,
    n_latent: int = 6,
    T_deep: int = 3,
    dropout: float = 0.1,
) -> TRM:
    """Factory function to create an S-TRM model with default settings."""
    return TRM(
        input_dim=input_dim,
        num_classes=num_classes,
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        n_latent=n_latent,
        T_deep=T_deep,
        deep_supervision=True,
        dropout=dropout,
    )


if __name__ == "__main__":
    # Quick test
    print("Testing S-TRM (Stateful TRM) model...")

    model = create_trm_model(input_dim=63, num_classes=24, dropout=0.1)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test forward pass
    x = torch.randn(8, 63)  # Batch of 8, 63 dims (21 landmarks * 3)

    # Training mode (returns all outputs)
    model.train()
    final, all_outputs = model(x)
    print(f"Training - Final output shape: {final.shape}")
    print(f"Training - Number of intermediate outputs: {len(all_outputs)}")

    # Eval mode (returns only final)
    model.eval()
    final = model(x)
    print(f"Eval - Final output shape: {final.shape}")

    # Test stateful forward (simulating video frames)
    print("\n--- Testing Stateful Recurrence ---")
    state = None
    for frame_idx in range(5):
        x_frame = torch.randn(8, 63)
        final, state = model(x_frame, prev_state=state, return_state=True)
        print(f"Frame {frame_idx}: output shape {final.shape}, state z shape {state[1].shape}")

    # Test sequence forward (for BPTT training)
    print("\n--- Testing Sequence Forward (BPTT) ---")
    x_seq = torch.randn(4, 10, 63)  # Batch=4, Seq_Len=10, Features=63
    model.train()
    outputs, states, intermediates = model.forward_sequence(x_seq, return_all_outputs=True)
    print(f"Sequence output shape: {outputs.shape}")  # [4, 10, 24]
    print(f"Number of states: {len(states)}")
    print(f"Intermediates per frame: {len(intermediates[0])}")

    # Test loss
    print("\n--- Testing Deep Supervision Loss ---")
    targets = torch.randint(0, 24, (8,))
    loss_fn = DeepSupervisionLoss(num_classes=24)

    model.train()
    final, all_outputs = model(x)
    loss, metrics = loss_fn(final, all_outputs, targets)
    print(f"Loss: {metrics['total_loss']:.4f}, Accuracy: {metrics['accuracy']:.2%}")

    print("\nS-TRM model test passed!")
