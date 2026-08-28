"""Model definitions and lightweight loading helpers for evaluation notebooks."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn


class PixelMLP(nn.Module):
    """MLP regressor used for pixel-wise PR prediction."""

    def __init__(self, in_features: int = 12, hidden_sizes: tuple[int, ...] = (128, 64, 32)):
        super().__init__()
        layers: list[nn.Module] = []
        previous_size = in_features
        for hidden_size in hidden_sizes:
            layers.extend(
                [
                    nn.Linear(previous_size, hidden_size),
                    nn.BatchNorm1d(hidden_size),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.30),
                ]
            )
            previous_size = hidden_size
        layers.append(nn.Linear(previous_size, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, inputs):
        return self.net(inputs)


def load_pixel_mlp_checkpoint(
    checkpoint_path: Path | str,
    map_location: str | torch.device | None = None,
    in_features: int = 12,
    hidden_sizes: tuple[int, ...] = (128, 64, 32),
) -> PixelMLP:
    """Load a ``PixelMLP`` model from a checkpoint path."""
    model = PixelMLP(in_features=in_features, hidden_sizes=hidden_sizes)
    state = torch.load(Path(checkpoint_path), map_location=map_location, weights_only=False)

    # Support checkpoints saved as raw state_dict or wrapped dict entries.
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    model.load_state_dict(state)
    model.eval()
    return model
