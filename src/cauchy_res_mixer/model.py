from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class CauchyActivation(nn.Module):
    """Trainable Cauchy activation from XNet-style formulation."""

    def __init__(
        self, lambda1: float = 1.0, lambda2: float = 0.0, d: float = 1.0
    ) -> None:
        super().__init__()
        self.lambda1 = nn.Parameter(torch.tensor(float(lambda1)))
        self.lambda2 = nn.Parameter(torch.tensor(float(lambda2)))
        self.raw_d = nn.Parameter(torch.tensor(float(d)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d = F.softplus(self.raw_d) + 1e-6
        denom = x.pow(2) + d.pow(2)
        return self.lambda1 * x / denom + self.lambda2 / denom


class CauchyResidualMLP(nn.Module):
    """Minimal MLP with optional Cauchy distance-based residual mixing."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_classes: int,
        num_layers: int = 4,
        activation_mode: str = "cauchy",
        residual_mode: str = "cauchy",
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        if residual_mode not in {"standard", "cauchy"}:
            raise ValueError("residual_mode must be one of: standard, cauchy")

        self.residual_mode = residual_mode
        self.num_layers = num_layers

        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(hidden_dim),
                    nn.Linear(hidden_dim, hidden_dim),
                    CauchyActivation() if activation_mode == "cauchy" else nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                )
                for _ in range(num_layers)
            ]
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, num_classes)
        )

        # Per-layer parameters used only when residual_mode == "cauchy".
        self.mix_raw_lambda = nn.Parameter(torch.zeros(num_layers + 1))
        self.mix_raw_d = nn.Parameter(torch.ones(num_layers + 1))

    def _cauchy_mix(
        self, history: list[torch.Tensor], target_layer: int
    ) -> torch.Tensor:
        # target_layer is 1-based index of the current layer in the stack.
        lam = F.softplus(self.mix_raw_lambda[target_layer]) + 1e-6
        d = F.softplus(self.mix_raw_d[target_layer]) + 1e-6
        count = len(history)

        distances = torch.arange(
            target_layer, target_layer - count, -1, device=history[0].device
        )
        weights = lam / (distances.float().pow(2) + d.pow(2))
        weights = weights / (weights.sum() + 1e-8)

        stacked = torch.stack(history, dim=0)
        return (weights[:, None, None] * stacked).sum(dim=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        history: list[torch.Tensor] = [h]

        for layer_idx, layer in enumerate(self.layers, start=1):
            transformed = layer(h)
            if self.residual_mode == "standard":
                h = h + transformed
            else:
                h = transformed + self._cauchy_mix(history, layer_idx)
            history.append(h)

        return self.head(h)
