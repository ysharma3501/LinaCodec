# Adapted from: https://github.com/facebookresearch/DiT


import torch
from torch import nn


class AdaLNZero(nn.Module):
    """
    Adaptive Layer Normalization Zero (AdaLNZero) module.

    Combines LayerNorm with adaptive conditioning to produce shift, scale, and gate values.
    The gate is used to scale features before residual connection.

    Args:
        dim: Feature dimension
        condition_dim: Conditioning dimension
        eps: LayerNorm epsilon
        return_gate: If True, returns gate value for scaling.
    """

    def __init__(
        self,
        dim: int,
        condition_dim: int,
        eps: float = 1e-5,
        return_gate: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.condition_dim = condition_dim
        self.return_gate = return_gate

        # LayerNorm without learnable parameters
        self.norm = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)

        # Conditioning network: condition -> shift, scale, gate
        output_dim = 3 * dim if return_gate else 2 * dim
        self.condition_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(condition_dim, output_dim),
        )

        # Initialize to zero for stable training
        nn.init.zeros_(self.condition_proj[1].weight)
        nn.init.zeros_(self.condition_proj[1].bias)

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor] | None:
        """
        Args:
            x: Input tensor of shape (B, L, dim)
            condition: Conditioning tensor of shape (B, L, condition_dim) or (B, 1, condition_dim)

        Returns:
            modulated_x: Normalized and modulated features
            gate: Gate values for scaling (None if return_gate=False)
        """
        x_norm = self.norm(x)
        condition_params = self.condition_proj(condition)

        if self.return_gate:
            shift, scale, gate = condition_params.chunk(3, dim=-1)
        else:
            shift, scale = condition_params.chunk(2, dim=-1)
            gate = None

        modulated_x = x_norm * (1 + scale) + shift
        return modulated_x, gate
