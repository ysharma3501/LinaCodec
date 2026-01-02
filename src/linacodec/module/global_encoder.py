# Adapted from: https://github.com/microsoft/UniSpeech/blob/main/downstreams/speaker_verification/models/ecapa_tdnn.py

import torch
import torch.nn as nn

from .convnext import ConvNextBackbone


class AttentiveStatsPool(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, attention_channels: int = 128):
        super().__init__()

        self.attn = nn.Sequential(
            nn.Conv1d(input_channels, attention_channels, kernel_size=1),
            nn.Tanh(),
            nn.Conv1d(attention_channels, input_channels, kernel_size=1),
            nn.Softmax(dim=2),
        )
        self.proj = nn.Linear(input_channels * 2, output_channels)
        self.norm = nn.LayerNorm(output_channels)

    def forward(self, x):
        alpha = self.attn(x)

        mean = torch.sum(alpha * x, dim=2)
        residuals = torch.sum(alpha * (x**2), dim=2) - mean**2
        std = torch.sqrt(residuals.clamp(min=1e-4, max=1e4))

        x = torch.cat([mean, std], dim=1)
        return self.norm(self.proj(x))


class GlobalEncoder(nn.Module):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        dim: int,
        intermediate_dim: int,
        num_layers: int,
        skip_embed: bool = False,
        attention_channels: int = 128,
        use_attn_pool: bool = True,
    ):
        super().__init__()

        self.backbone = ConvNextBackbone(
            input_channels=input_channels,
            dim=dim,
            intermediate_dim=intermediate_dim,
            num_layers=num_layers,
            skip_embed=skip_embed,
        )
        if use_attn_pool:
            self.pooling = AttentiveStatsPool(
                input_channels=dim, output_channels=output_channels, attention_channels=attention_channels
            )
        else:
            self.pooling = nn.Sequential(
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(1),
                nn.Linear(dim, output_channels),
                nn.LayerNorm(output_channels),
            )
        self.output_channels = output_channels

    @property
    def output_dim(self):
        return self.output_channels

    def forward(self, x):
        features = self.backbone(x)
        # (B, T, C) -> (B, C, T)
        features = features.transpose(1, 2)
        return self.pooling(features)  # (B, C_out)
