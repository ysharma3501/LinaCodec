# Adapted from: https://github.com/ming024/FastSpeech2

import torch
import torch.nn as nn


def get_padding(kernel_size: int, dilation: int = 1):
    return ((kernel_size - 1) * dilation) // 2


class Norm(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        # (batch_size, channels, sequence_length)
        x = x.transpose(1, 2)
        x = self.norm(x)
        return x.transpose(1, 2)


class PostNet(nn.Module):
    def __init__(
        self,
        input_channels: int = 100,
        channels: int = 512,
        kernel_size: int = 5,
        num_layers: int = 5,
        dropout: float = 0.5,
        use_layer_norm: bool = False,
    ):
        super().__init__()

        padding = get_padding(kernel_size)
        self.convolutions = nn.ModuleList()

        self.convolutions.append(
            nn.Sequential(
                nn.Conv1d(input_channels, channels, kernel_size=kernel_size, padding=padding),
                Norm(channels) if use_layer_norm else nn.BatchNorm1d(channels),
            )
        )
        for i in range(1, num_layers - 1):
            self.convolutions.append(
                nn.Sequential(
                    nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding),
                    Norm(channels) if use_layer_norm else nn.BatchNorm1d(channels),
                )
            )
        self.convolutions.append(
            nn.Sequential(
                nn.Conv1d(channels, input_channels, kernel_size=kernel_size, padding=padding),
                Norm(input_channels) if use_layer_norm else nn.BatchNorm1d(input_channels),
            )
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x

        for i in range(len(self.convolutions) - 1):
            x = self.convolutions[i](x)
            x = torch.tanh(x)
            x = self.dropout(x)

        x = self.convolutions[-1](x)
        x = self.dropout(x)

        return x + residual
