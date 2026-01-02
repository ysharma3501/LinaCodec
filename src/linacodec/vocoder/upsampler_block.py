## Upsampler block is custom hifigan based upsampling block used to upscale vocos backbone features. Modified it to use snake function for faster training.

import warnings
from typing import Any, List, Optional, Tuple

import torch
from torch import Tensor, nn
from torch import Tensor, nn
from typing import Any, List, Optional, Tuple
from torch.nn.utils.parametrizations import weight_norm

@torch.jit.script
def snake(x, alpha):
    shape = x.shape
    x = x.reshape(shape[0], shape[1], -1)
    x = x + (alpha + 1e-9).reciprocal() * torch.sin(alpha * x).pow(2)
    x = x.reshape(shape)
    return x


class Snake1d(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, channels, 1))

    def forward(self, x):
        return snake(x, self.alpha)

def nonlinearity(x):
    return x * torch.sigmoid(x)


def Normalize(in_channels, num_groups=32):
    return torch.nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)

class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False,
                 dropout, temb_channels=512):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut
        self.snake1 = Snake1d(in_channels)
        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv1d(in_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        if temb_channels > 0:
            self.temb_proj = torch.nn.Linear(temb_channels,
                                             out_channels)
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv1d(out_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        self.snake2 = Snake1d(out_channels)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv1d(in_channels,
                                                     out_channels,
                                                     kernel_size=3,
                                                     stride=1,
                                                     padding=1)
            else:
                self.nin_shortcut = torch.nn.Conv1d(in_channels,
                                                    out_channels,
                                                    kernel_size=1,
                                                    stride=1,
                                                    padding=0)

    def forward(self, x, temb=None):
        h = x
        h = self.norm1(h)
        h = self.snake1(h)
        h = self.conv1(h)

        if temb is not None:
            h = h + self.temb_proj(nonlinearity(temb))[:, :, None]

        h = self.norm2(h)
        h = self.snake2(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x + h
class UpSamplerBlock(nn.Module):
    """Transpose Conv plus Resnet Blocks to upsample feature embedding."""
    def __init__(self, in_channels: int, upsample_factors: List[int], kernel_sizes: Optional[List[int]] = None):
        super().__init__()
        self.in_channels = in_channels
        self.upsample_factors = list(upsample_factors or [])
        self.kernel_sizes = list(kernel_sizes or [8] * len(self.upsample_factors))

        assert len(self.kernel_sizes) == len(self.upsample_factors), "kernel_sizes and upsample_factors must have the same length"

        self.upsample_layers = nn.ModuleList()
        self.resnet_blocks  = nn.ModuleList()
        self.out_proj = nn.Linear(self.in_channels // (2 ** len(self.upsample_factors)), self.in_channels, bias=True)

        for i, (k, u) in enumerate(zip(self.kernel_sizes, self.upsample_factors)):
            c_in  = self.in_channels // (2 ** i)
            c_out = self.in_channels // (2 ** (i + 1))
            self.upsample_layers.append(
                weight_norm(nn.ConvTranspose1d(c_in, c_out, kernel_size=k, stride=u, padding=(k - u) // 2))
            )
            self.resnet_blocks.append(
                ResnetBlock(in_channels=c_out, out_channels=c_out, dropout=0.0, temb_channels=0)
            )
        self.final_snake = Snake1d(self.in_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, L] -> ... -> [B, C', L']
        for up, rsblk in zip(self.upsample_layers, self.resnet_blocks):
            x = rsblk(up(x))
        x = self.out_proj(x.transpose(1, 2))

        # 2. Transpose back for Snake: [B, L, C_high] -> [B, C_high, L]
        x = x.transpose(1, 2)
        return self.final_snake(x)
      
def init_weights(m):
    if isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose1d):
        # Xavier initialization helps keep signal variance steady
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, Snake1d):
        # Crucial: Start alpha at 1.0.
        # Too high = noisy; too low = linear.
        nn.init.constant_(m.alpha, 1.0)
