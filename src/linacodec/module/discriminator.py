# Adapted from:
# https://github.com/gemelo-ai/vocos/blob/main/vocos/discriminators.py
# https://github.com/gemelo-ai/vocos/blob/main/vocos/loss.py

import torch
from einops import rearrange
from torch import nn
from torch.nn.utils.parametrizations import weight_norm


def get_2d_padding(kernel_size: tuple[int, int], dilation: tuple[int, int] = (1, 1)):
    return (((kernel_size[0] - 1) * dilation[0]) // 2, ((kernel_size[1] - 1) * dilation[1]) // 2)


class SpectrogramDiscriminator(nn.Module):
    def __init__(
        self,
        frequency_bins: int,
        channels: int = 32,
        kernel_size: tuple[int, int] = (3, 3),
        dilation: list[int] = [1, 2, 4],
        bands: tuple[tuple[float, float], ...] = ((0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)),
        use_downsample: bool = True,
    ):
        super().__init__()
        self.bands = [(int(b[0] * frequency_bins), int(b[1] * frequency_bins)) for b in bands]

        self.stacks = nn.ModuleList()
        for _ in self.bands:
            stack = nn.ModuleList(
                [weight_norm(nn.Conv2d(1, channels, kernel_size, padding=get_2d_padding(kernel_size)))]
            )

            for d in dilation:
                # dilation on time axis
                pad = get_2d_padding(kernel_size, (d, 1))
                stack.append(weight_norm(nn.Conv2d(channels, channels, kernel_size, dilation=(d, 1), padding=pad)))

            stack.append(weight_norm(nn.Conv2d(channels, channels, kernel_size, padding=get_2d_padding(kernel_size))))

            self.stacks.append(stack)

        self.conv_post = weight_norm(nn.Conv2d(channels, 1, kernel_size, padding=get_2d_padding(kernel_size)))
        if use_downsample:
            self.downsample = nn.AvgPool2d(4, stride=2, padding=1, count_include_pad=False)
        else:
            self.downsample = nn.Identity()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Args:
            x (Tensor): Input spectrogram (B, C, F, T).
        Returns:
            output (Tensor): Discriminator output.
            intermediates (list[Tensor]): List of intermediate feature maps.
        """
        if x.dim() == 3:
            x = x.unsqueeze(1)
        assert x.dim() == 4, f"Expected 4D input, got {x.dim()}D"

        # Split into bands
        x = rearrange(x, "b c f t -> b c t f")
        x_bands = [x[..., b[0] : b[1]] for b in self.bands]

        x = []
        intermediates = []
        for x_band, stack in zip(x_bands, self.stacks):
            for layer in stack:
                x_band = layer(x_band)
                x_band = torch.nn.functional.leaky_relu(x_band, 0.1)
                intermediates.append(x_band)
            x.append(x_band)

        # Concatenate the outputs from all bands
        x = torch.cat(x, dim=-1)
        x = self.conv_post(x)
        x = self.downsample(x)
        return x, intermediates
