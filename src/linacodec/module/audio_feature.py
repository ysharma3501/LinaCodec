# Adapted from:
# Vocos: https://github.com/gemelo-ai/vocos/blob/main/vocos/feature_extractors.py

import torch
import torchaudio
from torch import nn


def safe_log(x: torch.Tensor, clip_val: float = 1e-7) -> torch.Tensor:
    return torch.log(torch.clip(x, min=clip_val))


class MelSpectrogramFeature(nn.Module):
    def __init__(
        self,
        sample_rate: int = 24000,
        n_fft: int = 1024,
        hop_length: int = 256,
        n_mels: int = 100,
        padding: str = "center",
    ):
        super().__init__()

        # Vocos style: center padding, HTK mel scale, without normalization
        if padding not in ["center", "same"]:
            raise ValueError("Padding must be 'center' or 'same'.")

        self.padding = padding
        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            center=padding == "center",
            power=1,
        )

    def forward(self, audio):
        """
        Returns:
            mel_specgram (Tensor): Mel spectrogram of the input audio. (B, C, L)
        """
        if self.padding == "same":
            pad = self.mel_spec.win_length - self.mel_spec.hop_length
            audio = torch.nn.functional.pad(audio, (pad // 2, pad // 2), mode="reflect")

        specgram = self.mel_spec.spectrogram(audio)
        mel_specgram = self.mel_spec.mel_scale(specgram)
        mel_specgram = safe_log(mel_specgram)
        return mel_specgram
