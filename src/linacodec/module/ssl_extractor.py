import torch
import torch.nn as nn
import torchaudio
import torchaudio.pipelines as pipelines
from torchaudio.models.wav2vec2 import Wav2Vec2Model
from torchaudio.models.wav2vec2.components import ConvLayerBlock

from ..util import get_logger

logger = get_logger()


# Map of friendly names to torchaudio pipeline bundles
MODEL_REGISTRY = {
    "wav2vec2_base": pipelines.WAV2VEC2_BASE,
    "wav2vec2_large": pipelines.WAV2VEC2_LARGE,
    "wav2vec2_large_lv60k": pipelines.WAV2VEC2_LARGE_LV60K,
    "hubert_base": pipelines.HUBERT_BASE,
    "hubert_large": pipelines.HUBERT_LARGE,
    "hubert_xlarge": pipelines.HUBERT_XLARGE,
    "wavlm_base": pipelines.WAVLM_BASE,
    "wavlm_base_plus": pipelines.WAVLM_BASE_PLUS,
    "wavlm_large": pipelines.WAVLM_LARGE,
}


class SSLFeatureExtractor(nn.Module):
    def __init__(self, model_name: str = "wavlm_base_plus", output_layer: int | None = None, sample_rate: int = 16000):
        """
        Args:
            model_name: Name of the SSL model to use
            output_layer: Which layer's features to extract (None for last layer), 1-based indexing
            sample_rate: Sample rate of input audio
        """
        super().__init__()
        self.output_layer = output_layer if output_layer is not None else -1

        if model_name not in MODEL_REGISTRY:
            raise ValueError(f"Unknown model: {model_name}. Available models: {list(MODEL_REGISTRY.keys())}")
        bundle = MODEL_REGISTRY[model_name]
        self.model: Wav2Vec2Model = bundle.get_model()
        self.model.eval()
        self.feature_dim: int = bundle._params["encoder_embed_dim"]

        self.ssl_sample_rate = bundle.sample_rate
        # Create resampler if needed
        if sample_rate != self.ssl_sample_rate:
            logger.debug(f"Resampling from {sample_rate} to {self.ssl_sample_rate} required by {model_name}.")
            self.resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=self.ssl_sample_rate)
        else:
            self.resampler = None

    @property
    def hop_size(self) -> int:
        """Get the hop size of the model's convolutional layers."""
        hop_size = 1
        for _, stride in self.conv_config:
            hop_size *= stride
        return hop_size

    @property
    def conv_config(self) -> list[tuple[int, int]]:
        """Get the configuration of the convolutional layers in the model."""
        conv_layers = []
        for layer in self.model.feature_extractor.conv_layers:
            layer: ConvLayerBlock
            conv_layers.append((layer.kernel_size, layer.stride))
        return conv_layers

    def get_minimum_input_length(self, desired_output_length: int) -> int:
        """Calculate the minimum input length required to produce a given output length."""
        length = desired_output_length
        for kernel_size, stride in reversed(self.conv_config):
            length = (length - 1) * stride + kernel_size
        return length

    @torch.no_grad()
    def forward(
        self,
        waveform: torch.Tensor,
        lengths: torch.Tensor | None = None,
        num_layers: int | None = None,
        return_lengths: bool = False,
    ) -> list[torch.Tensor]:
        """
        Args:
            waveform: (batch_size, num_samples)
            lengths: Optional tensor of sequence lengths for each batch item (used for attention masking)

        Returns:
            features: List of feature tensors for each layer (batch_size, frame, dim)
            lengths: Sequence lengths for each batch item
        """
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        # Resample if needed
        if self.resampler is not None:
            waveform = self.resampler(waveform)

        features, feature_lengths = self.model.extract_features(
            waveform, lengths, num_layers=num_layers or self.output_layer
        )

        if return_lengths:
            return features, feature_lengths
        return features
