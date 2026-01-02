import math
from dataclasses import dataclass

import jsonargparse
import torch
import torch.nn as nn
import torch.nn.functional as F

from .module.fsq import FiniteScalarQuantizer
from .module.global_encoder import GlobalEncoder
from .module.postnet import PostNet
from .module.ssl_extractor import SSLFeatureExtractor
from .module.transformer import Transformer
from .module.distill_wavlm import wav2vec2_model
from .util import freeze_modules, get_logger

logger = get_logger()


@dataclass
class LinaCodecConfig:
    # SSL Feature settings
    local_ssl_layers: tuple[int, ...] = (6, 9)  # Indices of SSL layers for local branch
    global_ssl_layers: tuple[int, ...] = (1, 2)  # Indices of SSL layers for global branch
    normalize_ssl_features: bool = True  # Whether to normalize local SSL features before encoding

    # Down/up-sampling settings
    downsample_factor: int = 2  # Temporal downsampling factor for local features
    mel_upsample_factor: int = 4  # Conv1DTranspose upsampling factor for mel features before interpolation
    use_conv_downsample: bool = True  # Whether to use Conv1D for downsampling instead average pooling
    local_interpolation_mode: str = "linear"  # Interpolation mode for local upsampling ("linear", "nearest")
    mel_interpolation_mode: str = "linear"  # Interpolation mode for mel upsampling ("linear", "nearest")

    # Mel spectrogram settings
    sample_rate: int = 24000
    n_fft: int = 1024
    hop_length: int = 256
    n_mels: int = 100
    padding: str = "center"


@dataclass
class LinaCodecFeatures:
    content_embedding: torch.Tensor | None = None  # (seq_len, dim)
    content_token_indices: torch.Tensor | None = None  # (seq_len,)
    global_embedding: torch.Tensor | None = None  # (dim,)


class LinaCodecModel(nn.Module):
    """Model architecture and forward pass logic for Kanade tokenizer."""

    def __init__(
        self,
        config: LinaCodecConfig,
        ssl_feature_extractor: SSLFeatureExtractor,
        local_encoder: Transformer,
        local_quantizer: FiniteScalarQuantizer,
        feature_decoder: Transformer | None,
        global_encoder: GlobalEncoder,
        mel_prenet: Transformer,
        mel_decoder: Transformer,
        mel_postnet: PostNet,
    ):
        super().__init__()
        self.config = config
        self._init_ssl_extractor(config, ssl_feature_extractor)
        self._init_local_branch(config, local_encoder, local_quantizer, feature_decoder)
        self._init_global_branch(global_encoder)
        self._init_mel_decoder(config, mel_prenet, mel_decoder, mel_postnet)
        
    def load_distilled_wavlm(self, path: str):
        """Loads distilled wavlm model, 970m params --> 250m params"""
        ckpt = torch.load(path)
        wavlm_model = wav2vec2_model(**ckpt["config"])
        result = wavlm_model.load_state_dict(ckpt["state_dict"], strict=False)
        self.wavlm_model = wavlm_model.cuda()
        self.distilled_layers = [6, 8] ## can set custom, 6-8 seems best however

    def _init_ssl_extractor(self, config: LinaCodecConfig, ssl_feature_extractor: SSLFeatureExtractor):
        """Initialize and configure SSL feature extractor."""
        self.ssl_feature_extractor = ssl_feature_extractor
        freeze_modules([self.ssl_feature_extractor])
        logger.debug(
            f"SSL feature extractor initialized and frozen, feature dim: {self.ssl_feature_extractor.feature_dim}"
        )

        # Configure local SSL layers
        self.local_ssl_layers = list(config.local_ssl_layers)
        if len(self.local_ssl_layers) > 1:
            logger.debug(
                f"Using average of {len(self.local_ssl_layers)} SSL layers for local branch: {self.local_ssl_layers}"
            )
        else:
            logger.debug(f"Using single SSL layer {self.local_ssl_layers[0]} for local branch")

        if config.normalize_ssl_features:
            logger.debug("Normalizing local SSL features before encoding")

        # Configure global SSL layers
        self.global_ssl_layers = list(config.global_ssl_layers)
        if len(self.global_ssl_layers) > 1:
            logger.debug(
                f"Using average of {len(self.global_ssl_layers)} SSL layers for global branch: {self.global_ssl_layers}"
            )
        else:
            logger.debug(f"Using single SSL layer {self.global_ssl_layers[0]} for global branch")

    def _init_local_branch(
        self,
        config: LinaCodecConfig,
        local_encoder: Transformer,
        local_quantizer: FiniteScalarQuantizer,
        feature_decoder: Transformer | None,
    ):
        """Initialize local branch components (encoder, downsampling, quantizer, decoder)."""
        self.local_encoder = local_encoder
        self.local_quantizer = local_quantizer
        self.feature_decoder = feature_decoder

        # Configure downsampling
        self.downsample_factor = config.downsample_factor
        if self.downsample_factor > 1:
            logger.debug(f"Using temporal downsampling with factor {self.downsample_factor}")
            if config.use_conv_downsample:
                # Create Conv1d layers for downsampling and upsampling local embeddings
                feature_dim = local_encoder.output_dim
                self.conv_downsample = nn.Conv1d(
                    feature_dim, feature_dim, kernel_size=config.downsample_factor, stride=config.downsample_factor
                )
                self.conv_upsample = nn.ConvTranspose1d(
                    feature_dim, feature_dim, kernel_size=config.downsample_factor, stride=config.downsample_factor
                )  # won't be used unless training feature reconstruction
                logger.debug(f"Using Conv1d downsampling/upsampling with kernel size {config.downsample_factor}")
            else:
                self.conv_downsample = None
                self.conv_upsample = None
                logger.debug("Using average pooling and linear interpolation for downsampling/upsampling")
        else:
            self.conv_downsample = None
            self.conv_upsample = None

    def _init_global_branch(self, global_encoder: GlobalEncoder):
        """Initialize global branch components."""
        self.global_encoder = global_encoder

    def _init_mel_decoder(
        self, config: LinaCodecConfig, mel_prenet: Transformer, mel_decoder: Transformer, mel_postnet: PostNet
    ):
        """Initialize mel decoder components (prenet, upsampling, decoder, postnet)."""
        self.mel_prenet = mel_prenet
        self.mel_decoder = mel_decoder
        self.mel_postnet = mel_postnet

        # Configure mel upsampling
        self.mel_conv_upsample = None
        if config.mel_upsample_factor > 1:
            # Create Conv1DTranspose layer for mel upsampling
            input_dim = mel_prenet.output_dim
            self.mel_conv_upsample = nn.ConvTranspose1d(
                input_dim, input_dim, kernel_size=config.mel_upsample_factor, stride=config.mel_upsample_factor
            )
            logger.debug(f"Using Conv1DTranspose for mel upsampling with factor {config.mel_upsample_factor}")

    def _calculate_waveform_padding(self, audio_length: int, ensure_recon_length: bool = False) -> int:
        """Calculate required padding for input waveform to ensure consistent SSL feature lengths."""
        extractor = self.ssl_feature_extractor
        sample_rate = self.config.sample_rate
        # SSL may resample the input to its own sample rate, so calculate the number of samples after resampling
        num_samples_after_resampling = audio_length / sample_rate * extractor.ssl_sample_rate
        # We expect the SSL feature extractor to be consistent with its hop size
        expected_ssl_output_length = math.ceil(num_samples_after_resampling / extractor.hop_size)
        # If ensure_recon_length is True, we want to make sure the output length is exactly divisible by downsample factor
        if ensure_recon_length and (remainder := expected_ssl_output_length % self.downsample_factor) != 0:
            expected_ssl_output_length += self.downsample_factor - remainder
        # But it may require more input samples to produce that output length, so calculate the required input length
        num_samples_required_after_resampling = extractor.get_minimum_input_length(expected_ssl_output_length)
        # That number of samples is at the SSL sample rate, so convert back to our original sample rate
        num_samples_required = num_samples_required_after_resampling / extractor.ssl_sample_rate * sample_rate
        # Calculate padding needed on each side
        padding = math.ceil((num_samples_required - audio_length) / 2)
        return padding

    def _calculate_original_audio_length(self, token_length: int) -> int:
        """Calculate the original audio length based on token length."""
        extractor = self.ssl_feature_extractor
        sample_rate = self.config.sample_rate
        # Calculate the feature length before downsampling
        feature_length = token_length * self.downsample_factor
        num_samples_required_after_resampling = extractor.get_minimum_input_length(feature_length)
        num_samples_required = num_samples_required_after_resampling / extractor.ssl_sample_rate * sample_rate
        return math.ceil(num_samples_required)

    def _calculate_target_mel_length(self, audio_length: int) -> int:
        """Calculate the target mel spectrogram length based on audio length."""
        if self.config.padding == "center":
            return audio_length // self.config.hop_length + 1
        elif self.config.padding == "same":
            return audio_length // self.config.hop_length
        else:
            return (audio_length - self.config.n_fft) // self.config.hop_length + 1

    def _process_ssl_features(self, features: list[torch.Tensor], layers: list[int]) -> torch.Tensor:
        if len(layers) > 1:
            # Get features from multiple layers and average them
            selected_features = [features[i - 1] for i in layers]
            mixed_features = torch.stack(selected_features, dim=0).mean(dim=0)
        else:
            # Just take the single specified layer
            mixed_features = features[layers[0] - 1]
        return mixed_features

    def _normalize_ssl_features(self, features: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        if not self.config.normalize_ssl_features:
            return features

        # Compute mean and std across time steps for each sample and feature dimension
        mean = torch.mean(features, dim=1, keepdim=True)  # (B, 1, C)
        std = torch.std(features, dim=1, keepdim=True)  # (B, 1, C)
        return (features - mean) / (std + eps)

    def forward_ssl_features(
        self, waveform: torch.Tensor, padding: int | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass to extract SSL features. (B, T, C)
        Args:
            waveform: Input waveform tensor of shape (B, channels, samples)
            padding: Optional padding to apply on both sides of the waveform. This is useful to ensure
                     that the SSL feature extractor produces consistent output lengths.
        Returns:
            local_ssl_features: Local SSL features for local branch. (B, T, C)
            global_ssl_features: Global SSL features for global branch. (B, T, C)
        """
        # Prepare input waveform
        if waveform.dim() == 3:
            waveform = waveform.squeeze(1)

        # 1. Extract SSL features
        if padding > 0:
            waveform = F.pad(waveform, (padding, padding), mode="constant")

        with torch.no_grad():
            acoustic_wavlm_features = self.ssl_feature_extractor(waveform, num_layers=2) ## only needs 2 layers as acoustic info is present in them
            waveform = self.ssl_feature_extractor.resampler(waveform)
            distilled_wavlm_features = self.wavlm_model.extract_features(waveform, num_layers=max(self.distilled_layers))[0] ## semantic and prosody info is present in layer 4-10, 6-8 is best for quality

        local_ssl_features = self._process_ssl_features(distilled_wavlm_features, self.distilled_layers)
        local_ssl_features = self._normalize_ssl_features(local_ssl_features)

        global_ssl_features = self._process_ssl_features(acoustic_wavlm_features, self.global_ssl_layers)

        return local_ssl_features, global_ssl_features

    def forward_content(
        self, local_ssl_features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None:
        """Forward pass to extract content embeddings from the local branch.
        Args:
            local_ssl_features: Local SSL features tensor of shape (B, T, C)
        Returns:
            local_quantized: Quantized local embeddings. (B, T/factor, C)
            indices: Content token indices. (B, T/factor)
            ssl_recon: Reconstructed SSL features (if feature decoder is present). (B, T, C)
            perplexity: Quantizer perplexity (if feature decoder is present). Scalar tensor.
        """
        local_encoded = self.local_encoder(local_ssl_features)

        # Downsample temporally if needed: (B, T, C) -> (B, T/factor, C)
        if self.downsample_factor > 1:
            if self.config.use_conv_downsample:
                local_encoded = self.conv_downsample(local_encoded.transpose(1, 2)).transpose(1, 2)
            else:
                local_encoded = F.avg_pool1d(
                    local_encoded.transpose(1, 2), kernel_size=self.downsample_factor, stride=self.downsample_factor
                ).transpose(1, 2)

        # If training feature reconstruction, decode local embeddings
        ssl_recon = None
        perplexity = torch.tensor(0.0)
        if self.feature_decoder is not None:
            local_quantized, local_quantize_info = self.local_quantizer(local_encoded)
            indices = local_quantize_info["indices"]
            perplexity = torch.mean(local_quantize_info["perplexity"])

            local_latent_for_ssl = local_quantized
            # Upsample if needed
            if self.downsample_factor > 1:
                if self.config.use_conv_downsample:
                    # Use conv transpose for upsampling: (B, T/factor, C) -> (B, C, T/factor) -> conv -> (B, C, T) -> (B, T, C)
                    local_latent_for_ssl = self.conv_upsample(local_latent_for_ssl.transpose(1, 2)).transpose(1, 2)
                else:
                    # (B, T/factor, C) -> (B, T, C)
                    local_latent_for_ssl = F.interpolate(
                        local_latent_for_ssl.transpose(1, 2),
                        size=local_ssl_features.shape[1],
                        mode=self.config.local_interpolation_mode,
                    ).transpose(1, 2)

            ssl_recon = self.feature_decoder(local_latent_for_ssl)
        else:
            # If not training feature reconstruction, just get quantized local embeddings
            local_quantized, indices = self.local_quantizer.encode(local_encoded)

        return local_quantized, indices, ssl_recon, perplexity

    def forward_global(self, global_ssl_features: torch.Tensor) -> torch.Tensor:
        """Forward pass to extract global embeddings from the global branch.
        Args:
            global_ssl_features: Global SSL features tensor of shape (B, T, C)
        Returns:
            global_encoded: Global embeddings. (B, C)
        """
        global_encoded = self.global_encoder(global_ssl_features)
        return global_encoded

    def forward_mel(
        self, content_embeddings: torch.Tensor, global_embeddings: torch.Tensor, mel_length: int
    ) -> torch.Tensor:
        """Forward pass to generate mel spectrogram from content and global embeddings.
        Args:
            content_embeddings: Content embeddings tensor of shape (B, T, C)
            global_embeddings: Global embeddings tensor of shape (B, C)
            mel_length: Target mel spectrogram length (T_mel)
        Returns:
            mel_recon: Reconstructed mel spectrogram tensor of shape (B, n_mels, T_mel)
        """
        local_latent = self.mel_prenet(content_embeddings)

        # Upsample local latent to match mel spectrogram length
        # First use Conv1DTranspose if configured
        if self.mel_conv_upsample is not None:
            # (B, T/factor, C) -> (B, C, T/factor) -> conv -> (B, C, T*upsample_factor) -> (B, T*upsample_factor, C)
            local_latent = self.mel_conv_upsample(local_latent.transpose(1, 2)).transpose(1, 2)
        local_latent = F.interpolate(
            local_latent.transpose(1, 2), size=mel_length, mode=self.config.mel_interpolation_mode
        ).transpose(1, 2)  # (B, T_current, C) -> (B, T_mel, C)

        # Generate mel spectrogram, conditioned on global embeddings
        mel_recon = self.mel_decoder(local_latent, condition=global_embeddings.unsqueeze(1))
        mel_recon = mel_recon.transpose(1, 2)  # (B, n_mels, T)

        mel_recon = self.mel_postnet(mel_recon)
        return mel_recon

    # ======== Inference methods ========

    def weights_to_save(self, *, include_modules: list[str]) -> dict[str, torch.Tensor]:
        """Get model weights for saving. Excludes certain modules not needed for inference."""
        excluded_modules = [
            m
            for m in ["ssl_feature_extractor", "feature_decoder", "conv_upsample"]
            if m not in include_modules
        ]
        state_dict = {
            name: param
            for name, param in self.named_parameters()
            if not any(name.startswith(excl) for excl in excluded_modules)
        }
        return state_dict

    @classmethod
    def from_hparams(cls, config_path: str) -> "LinaCodecModel":
        """Instantiate KanadeModel from config file.
        Args:
            config_path (str): Path to model configuration file (.yaml).
        Returns:
            KanadeModel: Instantiated KanadeModel.
        """
        parser = jsonargparse.ArgumentParser(exit_on_error=False)
        parser.add_argument("--model", type=LinaCodecModel)
        cfg = parser.parse_path(config_path)
        cfg = parser.instantiate_classes(cfg)
        return cfg.model

    @classmethod
    def from_pretrained(
        cls,
        repo_id: str | None = None,
        revision: str | None = None,
        config_path: str | None = None,
        weights_path: str | None = None,
    ) -> "KanadeModel":
        """Load LinaCodec either from HuggingFace Hub or local config and weights files.
        Args:
            repo_id (str, optional): HuggingFace Hub repository ID. If provided, loads config and weights from the hub.
            revision (str, optional): Revision (branch, tag, commit) for the HuggingFace Hub repo.
            config_path (str, optional): Path to model configuration file (.yaml). Required if repo_id is not provided.
            weights_path (str, optional): Path to model weights file (.safetensors). Required if repo_id is not provided.
        Returns:
            LinaCodec: Loaded LinaCodec instance.
        """
        if repo_id is not None:
            # Load from HuggingFace Hub
            from huggingface_hub import hf_hub_download

            config_path = hf_hub_download(repo_id, "config.yaml", revision=revision)
            weights_path = hf_hub_download(repo_id, "model.safetensors", revision=revision)
        else:
            # Check local paths
            if config_path is None or weights_path is None:
                raise ValueError(
                    "Please provide either HuggingFace Hub repo_id or both config_path and weights_path for model loading."
                )

        # Load model from config
        model = cls.from_hparams(config_path)

        # Load weights
        from safetensors.torch import load_file

        state_dict = load_file(weights_path, device="cpu")
        model.load_state_dict(state_dict, strict=False)
        logger.info(f"Loaded weights from safetensors file: {weights_path}")

        return model

    @torch.inference_mode()
    def encode(self, waveform: torch.Tensor, return_content: bool = True, return_global: bool = True) -> LinaCodecFeatures:
        """Extract content and/or global features from audio using Kanade model.
        Args:
            waveform (torch.Tensor): Input audio waveform tensor (samples,). The sample rate should match model config.
            return_content (bool): Whether to extract content features.
            return_global (bool): Whether to extract global features.
        Returns:
            dict[str, torch.Tensor]: Extracted features.
        """
        audio_length = waveform.size(0)
        padding = self._calculate_waveform_padding(audio_length)
        local_ssl_features, global_ssl_features = self.forward_ssl_features(waveform.unsqueeze(0), padding=padding)

        result = LinaCodecFeatures()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
            if return_content:
                content_embedding, token_indices, _, _ = self.forward_content(local_ssl_features)
                result.content_embedding = content_embedding.squeeze(0)  # (seq_len, dim)
                result.content_token_indices = token_indices.squeeze(0)  # (seq_len,)

            if return_global:
                global_embedding = self.forward_global(global_ssl_features)
                result.global_embedding = global_embedding.squeeze(0)  # (dim,)

        return result

    def decode_token_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """Get content embeddings from content token indices. (..., seq_len) -> (..., seq_len, dim)"""
        content_embedding = self.local_quantizer.decode(indices)
        return content_embedding

    @torch.inference_mode()
    def decode(
        self,
        global_embedding: torch.Tensor,
        content_token_indices: torch.Tensor | None = None,
        content_embedding: torch.Tensor | None = None,
        target_audio_length: int | None = None,
    ) -> torch.Tensor:
        """Synthesize audio from content and global features using LinaCodec model and Vocos.
        Args:
            global_embedding (torch.Tensor): Global embedding tensor (dim,).
            content_token_indices (torch.Tensor, optional): Optional content token indices tensor (seq_len).
            content_embedding (torch.Tensor, optional): Optional content embedding tensor (seq_len, dim).
                If both content_token_indices and content_embedding are provided, content_embedding takes precedence.
            target_audio_length (int, optional): Target length of the output audio in samples.
                If None, uses the original audio length estimated from the sequence length of content tokens.
        Returns:
            torch.Tensor: Generated mel spectrogram tensor (n_mels, T).
        """
        # Obtain content embedding if not provided
        if content_embedding is None:
            if content_token_indices is None:
                raise ValueError("Either content_token_indices or content_embedding must be provided.")
            content_embedding = self.decode_token_indices(content_token_indices)

        if target_audio_length is None:
            # Estimate original audio length from content token sequence length
            seq_len = content_embedding.size(0)
            target_audio_length = self._calculate_original_audio_length(seq_len)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
            mel_length = self._calculate_target_mel_length(target_audio_length)
            content_embedding = content_embedding.unsqueeze(0)  # (1, seq_len, dim)
            global_embedding = global_embedding.unsqueeze(0)  # (1, dim)
            mel_spectrogram = self.forward_mel(content_embedding, global_embedding, mel_length=mel_length)

        return mel_spectrogram.squeeze(0)  # (n_mels, T)

    @torch.inference_mode()
    def voice_conversion(self, source_waveform: torch.Tensor, reference_waveform: torch.Tensor) -> torch.Tensor:
        """Convert voice using LinaCodec model and Vocos, keeping content from source and global characteristics from reference.
        Only supports single audio input. Just a convenient wrapper around encode and decode methods.
        Args:
            source_waveform (torch.Tensor): Source audio waveform tensor (samples,).
            reference_waveform (torch.Tensor): Reference audio waveform tensor (samples_ref,).
        Returns:
            torch.Tensor: Converted mel spectrogram tensor (n_mels, T).
        """
        # Extract source content features and reference global features
        source_features = self.encode(source_waveform, return_content=True, return_global=False)
        reference_features = self.encode(reference_waveform, return_content=False, return_global=True)

        # Synthesize mel spectrogram using source content and reference global features
        mel_spectrogram = self.decode(
            content_embedding=source_features.content_embedding,
            global_embedding=reference_features.global_embedding,
            target_audio_length=source_waveform.size(0),
        )
        return mel_spectrogram
