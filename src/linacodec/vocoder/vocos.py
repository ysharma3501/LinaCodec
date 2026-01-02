from __future__ import annotations

from typing import Any, Dict, Tuple, Union, Optional

import torch
import yaml
from huggingface_hub import hf_hub_download
from torch import nn
from vocos.feature_extractors import FeatureExtractor, EncodecFeatures
from vocos.heads import FourierHead
from vocos.models import Backbone
from vocos.heads import ISTFTHead
from torch.cuda.amp import autocast
import torchaudio.functional as AF

from .linkwitz import crossover_merge_linkwitz_riley
from .upsampler_block import UpSamplerBlock

def instantiate_class(args: Union[Any, Tuple[Any, ...]], init: Dict[str, Any]) -> Any:
    """Instantiates a class with the given args and init.

    Args:
        args: Positional arguments required for instantiation.
        init: Dict of the form {"class_path":...,"init_args":...}.

    Returns:
        The instantiated class object.
    """
    kwargs = init.get("init_args", {})
    if not isinstance(args, tuple):
        args = (args,)
    class_module, class_name = init["class_path"].rsplit(".", 1)
    module = __import__(class_module, fromlist=[class_name])
    args_class = getattr(module, class_name)
    return args_class(*args, **kwargs)


class Vocos(nn.Module):
    """
    The Vocos class represents a Fourier-based neural vocoder for audio synthesis.
    This class is primarily designed for inference, with support for loading from pretrained
    model checkpoints. It consists of three main components: a feature extractor,
    a backbone, and a head.
    """

    def __init__(
        self, feature_extractor: FeatureExtractor, backbone: Backbone, head: FourierHead, upsampler: UpSamplerBlock, head_48k: ISTFTHead,
    ):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.backbone = backbone
        self.head = head
        self.upsampler = upsampler
        self.head_48k = head_48k
        self.freq_range = 4000

    @classmethod
    def from_hparams(cls, config_path: str) -> Vocos:
        """
        Class method to create a new Vocos model instance from hyperparameters stored in a yaml configuration file.
        """
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        feature_extractor = instantiate_class(args=(), init=config["feature_extractor"])
        backbone = instantiate_class(args=(), init=config["backbone"])
        head = instantiate_class(args=(), init=config["head"])
        upsampler = instantiate_class(args=(), init=config["upsampler"])
        head_48k = instantiate_class(args=(), init=config["head_48k"])
        model = cls(feature_extractor=feature_extractor, backbone=backbone, head=head, upsampler=upsampler, head_48k=head_48k)
        return model

    @classmethod
    def from_pretrained(cls, repo_id: str, revision: Optional[str] = None) -> Vocos:
        """
        Class method to create a new Vocos model instance from a pre-trained model stored in the Hugging Face model hub.
        """
        config_path = hf_hub_download(repo_id=repo_id, filename="config.yaml", revision=revision)
        model_path = hf_hub_download(repo_id=repo_id, filename="pytorch_model.bin", revision=revision)
        model = cls.from_hparams(config_path)
        state_dict = torch.load(model_path, map_location="cpu")
        if isinstance(model.feature_extractor, EncodecFeatures):
            encodec_parameters = {
                "feature_extractor.encodec." + key: value
                for key, value in model.feature_extractor.encodec.state_dict().items()
            }
            state_dict.update(encodec_parameters)
        model.load_state_dict(state_dict)
        model.eval()
        return model

    @torch.inference_mode()
    def forward(self, audio_input: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """
        Method to run a copy-synthesis from audio waveform. The feature extractor first processes the audio input,
        which is then passed through the backbone and the head to reconstruct the audio output.

        Args:
            audio_input (Tensor): The input tensor representing the audio waveform of shape (B, T),
                                        where B is the batch size and L is the waveform length.


        Returns:
            Tensor: The output tensor representing the reconstructed audio waveform of shape (B, T).
        """
        features = self.feature_extractor(audio_input, **kwargs)
        audio_output = self.decode(features, **kwargs)
        return audio_output

    @torch.inference_mode()
    def decode(self, features_input: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """
        Method to decode audio waveform from already calculated features. The features input is passed through
        the backbone and the head to reconstruct the audio output.

        Args:
            features_input (Tensor): The input tensor of features of shape (B, C, L), where B is the batch size,
                                     C denotes the feature dimension, and L is the sequence length.

        Returns:
            Tensor: The output tensor representing the reconstructed audio waveform of shape (B, T).
        """

        ## uses a dual path technique(one head predicts 24khz, other predicts 48khz) and then merged using sigmoid crossover for best quality in just 20 hours data!
        features = self.backbone(features_input, **kwargs).transpose(1, 2)
        upsampled_features = self.upsampler(features).transpose(1, 2)
        pred_audio = self.head_48k(upsampled_features)

        pred_audio2 = self.head(features.transpose(1, 2))
        pred_audio2 = AF.resample(pred_audio2, 24000, 48000)
        pred_audio = pred_audio[:, :pred_audio2.shape[1]]
        with autocast(enabled=False):
            merged_audio = crossover_merge_linkwitz_riley(pred_audio.float(), pred_audio2.float(), cutoff=self.freq_range)
        return merged_audio
