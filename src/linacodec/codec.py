import torch
from linacodec.vocoder.vocos import Vocos
from huggingface_hub import snapshot_download
from .model import LinaCodecModel
from .util import load_audio, load_vocoder, vocode

class LinaCodec:
    def __init__(self, model_path=None):

        ## download from hf
        if model_path is None:
            model_path = snapshot_download("YatharthS/LinaCodec")

        ## loads linacodec model
        model = LinaCodecModel.from_pretrained(config_path=f"{model_path}/config.yaml", weights_path=f'{model_path}/model.safetensors').eval().cuda()

        ## loads distilled wavlm model, 97m params --> 25m + 18m params
        model.load_distilled_wavlm(f"{model_path}/wavlm_encoder.pth")
        model.wavlm_model.cuda()
        model.distilled_layers = [6, 9]

        ## loads vocoder, based of custom vocos and hifigan model with snake
        vocos = Vocos.from_hparams(f'{model_path}/vocoder/config.yaml').cuda()
        vocos.load_state_dict(torch.load(f'{model_path}/vocoder/pytorch_model.bin'))

        self.model = model
        self.vocos = vocos

    @torch.no_grad()
    def encode(self, audio_path):
        """encodes audio into discrete content tokens at a rate of 12.5 t/s or 25 t/s and 128 dim global embedding, single codebook"""
        ## load audio and extract features
        audio = load_audio(audio_path, sample_rate=self.model.config.sample_rate).cuda()
        features = self.model.encode(audio)
        return features.content_token_indices, features.global_embedding

    @torch.no_grad()
    @torch.autocast(device_type='cuda', dtype=torch.float16)
    def decode(self, content_tokens, global_embedding):
        """decodes tokens and embedding into 48khz waveform"""
        ## decode tokens and embedding to mel spectrogram
        mel_spectrogram = self.model.decode(content_token_indices=content_tokens, global_embedding=global_embedding)

        ## decode mel spectrogram into 48khz audio using custom vocos model
        waveform = vocode(self.vocos, mel_spectrogram.unsqueeze(0))
        return waveform
        
    def convert_voice(self, source_file, reference_file):
        """converts voice timbre, will keep content of source file but timbre of reference file"""

        ## get tokens and embedding
        speech_tokens, global_embedding = self.encode(source_file)
        ref_speech_tokens, ref_global_embedding = self.encode(reference_file)

        ## decode to audio
        audio = self.decode(speech_tokens, ref_global_embedding)
        return audio

