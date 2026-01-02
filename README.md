## Linacodec: Highly compressive audio tokenizer for speech models.
<p align="center">
  <a href="https://huggingface.co/YatharthS/LinaCodec">
    <img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-FFD21E" alt="Hugging Face Model">
  </a>
</p>

Linacodec is an audio tokenizer that compresses audio into just 12.5 tokens per second (171 bps) and decodes to 48khz audio!

### Key benefits
* Compression: 12.5 tokens/sec (60x more compressed than DAC).
* Audio Quality: 48khz output (much clearer then 16khz/24khz which is the standard).
* Encoder Speed: 200x realtime.
* Decoder Speed: 400x realtime(even faster with batching)
* Many Tasks: Indirectly even supports voice conversion, audio super-resolution, and audio denoising!

### Why is this even useful?
Audio tokenizers directly contribute to speed, quality, and capability of TTS/ASR models. LinaCodec massively improves upon previous codecs in these areas.
* Inference Speed: Enables TTS models to run 800x realtime, 8x faster than [MiraTTS](https://github.com/ysharma3501)!
* Fast training: High-quality TTS models can be trained in less then 1 day.
* Versatile: Works for both Text-to-Speech and Speech-to-Text unlike most other codecs.

### Comparisons
| Model | Total Tokens/Sec | Sample Rate |
| :--- | :--- | :--- |
| Linacodec | 12.5 | 48khz |
| DAC | 774 | 44.1khz |
| EnCodec | 300 | 24khz |
| Xcodec2 | 50 | 16khz |
| Mimi | 200 | 24khz |

### Usage

Simple 1 line installation:
```
pip install git+https://github.com/ysharma3501/LinaCodec.git
```

Reconstruction
```python
from IPython.display import Audio
from linacodec.codec import LinaCodec

## load model
lina_tokenizer = LinaCodec() ## will download YatharthS/LinaCodec from huggingface

## get speech tokens and global embedding
speech_tokens, global_embedding = lina_tokenizer.encode("your_audio_path.wav")

## decode them into 48khz audio
audio = lina_tokenizer.decode(speech_tokens, global_embedding)

## display audio
display(Audio(audio.cpu(), rate=48000))
```

Voice conversion
```python
## Assuming you have loaded model
source_wav = "source_wav.wav" ## the content you want
reference_wav = "reference_wav.wav" ## the timbre(style) you want

## convert voice
audio = lina_tokenizer.convert_voice(source_wav, reference_wav)

## display audio
display(Audio(audio.cpu(), rate=48000))
```

Audio super resolution
```python
## get speech tokens and global embedding from 24khz wav
speech_tokens, global_embedding = lina_tokenizer.encode("your_audio_path.wav")

## decode them into 48khz audio(upsamples from 24khz-->48khz)
audio = lina_tokenizer.decode(speech_tokens, global_embedding)

## display audio
display(Audio(audio.cpu(), rate=48000))
```


### Notes
This is heavily based of [kanade-tokenizer](https://github.com/frothywater/kanade-tokenizer) so massive thanks to them! 

The key novel parts I added are:
1. Dual-Path Vocos Decoder: Enables high-quality 48kHz reconstruction from original 24khz vocos using only 30 hours of training data (compared to the typical hundreds of hours).
2. Distilled WavLM Base+: Increased encoder speed while being similar quality.
3. Snake based upsampling: Used custom upsampling block to upscale features based off snake activation from [BigVGAN](https://github.com/NVIDIA/BigVGAN).

## Next steps
- [x] Release code and model
- [ ] Release article on how kanade and Lina work so well at rates of 12.5 t/s compared to others.
- [ ] Possible paper on how these techniques can easily work on any codec.

Stars and Likes would be appreciated if found helpful, thank you.

Model link: https://huggingface.co/YatharthS/LinaCodec
Email: yatharthsharma3501@gmail.com
