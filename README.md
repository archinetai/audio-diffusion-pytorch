<img src="./LOGO.png"></img>

A fully featured audio diffusion library, for PyTorch. Includes models for unconditional audio generation, text-conditional audio generation, diffusion autoencoding, upsampling, and vocoding. The provided models are waveform-based, however, the U-Net (built using [`a-unet`](https://github.com/archinetai/a-unet)), `DiffusionModel`, diffusion method, and diffusion samplers are both generic to any dimension and highly customizable to work on other formats. **Notes: (1) no pre-trained models are provided here, (2) the configs shown are indicative and untested, see [Moûsai](https://arxiv.org/abs/2301.11757) for the configs used in the paper.**


## Install

```bash
pip install audio-diffusion-pytorch
```

[![PyPI - Python Version](https://img.shields.io/pypi/v/audio-diffusion-pytorch?style=flat&colorA=black&colorB=black)](https://pypi.org/project/audio-diffusion-pytorch/)
[![Downloads](https://static.pepy.tech/personalized-badge/audio-diffusion-pytorch?period=total&units=international_system&left_color=black&right_color=black&left_text=Downloads)](https://pepy.tech/project/audio-diffusion-pytorch)


## Usage

### Unconditional Generator

```py
from audio_diffusion_pytorch import DiffusionModel, UNetV0, VDiffusion, VSampler

model = DiffusionModel(
    net_t=UNetV0, # The model type used for diffusion (U-Net V0 in this case)
    in_channels=2, # U-Net: number of input/output (audio) channels
    channels=[8, 32, 64, 128, 256, 512, 512, 1024, 1024], # U-Net: channels at each layer
    factors=[1, 4, 4, 4, 2, 2, 2, 2, 2], # U-Net: downsampling and upsampling factors at each layer
    items=[1, 2, 2, 2, 2, 2, 2, 4, 4], # U-Net: number of repeating items at each layer
    attentions=[0, 0, 0, 0, 0, 1, 1, 1, 1], # U-Net: attention enabled/disabled at each layer
    attention_heads=8, # U-Net: number of attention heads per attention item
    attention_features=64, # U-Net: number of attention features per attention item
    diffusion_t=VDiffusion, # The diffusion method used
    sampler_t=VSampler, # The diffusion sampler used
)

# Train model with audio waveforms
audio = torch.randn(1, 2, 2**18) # [batch_size, in_channels, length]
loss = model(audio)
loss.backward()

# Turn noise into new audio sample with diffusion
noise = torch.randn(1, 2, 2**18) # [batch_size, in_channels, length]
sample = model.sample(noise, num_steps=10) # Suggested num_steps 10-100
```

### Text-Conditional Generator
A text-to-audio diffusion model that conditions the generation with `t5-base` text embeddings, requires `pip install transformers`.
```py
from audio_diffusion_pytorch import DiffusionModel, UNetV0, VDiffusion, VSampler

model = DiffusionModel(
    # ... same as unconditional model
    use_text_conditioning=True, # U-Net: enables text conditioning (default T5-base)
    use_embedding_cfg=True, # U-Net: enables classifier free guidance
    embedding_max_length=64, # U-Net: text embedding maximum length (default for T5-base)
    embedding_features=768, # U-Net: text mbedding features (default for T5-base)
    cross_attentions=[0, 0, 0, 1, 1, 1, 1, 1, 1], # U-Net: cross-attention enabled/disabled at each layer
)

# Train model with audio waveforms
audio_wave = torch.randn(1, 2, 2**18) # [batch, in_channels, length]
loss = model(
    audio_wave,
    text=['The audio description'], # Text conditioning, one element per batch
    embedding_mask_proba=0.1 # Probability of masking text with learned embedding (Classifier-Free Guidance Mask)
)
loss.backward()

# Turn noise into new audio sample with diffusion
noise = torch.randn(1, 2, 2**18)
sample = model.sample(
    noise,
    text=['The audio description'],
    embedding_scale=5.0, # Higher for more text importance, suggested range: 1-15 (Classifier-Free Guidance Scale)
    num_steps=2 # Higher for better quality, suggested num_steps: 10-100
)
```

### Diffusion Upsampler
Upsample audio from a lower sample rate to higher sample rate using diffusion, e.g. 3kHz to 48kHz.
```py
from audio_diffusion_pytorch import DiffusionUpsampler, UNetV0, VDiffusion, VSampler

upsampler = DiffusionUpsampler(
    net_t=UNetV0, # The model type used for diffusion
    upsample_factor=16, # The upsample factor (e.g. 16 can be used for 3kHz to 48kHz)
    in_channels=2, # U-Net: number of input/output (audio) channels
    channels=[8, 32, 64, 128, 256, 512, 512, 1024, 1024], # U-Net: channels at each layer
    factors=[1, 4, 4, 4, 2, 2, 2, 2, 2], # U-Net: downsampling and upsampling factors at each layer
    items=[1, 2, 2, 2, 2, 2, 2, 4, 4], # U-Net: number of repeating items at each layer
    diffusion_t=VDiffusion, # The diffusion method used
    sampler_t=VSampler, # The diffusion sampler used
)

# Train model with high sample rate audio waveforms
audio = torch.randn(1, 2, 2**18) # [batch, in_channels, length]
loss = upsampler(audio)
loss.backward()

# Turn low sample rate audio into high sample rate
downsampled_audio = torch.randn(1, 2, 2**14) # [batch, in_channels, length]
sample = upsampler.sample(downsampled_audio, num_steps=10) # Output has shape: [1, 2, 2**18]
```

### Diffusion Vocoder
Convert a mel-spectrogram to wavefrom using diffusion.
```py
from audio_diffusion_pytorch import DiffusionVocoder, UNetV0, VDiffusion, VSampler

vocoder = DiffusionVocoder(
    mel_n_fft=1024, # Mel-spectrogram n_fft
    mel_channels=80, # Mel-spectrogram channels
    mel_sample_rate=48000, # Mel-spectrogram sample rate
    mel_normalize_log=True, # Mel-spectrogram log normalization (alternative is mel_normalize=True for [-1,1] power normalization)
    net_t=UNetV0, # The model type used for diffusion vocoding
    channels=[8, 32, 64, 128, 256, 512, 512, 1024, 1024], # U-Net: channels at each layer
    factors=[1, 4, 4, 4, 2, 2, 2, 2, 2], # U-Net: downsampling and upsampling factors at each layer
    items=[1, 2, 2, 2, 2, 2, 2, 4, 4], # U-Net: number of repeating items at each layer
    diffusion_t=VDiffusion, # The diffusion method used
    sampler_t=VSampler, # The diffusion sampler used
)

# Train model on waveforms (automatically converted to mel internally)
audio = torch.randn(1, 2, 2**18) # [batch, in_channels, length]
loss = vocoder(audio)
loss.backward()

# Turn mel spectrogram into waveform
mel_spectrogram = torch.randn(1, 2, 80, 1024) # [batch, in_channels, mel_channels, mel_length]
sample = vocoder.sample(mel_spectrogram, num_steps=10) # Output has shape: [1, 2, 2**18]
```

### Diffusion Autoencoder
Autoencode audio into a compressed latent using diffusion. Any encoder can be provided as long as it subclasses the `EncoderBase` class or contains an `out_channels` and `downsample_factor` field.
```py
from audio_diffusion_pytorch import DiffusionAE, UNetV0, VDiffusion, VSampler
from audio_encoders_pytorch import MelE1d, TanhBottleneck

autoencoder = DiffusionAE(
    encoder=MelE1d( # The encoder used, in this case a mel-spectrogram encoder
        in_channels=2,
        channels=512,
        multipliers=[1, 1],
        factors=[2],
        num_blocks=[12],
        out_channels=32,
        mel_channels=80,
        mel_sample_rate=48000,
        mel_normalize_log=True,
        bottleneck=TanhBottleneck(),
    ),
    inject_depth=6,
    net_t=UNetV0, # The model type used for diffusion upsampling
    in_channels=2, # U-Net: number of input/output (audio) channels
    channels=[8, 32, 64, 128, 256, 512, 512, 1024, 1024], # U-Net: channels at each layer
    factors=[1, 4, 4, 4, 2, 2, 2, 2, 2], # U-Net: downsampling and upsampling factors at each layer
    items=[1, 2, 2, 2, 2, 2, 2, 4, 4], # U-Net: number of repeating items at each layer
    diffusion_t=VDiffusion, # The diffusion method used
    sampler_t=VSampler, # The diffusion sampler used
)

# Train autoencoder with audio samples
audio = torch.randn(1, 2, 2**18) # [batch, in_channels, length]
loss = autoencoder(audio)
loss.backward()

# Encode/decode audio
audio = torch.randn(1, 2, 2**18) # [batch, in_channels, length]
latent = autoencoder.encode(audio) # Encode
sample = autoencoder.decode(latent, num_steps=10) # Decode by sampling diffusion model conditioning on latent
```

## Other

### Inpainting
```py
from audio_diffusion_pytorch import UNetV0, VInpainter

# The diffusion UNetV0 (this is an example, the net must be trained to work)
net = UNetV0(
    dim=1,
    in_channels=2, # U-Net: number of input/output (audio) channels
    channels=[8, 32, 64, 128, 256, 512, 512, 1024, 1024], # U-Net: channels at each layer
    factors=[1, 4, 4, 4, 2, 2, 2, 2, 2], # U-Net: downsampling and upsampling factors at each layer
    items=[1, 2, 2, 2, 2, 2, 2, 4, 4], # U-Net: number of repeating items at each layer
    attentions=[0, 0, 0, 0, 0, 1, 1, 1, 1], # U-Net: attention enabled/disabled at each layer
    attention_heads=8, # U-Net: number of attention heads per attention block
    attention_features=64, # U-Net: number of attention features per attention block,
)

# Instantiate inpainter with trained net
inpainter = VInpainter(net=net)

# Inpaint source
y = inpainter(
    source=torch.randn(1, 2, 2**18), # Start source
    mask=torch.randint(0, 2, (1, 2, 2 ** 18), dtype=torch.bool),  # Set to `True` the parts you want to keep
    num_steps=10, # Number of inpainting steps
    num_resamples=2, # Number of resampling steps
    show_progress=True,
) # [1, 2, 2 ** 18]
```

## Appreciation

* [StabilityAI](https://stability.ai/) for the compute, [Zach Evans](https://github.com/zqevans) and everyone else from [HarmonAI](https://www.harmonai.org/) for the interesting research discussions.
* [ETH Zurich](https://inf.ethz.ch/) for the resources, [Zhijing Jin](https://zhijing-jin.com/), [Bernhard Schoelkopf](https://is.mpg.de/~bs), and [Mrinmaya Sachan](http://www.mrinmaya.io/) for supervising this Thesis.
* [Phil Wang](https://github.com/lucidrains) for the beautiful open source contributions on [diffusion](https://github.com/lucidrains/denoising-diffusion-pytorch) and [Imagen](https://github.com/lucidrains/imagen-pytorch).
* [Katherine Crowson](https://github.com/crowsonkb) for the experiments with [k-diffusion](https://github.com/crowsonkb/k-diffusion) and the insane collection of samplers.

## Citations

DDPM Diffusion
```bibtex
@misc{2006.11239,
Author = {Jonathan Ho and Ajay Jain and Pieter Abbeel},
Title = {Denoising Diffusion Probabilistic Models},
Year = {2020},
Eprint = {arXiv:2006.11239},
}
```

DDIM (V-Sampler)
```bibtex
@misc{2010.02502,
Author = {Jiaming Song and Chenlin Meng and Stefano Ermon},
Title = {Denoising Diffusion Implicit Models},
Year = {2020},
Eprint = {arXiv:2010.02502},
}
```

V-Diffusion
```bibtex
@misc{2202.00512,
Author = {Tim Salimans and Jonathan Ho},
Title = {Progressive Distillation for Fast Sampling of Diffusion Models},
Year = {2022},
Eprint = {arXiv:2202.00512},
}
```

Imagen (T5 Text Conditioning)
```bibtex
@misc{2205.11487,
Author = {Chitwan Saharia and William Chan and Saurabh Saxena and Lala Li and Jay Whang and Emily Denton and Seyed Kamyar Seyed Ghasemipour and Burcu Karagol Ayan and S. Sara Mahdavi and Rapha Gontijo Lopes and Tim Salimans and Jonathan Ho and David J Fleet and Mohammad Norouzi},
Title = {Photorealistic Text-to-Image Diffusion Models with Deep Language Understanding},
Year = {2022},
Eprint = {arXiv:2205.11487},
}
```
