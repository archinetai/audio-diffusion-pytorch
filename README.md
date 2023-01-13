<img src="./LOGO.png"></img>

A fully featured audio diffusion library, for PyTorch. Includes models for unconditional audio generation, text-conditional audio generation, diffusion autoencoding, upsampling, and vocoding. The provided models work on waveforms, however, the U-Net (built using [`a-unet`](https://github.com/archinetai/a-unet)), `DiffusionModel`, diffusion method, and diffusion samplers are both generic to any dimension and highly customizable.

## Install

```bash
pip install -U git+https://github.com/archinetai/audio-diffusion-pytorch.git@nightly
```

[![PyPI - Python Version](https://img.shields.io/pypi/v/audio-diffusion-pytorch?style=flat&colorA=black&colorB=black)](https://pypi.org/project/audio-diffusion-pytorch/)
[![Downloads](https://static.pepy.tech/personalized-badge/audio-diffusion-pytorch?period=total&units=international_system&left_color=black&right_color=black&left_text=Downloads)](https://pepy.tech/project/audio-diffusion-pytorch)


## Usage

### Unconditional Generation
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
sample = model.sample(noise, num_steps=10) # Suggested num_steps 10-50
```

### Text-Conditional Generation
```py
from audio_diffusion_pytorch.models import DiffusionModel, UNetV0, VDiffusion, VSampler

model = DiffusionModel(
    # ... same as unconditional model
    use_text_conditioning=True, # U-Net: enables text conditioning (default T5-base)
    use_embedding_cfg=True # U-Net: enables classifier free guidance
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
    num_steps=2 # Higher for better quality, suggested num_steps: 10-50
)
```

### Upsampling
```py
from audio_diffusion_pytorch.models import DiffusionUpsampler, UNetV0, VDiffusion, VSampler

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

### Vocoding
```py
from audio_diffusion_pytorch.models import DiffusionVocoder, UNetV0, VDiffusion, VSampler

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
