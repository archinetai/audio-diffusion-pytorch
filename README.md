<img src="./LOGO.png"></img>

Nightly branch.

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

# Train model with audio
audio_wave = torch.randn(1, 2, 2**18) # [batch, in_channels, length]
loss = model(
    audio_wave,
    text=['The audio description'], # Text conditioning, one element per batch
    embedding_mask_proba=0.1 # Probability of masking text with learned embedding (Classifier-Free Guidance Mask)
)
loss.backward()

noise = torch.randn(1, 2, 2**18)
sample = model.sample(
    noise,
    text=['The audio description'],
    embedding_scale=5.0, # Higher for more text importance, suggested range: 1-15 (Classifier-Free Guidance Scale)
    num_steps=2 # Higher for better quality, suggested num_steps: 10-50
)
```
