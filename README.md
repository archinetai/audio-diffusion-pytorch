
# Audio Diffusion - PyTorch

Unconditional audio generation using diffusion models, in PyTorch. 

## Install

```bash
pip install audio-diffusion-pytorch 
```

[![PyPI - Python Version](https://img.shields.io/pypi/v/audio-diffusion-pytorch?style=flat&colorA=0f0f0f&colorB=0f0f0f)](https://pypi.org/project/audio-diffusion-pytorch/) 

## Usage

```py
from audio_diffusion_pytorch import UNet1d, Diffusion, DiffusionSampler

# Construct denoising function 
unet = UNet1d(
    in_channels=1,
    channels=128,
    multipliers=(1, 2, 4, 4, 4, 4, 4),
    factors=(4, 4, 4, 4, 2, 2),
    attentions=(False, False, False, False, True, True),
    attention_heads=8,
    attention_features=64,
    attention_multiplier=2,
    dilations=[[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]],
    resnet_groups=8,
    kernel_multiplier_downsample=2,
    kernel_sizes_init=[1, 3, 7],
    use_nearest_upsample=False,
    use_skip_scale=True,
    use_attention_bottleneck=True,  
)
x = torch.randn(3, 1, 2 ** 15) 
t = torch.tensor([40, 10, 20])
y = unet(x, t) # [2, 1, 32768], 2 samples of ~1.5 seconds of generated audio at 22kHz 

# Build diffusion to train denoise function
diffusion = Diffusion(
    denoise_fn=unet,
    num_timesteps=50,
    loss_fn='l1', 
    loss_weight_gamma=0.5,
    loss_weight_k=1
)
x = torch.randn(3, 1, 2 ** 15)
loss = diffusion(x)
loss.backwards() # Do this many times 

# Sample from diffusion model by converting normal tensor to audio 
sampler = DiffusionSampler(diffusion)
y = sampler(x = torch.randn(1,1,2 ** 16)) # [1, 1, 32768] 
```

## Experiments 

### Alpha 
[Report on wandb](https://wandb.ai/schneider/audio/reports/Audio-Diffusion-UNet-Alpha---VmlldzoyMjk3MzIz)



## Citations 

```bibtex
```
