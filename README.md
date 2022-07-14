
# Audio Diffusion - PyTorch

Unconditional audio generation using diffusion models, in PyTorch. The goal of this repository is to explore different architectures and diffusion models to generate audio (speech and music) directly from/to the waveform.
Progress will be documented in the [experiments](#experiments) section.

## Install

```bash
pip install audio-diffusion-pytorch
```

[![PyPI - Python Version](https://img.shields.io/pypi/v/audio-diffusion-pytorch?style=flat&colorA=0f0f0f&colorB=0f0f0f)](https://pypi.org/project/audio-diffusion-pytorch/)

## Usage

```py
from audio_diffusion_pytorch import UNet1d
from audio_diffusion_pytorch.diffusion.ddpm import Diffusion, DiffusionSampler

# Construct denoising function
unet = UNet1d(
    in_channels=1,
    channels=128,
    multipliers=[1, 2, 4, 4, 4, 4, 4],
    factors=[4, 4, 4, 4, 2, 2],
    attentions=[False, False, False, False, True, True],
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
y = unet(x, t) # [3, 1, 32768], 3 audio tracks of ~1.6s sampled at 20050 Hz


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
y = sampler(x = torch.randn(1, 1, 2 ** 15)) # [1, 1, 32768]
```

## Experiments

### Alpha
[Report on wandb](https://wandb.ai/schneider/audio/reports/Audio-Diffusion-UNet-Alpha---VmlldzoyMjk3MzIz?accessToken=y0l3igdvnm4ogn4d3ph3b0i8twwcf7meufbviwt15f0qtasyn1i14hg340bkk1te)


## Appreciation

* [Phil Wang](https://github.com/lucidrains) for the beautiful open source contributions on [diffusion](https://github.com/lucidrains/denoising-diffusion-pytorch) and [Imagen](https://github.com/lucidrains/imagen-pytorch).

## Citations

DDPM
```bibtex
@misc{2006.11239,
Author = {Jonathan Ho and Ajay Jain and Pieter Abbeel},
Title = {Denoising Diffusion Probabilistic Models},
Year = {2020},
Month = {6},
Eprint = {arXiv:2006.11239},
}
```

Diffusion cosine schedule
```bibtex
@misc{2102.09672,
Author = {Alex Nichol and Prafulla Dhariwal},
Title = {Improved Denoising Diffusion Probabilistic Models},
Year = {2021},
Month = {2},
Eprint = {arXiv:2102.09672},
}
```

Diffusion weighted loss
```bibtex
@misc{2204.00227,
Author = {Jooyoung Choi and Jungbeom Lee and Chaehun Shin and Sungwon Kim and Hyunwoo Kim and Sungroh Yoon},
Title = {Perception Prioritized Training of Diffusion Models},
Year = {2022},
Month = {4},
Eprint = {arXiv:2204.00227},
}
```

Improved UNet architecture
```bibtex
@misc{2205.11487,
Author = {Chitwan Saharia and William Chan and Saurabh Saxena and Lala Li and Jay Whang and Emily Denton and Seyed Kamyar Seyed Ghasemipour and Burcu Karagol Ayan and S. Sara Mahdavi and Rapha Gontijo Lopes and Tim Salimans and Jonathan Ho and David J Fleet and Mohammad Norouzi},
Title = {Photorealistic Text-to-Image Diffusion Models with Deep Language Understanding},
Year = {2022},
Month = {5},
Eprint = {arXiv:2205.11487},
}
```

Elucidated diffusion
```bibtex
@misc{2206.00364,
Author = {Tero Karras and Miika Aittala and Timo Aila and Samuli Laine},
Title = {Elucidating the Design Space of Diffusion-Based Generative Models},
Year = {2022},
Month = {6},
Eprint = {arXiv:2206.00364},
}
```
