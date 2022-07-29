
# Audio Diffusion - PyTorch

Unconditional audio generation using diffusion models, in PyTorch. The goal of this repository is to explore different architectures and diffusion models to generate audio (speech and music) directly from/to the waveform.
Progress will be documented in the [experiments](#experiments) section.

## Install

```bash
pip install audio-diffusion-pytorch
```

[![PyPI - Python Version](https://img.shields.io/pypi/v/audio-diffusion-pytorch?style=flat&colorA=0f0f0f&colorB=0f0f0f)](https://pypi.org/project/audio-diffusion-pytorch/)

## Usage


### UNet1d
```py
from audio_diffusion_pytorch import UNet1d

# UNet used to denoise our 1D (audio) data
unet = UNet1d(
    in_channels=1,
    patch_size=16,
    channels=128,
    multipliers=[1, 2, 4, 4, 4, 4, 4],
    factors=[4, 4, 4, 2, 2, 2],
    attentions=[False, False, False, True, True, True],
    num_blocks=[2, 2, 2, 2, 2, 2],
    attention_heads=8,
    attention_features=64,
    attention_multiplier=2,
    resnet_groups=8,
    kernel_multiplier_downsample=2,
    kernel_sizes_init=[1, 3, 7],
    use_nearest_upsample=False,
    use_skip_scale=True,
    use_attention_bottleneck=True,
    use_learned_time_embedding=True,
)

x = torch.randn(3, 1, 2 ** 16)
t = torch.tensor([0.2, 0.8, 0.3])

y = unet(x, t) # [2, 1, 32768], 2 samples of ~1.5 seconds of generated audio at 22050kHz
```

### Diffusion

#### Training
```python
from audio_diffusion_pytorch import Diffusion, LogNormalSampler

diffusion = Diffusion(
    net=unet,
    sigma_sampler=LogNormalSampler(mean = -3.0, std = 1.0),
    sigma_data=0.1
)

x = torch.randn(3, 1, 2 ** 16) # Batch of training audio samples
loss = diffusion(x)
loss.backward() # Do this many times
```

#### Sampling
```python
from audio_diffusion_pytorch import DiffusionSampler, KerrasSchedule

sampler = DiffusionSampler(
    diffusion,
    num_steps=50, # Range 32-1000, higher for better quality
    sigma_schedule=KerrasSchedule(
        sigma_min=0.002,
        sigma_max=1
    ),
    s_tmin=0,
    s_tmax=10,
    s_churn=40,
    s_noise=1.003
)
# Generate a sample starting from the provided noise
y = sampler(x = torch.randn(1,1,2 ** 15))
```

#### Inpainting

```py
from audio_diffusion_pytorch import DiffusionInpainter, KerrasSchedule

inpainter = DiffusionInpainter(
    diffusion,
    num_steps=50, # Range 32-1000, higher for better quality
    num_resamples=5, # Range 1-10, higher for better quality
    sigma_schedule=KerrasSchedule(
        sigma_min=0.002,
        sigma_max=1
    ),
    s_tmin=0,
    s_tmax=10,
    s_churn=40,
    s_noise=1.003
)

inpaint = torch.randn(1,1,2 ** 15) # Start track, e.g. one sampled with DiffusionSampler
inpaint_mask = torch.randint(0,2, (1,1,2 ** 15), dtype=torch.bool) # Set to `True` the parts you want to keep
y = inpainter(inpaint = inpaint, inpaint_mask = inpaint_mask)
```

#### Infinite Generation
```python
from audio_diffusion_pytorch import SpanBySpanComposer

composer = SpanBySpanComposer(
    inpainter,
    num_spans=4 # Number of spans to inpaint after provided input
)
y_long = composer(y, keep_start=True) # [1, 1, 98304]
```


## Experiments


| Report | Snapshot | Description |
| --- | --- | --- |
| [Alpha](https://wandb.ai/schneider/audio/reports/Audio-Diffusion-UNet-Alpha---VmlldzoyMjk3MzIz?accessToken=y0l3igdvnm4ogn4d3ph3b0i8twwcf7meufbviwt15f0qtasyn1i14hg340bkk1te) | [6bd9279f19](https://github.com/archinetai/audio-diffusion-pytorch/tree/6bd9279f192fc0c11eb8a21cd919d9c41181bf35) | Initial tests on LJSpeech dataset with new architecture and basic DDPM diffusion model. |
| [Bravo](https://wandb.ai/schneider/audio/reports/Audio-Diffusion-Bravo---VmlldzoyMzE4NjIx?accessToken=qt2w1jeqch9l5v3ffjns99p69jsmexk849dszyiennfbivgg396378u6ken2fm2d) | [a05f30aa94](https://github.com/archinetai/audio-diffusion-pytorch/tree/a05f30aa94e07600038d36cfb96f8492ef735a99) | Elucidated diffusion, improved architecture with patching, longer duration, initial good (unsupervised) results on LJSpeech.
| [Charlie](https://wandb.ai/schneider/audio/reports/Audio-Diffusion-Charlie---VmlldzoyMzYyNDA1?accessToken=71gmurcwndv5e2abqrjnlh3n74j5555j3tycpd7h40tnv8fvb17k5pjkb57j9xxa) | (current) | Train on music with [YoutubeDataset](https://github.com/archinetai/audio-data-pytorch), larger patch tests for longer tracks, inpainting tests, initial test with infinite generation using SpanBySpanComposer. |


## Appreciation

* [Phil Wang](https://github.com/lucidrains) for the beautiful open source contributions on [diffusion](https://github.com/lucidrains/denoising-diffusion-pytorch) and [Imagen](https://github.com/lucidrains/imagen-pytorch).
* [Katherine Crowson](https://github.com/crowsonkb) for the experiments with [k-diffusion](https://github.com/crowsonkb/k-diffusion).

## Citations

DDPM
```bibtex
@misc{2006.11239,
Author = {Jonathan Ho and Ajay Jain and Pieter Abbeel},
Title = {Denoising Diffusion Probabilistic Models},
Year = {2020},
Eprint = {arXiv:2006.11239},
}
```

Diffusion inpainting
```bibtex
@misc{2201.09865,
Author = {Andreas Lugmayr and Martin Danelljan and Andres Romero and Fisher Yu and Radu Timofte and Luc Van Gool},
Title = {RePaint: Inpainting using Denoising Diffusion Probabilistic Models},
Year = {2022},
Eprint = {arXiv:2201.09865},
}
```

Diffusion cosine schedule
```bibtex
@misc{2102.09672,
Author = {Alex Nichol and Prafulla Dhariwal},
Title = {Improved Denoising Diffusion Probabilistic Models},
Year = {2021},
Eprint = {arXiv:2102.09672},
}
```

Diffusion weighted loss
```bibtex
@misc{2204.00227,
Author = {Jooyoung Choi and Jungbeom Lee and Chaehun Shin and Sungwon Kim and Hyunwoo Kim and Sungroh Yoon},
Title = {Perception Prioritized Training of Diffusion Models},
Year = {2022},
Eprint = {arXiv:2204.00227},
}
```

Improved UNet architecture
```bibtex
@misc{2205.11487,
Author = {Chitwan Saharia and William Chan and Saurabh Saxena and Lala Li and Jay Whang and Emily Denton and Seyed Kamyar Seyed Ghasemipour and Burcu Karagol Ayan and S. Sara Mahdavi and Rapha Gontijo Lopes and Tim Salimans and Jonathan Ho and David J Fleet and Mohammad Norouzi},
Title = {Photorealistic Text-to-Image Diffusion Models with Deep Language Understanding},
Year = {2022},
Eprint = {arXiv:2205.11487},
}
```

Elucidated diffusion
```bibtex
@misc{2206.00364,
Author = {Tero Karras and Miika Aittala and Timo Aila and Samuli Laine},
Title = {Elucidating the Design Space of Diffusion-Based Generative Models},
Year = {2022},
Eprint = {arXiv:2206.00364},
}
```
