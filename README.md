<img src="./LOGO.png"></img>

Unconditional audio generation using diffusion models, in PyTorch. The goal of this repository is to explore different architectures and diffusion models to generate audio (speech and music) directly from/to the waveform.
Progress will be documented in the [experiments](#experiments) section.

## Install

```bash
pip install audio-diffusion-pytorch
```

[![PyPI - Python Version](https://img.shields.io/pypi/v/audio-diffusion-pytorch?style=flat&colorA=0f0f0f&colorB=0f0f0f)](https://pypi.org/project/audio-diffusion-pytorch/)

## Usage

```py
from audio_diffusion_pytorch import AudioDiffusionModel

model = AudioDiffusionModel()

# Train model with audio sources
x = torch.randn(2, 1, 2 ** 18) # [batch, channels, samples], 2**18 ≈ 12s of audio at a frequency of 22050
loss = model(x)
loss.backward() # Do this many times

# Sample 2 sources given start noise
noise = torch.randn(2, 1, 2 ** 18)
sampled = model.sample(
    noise=noise,
    num_steps=5 # Suggested range: 2-100
) # [2, 1, 262144]
```

## Usage with Components

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

y = unet(x, t) # [3, 1, 32768], compute 3 samples of ~1.5 seconds at 22050Hz with the given noise levels t
```

### Diffusion

#### Training
```python
from audio_diffusion_pytorch import Diffusion, LogNormalDistribution

diffusion = Diffusion(
    net=unet,
    sigma_distribution=LogNormalDistribution(mean = -3.0, std = 1.0),
    sigma_data=0.1,
    dynamic_threshold=0.95
)

x = torch.randn(3, 1, 2 ** 18) # Batch of training audio samples
loss = diffusion(x)
loss.backward() # Do this many times
```

#### Sampling
```python
from audio_diffusion_pytorch import DiffusionSampler, KarrasSchedule

sampler = DiffusionSampler(
    diffusion,
    num_steps=5, # Suggested range 2-100, higher better quality but takes longer
    sampler=ADPM2Sampler(rho=1),
    sigma_schedule=KarrasSchedule(sigma_min=0.0001, sigma_max=3.0, rho=9.0)
)
# Generate a sample starting from the provided noise
y = sampler(noise = torch.randn(1,1,2 ** 18))
```

#### Inpainting

```py
from audio_diffusion_pytorch import DiffusionInpainter, KarrasSchedule, ADPM2Sampler

inpainter = DiffusionInpainter(
    diffusion,
    num_steps=5, # Suggested range 2-100, higher for better quality
    num_resamples=1, # Suggested range 1-10, higher for better quality
    sigma_schedule=KarrasSchedule(sigma_min=0.0001, sigma_max=3.0, rho=9.0),
    sampler=ADPM2Sampler(rho=1.0),
)

inpaint = torch.randn(1,1,2 ** 18) # Start track, e.g. one sampled with DiffusionSampler
inpaint_mask = torch.randint(0,2, (1,1,2 ** 18), dtype=torch.bool) # Set to `True` the parts you want to keep
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
| [Charlie](https://wandb.ai/schneider/audio/reports/Audio-Diffusion-Charlie---VmlldzoyMzYyNDA1?accessToken=71gmurcwndv5e2abqrjnlh3n74j5555j3tycpd7h40tnv8fvb17k5pjkb57j9xxa) | [50ecc30d70](https://github.com/archinetai/audio-diffusion-pytorch/tree/50ecc30d70a211b92cb9c38d4b0250d7cc30533f) | Train on music with [YoutubeDataset](https://github.com/archinetai/audio-data-pytorch), larger patch tests for longer tracks, inpainting tests, initial test with infinite generation using SpanBySpanComposer. |
| [Delta](https://wandb.ai/schneider/audio/reports/Audio-Diffusion-Delta---VmlldzoyNDYyMzk1?accessToken=n1d34n35qserpx7nhskkfdm1q12hlcxx1qcmfw5ypz53kjkzoh0ge2uvhshiseqx) | (current) | Test model with the faster `ADPM2` sampler and dynamic thresholding. |

## TODO

- [x] Add elucidated diffusion.
- [x] Add ancestral DPM2 sampler.
- [x] Add dynamic thresholding.
- [x] Add (variational) autoencoder option to compress audio before diffusion.
- [x] Fix inpainting and make it work with ADPM2 sampler.

## Appreciation

* [ETH Zurich](https://inf.ethz.ch/) for the compute, [Zhijing Jin](https://zhijing-jin.com/), [Mrinmaya Sachan](http://www.mrinmaya.io/), and [Bernhard Schoelkopf](https://is.mpg.de/~bs) for supervising this Thesis.
* [Phil Wang](https://github.com/lucidrains) for the beautiful open source contributions on [diffusion](https://github.com/lucidrains/denoising-diffusion-pytorch) and [Imagen](https://github.com/lucidrains/imagen-pytorch).
* [Katherine Crowson](https://github.com/crowsonkb) for the experiments with [k-diffusion](https://github.com/crowsonkb/k-diffusion) and discovering the insane `ADPM2` sampler.

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
