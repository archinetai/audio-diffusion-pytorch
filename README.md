<img src="./LOGO.png"></img>

Unconditional audio generation using diffusion models, in PyTorch. The goal of this repository is to explore different architectures and diffusion models to generate audio (speech and music) directly from/to the waveform.
Progress will be documented in the [experiments](#experiments) section. You can use the [`audio-diffusion-pytorch-trainer`](https://github.com/archinetai/audio-diffusion-pytorch-trainer) to run your own experiments – please share your findings in the [discussions](https://github.com/archinetai/audio-diffusion-pytorch/discussions) page!

## Install

```bash
pip install audio-diffusion-pytorch
```

[![PyPI - Python Version](https://img.shields.io/pypi/v/audio-diffusion-pytorch?style=flat&colorA=black&colorB=black)](https://pypi.org/project/audio-diffusion-pytorch/)
[![Downloads](https://static.pepy.tech/personalized-badge/audio-diffusion-pytorch?period=total&units=international_system&left_color=black&right_color=black&left_text=Downloads)](https://pepy.tech/project/audio-diffusion-pytorch)
[![HuggingFace](https://img.shields.io/badge/Trained%20Models-%F0%9F%A4%97-yellow?style=flat&colorA=black&colorB=black)](https://huggingface.co/archinetai/audio-diffusion-pytorch/tree/main)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](
https://colab.research.google.com/gist/flavioschneider/d1f67b07ffcbf6fd09fdd27515ba3701/audio-diffusion-pytorch-v0-2.ipynb)



## Usage

### Generation
```py
from audio_diffusion_pytorch import AudioDiffusionModel

model = AudioDiffusionModel(in_channels=1)

# Train model with audio sources
x = torch.randn(2, 1, 2 ** 18) # [batch, in_channels, samples], 2**18 ≈ 12s of audio at a frequency of 22050
loss = model(x)
loss.backward() # Do this many times

# Sample 2 sources given start noise
noise = torch.randn(2, 1, 2 ** 18)
sampled = model.sample(
    noise=noise,
    num_steps=5 # Suggested range: 2-50
) # [2, 1, 2 ** 18]
```

### Upsampling
```py
from audio_diffusion_pytorch import AudioDiffusionUpsampler

upsampler = AudioDiffusionUpsampler(
    in_channels=1,
    factor=8,
)

# Train on high frequency data
x = torch.randn(2, 1, 2 ** 18)
loss = upsampler(x)
loss.backward()

# Given start undersampled source, samples upsampled source
undersampled = torch.randn(1, 1, 2 ** 15)
upsampled = upsampler.sample(
    undersampled,
    num_steps=5
) # [1, 1, 2 ** 18]
```

### Autoencoding
```py
from audio_diffusion_pytorch import AudioDiffusionAutoencoder

autoencoder = AudioDiffusionAutoencoder(
    in_channels=1,
    encoder_depth=4,
    encoder_channels=32
)

# Train on audio samples
x = torch.randn(2, 1, 2 ** 18)
loss = autoencoder(x)
loss.backward()

# Encode audio source into latent
x = torch.randn(2, 1, 2 ** 18)
latent = autoencoder.encode(x) # [2, 32, 128]

# Decode latent by diffusion sampling
decoded = autoencoder.decode(
    latent,
    num_steps=5
) # [2, 32, 2**18]
```


### Conditional Generation
```py
from audio_diffusion_pytorch import AudioDiffusionConditional

model = AudioDiffusionConditional(
    in_channels=1,
    embedding_max_length=512,
    embedding_features=768,
    embedding_mask_proba=0.1 # Conditional dropout of batch elements
)

# Train on pairs of audio and embedding data (e.g. from a transformer output)
x = torch.randn(2, 1, 2 ** 18)
embedding = torch.randn(2, 512, 768)
loss = model(x, embedding=embedding)
loss.backward()

# Given start embedding and noise sample new source
embedding = torch.randn(1, 512, 768)
noise = torch.randn(1, 1, 2 ** 18)
sampled = model.sample(
    noise,
    embedding=embedding,
    embedding_scale=5.0, # Classifier-free guidance scale
    num_steps=5
) # [1, 1, 2 ** 18]
```

## Usage with Components

### UNet1d
```py
from audio_diffusion_pytorch import UNet1d

# UNet used to denoise our 1D (audio) data
unet = UNet1d(
    in_channels=1,
    channels=128,
    patch_factor=16,
    patch_blocks=1,
    multipliers=[1, 2, 4, 4, 4, 4, 4],
    factors=[4, 4, 4, 2, 2, 2],
    attentions=[0, 0, 0, 1, 1, 1, 1],
    num_blocks=[2, 2, 2, 2, 2, 2],
    attention_heads=8,
    attention_features=64,
    attention_multiplier=2,
    resnet_groups=8,
    kernel_multiplier_downsample=2,
    use_nearest_upsample=False,
    use_skip_scale=True,
    use_context_time=True,
    use_magnitude_channels=False
)

x = torch.randn(3, 1, 2 ** 16)
t = torch.tensor([0.2, 0.8, 0.3])

y = unet(x, t) # [3, 1, 32768], compute 3 samples of ~1.5 seconds at 22050Hz with the given noise levels t
```

### Diffusion

#### Training
```python
from audio_diffusion_pytorch import KDiffusion, VDiffusion, LogNormalDistribution, VDistribution

# Either use KDiffusion
diffusion = KDiffusion(
    net=unet,
    sigma_distribution=LogNormalDistribution(mean = -3.0, std = 1.0),
    sigma_data=0.1,
    dynamic_threshold=0.0
)

# Or use VDiffusion
diffusion = VDiffusion(
    net=unet,
    sigma_distribution=VDistribution()
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
| [Delta](https://wandb.ai/schneider/audio/reports/Audio-Diffusion-Delta---VmlldzoyNDYyMzk1?accessToken=n1d34n35qserpx7nhskkfdm1q12hlcxx1qcmfw5ypz53kjkzoh0ge2uvhshiseqx) | [672876bf13](https://github.com/archinetai/audio-diffusion-pytorch/tree/672876bf1373b1f10afd3adc8b3b984495bca91a) | Test model with the faster `ADPM2` sampler and dynamic thresholding. |
| [Echo](https://wandb.ai/schneider/audio/reports/Audio-Diffusion-Echo---VmlldzoyNTU2NTcw?accessToken=sthdn25n8is30gjo2x0w4fs9hwbua23rlbg7o4bv8h17y47xdtruiiyb33aoc5h4) | (current) | Test `AudioDiffusionUpsampler`.

## TODO

- [x] Add elucidated diffusion.
- [x] Add ancestral DPM2 sampler.
- [x] Add dynamic thresholding.
- [x] Add (variational) autoencoder option to compress audio before diffusion (removed).
- [x] Fix inpainting and make it work with ADPM2 sampler.
- [x] Add trainer with experiments.
- [x] Add diffusion upsampler.
- [x] Add ancestral euler sampler `AEulerSampler`.
- [x] Add diffusion autoencoder.
- [x] Add diffusion upsampler.
- [x] Add autoencoder bottleneck option for quantization.
- [x] Add option to provide context tokens (cross attention).
- [x] Add conditional model with classifier-free guidance.
- [x] Add option to provide context features mapping.
- [x] Add option to change number of (cross) attention blocks.
- [x] Add `VDiffusionn` option.
- [ ] Add flash attention.


## Appreciation

* [StabilityAI](https://stability.ai/) for the compute, [Zach](https://github.com/zqevans) and everyone else from [HarmonAI](https://www.harmonai.org/) for the interesting research discussions.
* [ETH Zurich](https://inf.ethz.ch/) for the resources, [Zhijing Jin](https://zhijing-jin.com/), [Mrinmaya Sachan](http://www.mrinmaya.io/), and [Bernhard Schoelkopf](https://is.mpg.de/~bs) for supervising this Thesis.
* [Phil Wang](https://github.com/lucidrains) for the beautiful open source contributions on [diffusion](https://github.com/lucidrains/denoising-diffusion-pytorch) and [Imagen](https://github.com/lucidrains/imagen-pytorch).
* [Katherine Crowson](https://github.com/crowsonkb) for the experiments with [k-diffusion](https://github.com/crowsonkb/k-diffusion) and the insane collection of samplers.

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
