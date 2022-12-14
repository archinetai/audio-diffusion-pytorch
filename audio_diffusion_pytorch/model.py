from math import pi
from random import randint
from typing import Any, Optional, Sequence, Tuple, Union

import torch
from audio_encoders_pytorch import Encoder1d
from einops import rearrange
from torch import Tensor, nn
from tqdm import tqdm

from .diffusion import LinearSchedule, UniformDistribution, VSampler, XDiffusion
from .modules import STFT, SinusoidalEmbedding, XUNet1d, rand_bool
from .utils import (
    closest_power_2,
    default,
    downsample,
    exists,
    groupby,
    to_list,
    upsample,
)

"""
Diffusion Classes (generic for 1d data)
"""


class Model1d(nn.Module):
    def __init__(self, unet_type: str = "base", **kwargs):
        super().__init__()
        diffusion_kwargs, kwargs = groupby("diffusion_", kwargs)
        self.unet = XUNet1d(type=unet_type, **kwargs)
        self.diffusion = XDiffusion(net=self.unet, **diffusion_kwargs)

    def forward(self, x: Tensor, **kwargs) -> Tensor:
        return self.diffusion(x, **kwargs)

    def sample(self, *args, **kwargs) -> Tensor:
        return self.diffusion.sample(*args, **kwargs)


class DiffusionUpsampler1d(Model1d):
    def __init__(
        self,
        in_channels: int,
        factor: Union[int, Sequence[int]],
        factor_features: Optional[int] = None,
        *args,
        **kwargs,
    ):
        self.factors = to_list(factor)
        self.use_conditioning = exists(factor_features)

        default_kwargs = dict(
            in_channels=in_channels,
            context_channels=[in_channels],
            context_features=factor_features if self.use_conditioning else None,
        )
        super().__init__(*args, **{**default_kwargs, **kwargs})  # type: ignore

        if self.use_conditioning:
            assert exists(factor_features)
            self.to_features = SinusoidalEmbedding(dim=factor_features)

    def random_reupsample(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        batch_size, device, factors = x.shape[0], x.device, self.factors
        # Pick random factor for each batch element
        random_factors = torch.randint(0, len(factors), (batch_size,), device=device)
        x = x.clone()

        for i, factor in enumerate(factors):
            # Pick random items with current factor, skip if 0
            n = torch.count_nonzero(random_factors == i)
            if n > 0:
                waveforms = x[random_factors == i]
                # Downsample and reupsample items
                downsampled = downsample(waveforms, factor=factor)
                reupsampled = upsample(downsampled, factor=factor)
                # Save reupsampled version in place
                x[random_factors == i] = reupsampled
        return x, random_factors

    def forward(self, x: Tensor, **kwargs) -> Tensor:
        channels, factors = self.random_reupsample(x)
        features = self.to_features(factors) if self.use_conditioning else None
        return self.diffusion(x, channels_list=[channels], features=features, **kwargs)

    def sample(  # type: ignore
        self, undersampled: Tensor, factor: Optional[int] = None, *args, **kwargs
    ):
        # Either user provides factor or we pick the first
        batch_size, device = undersampled.shape[0], undersampled.device
        factor = default(factor, self.factors[0])
        # Upsample channels by interpolation
        channels = upsample(undersampled, factor=factor)
        # Compute features if conditioning on factor
        factors = torch.tensor([factor] * batch_size, device=device)
        features = self.to_features(factors) if self.use_conditioning else None
        # Diffuse upsampled
        noise = torch.randn_like(channels)
        default_kwargs = dict(channels_list=[channels], features=features)
        return super().sample(noise, **{**default_kwargs, **kwargs})  # type: ignore


class DiffusionAE1d(Model1d):
    """Diffusion Auto Encoder"""

    def __init__(
        self, in_channels: int, encoder: Encoder1d, encoder_inject_depth: int, **kwargs
    ):
        super().__init__(
            in_channels=in_channels,
            context_channels=[0] * encoder_inject_depth + [encoder.out_channels],
            **kwargs,
        )
        self.in_channels = in_channels
        self.encoder = encoder

    def forward(  # type: ignore
        self, x: Tensor, with_info: bool = False, **kwargs
    ) -> Union[Tensor, Tuple[Tensor, Any]]:
        latent, info = self.encode(x, with_info=True)
        loss = super().forward(x, channels_list=[latent], **kwargs)
        return (loss, info) if with_info else loss

    def encode(self, *args, **kwargs):
        return self.encoder(*args, **kwargs)

    def decode(self, latent: Tensor, **kwargs) -> Tensor:
        b = latent.shape[0]
        length = closest_power_2(latent.shape[2] * self.encoder.downsample_factor)
        # Compute noise by inferring shape from latent length
        noise = torch.randn(b, self.in_channels, length, device=latent.device)
        # Compute context form latent
        default_kwargs = dict(channels_list=[latent])
        # Decode by sampling while conditioning on latent channels
        return super().sample(noise, **{**default_kwargs, **kwargs})


class DiffusionVocoder1d(Model1d):
    def __init__(
        self,
        in_channels: int,
        stft_num_fft: int,
        **kwargs,
    ):
        self.frequency_channels = stft_num_fft // 2 + 1
        spectrogram_channels = in_channels * self.frequency_channels

        stft_kwargs, kwargs = groupby("stft_", kwargs)
        default_kwargs = dict(
            in_channels=spectrogram_channels, context_channels=[spectrogram_channels]
        )

        super().__init__(**{**default_kwargs, **kwargs})  # type: ignore
        self.stft = STFT(num_fft=stft_num_fft, **stft_kwargs)

    def forward_wave(self, x: Tensor, **kwargs) -> Tensor:
        # Get magnitude and phase of true wave
        magnitude, phase = self.stft.encode(x)
        return self(magnitude, phase, **kwargs)

    def forward(self, magnitude: Tensor, phase: Tensor, **kwargs) -> Tensor:  # type: ignore # noqa
        magnitude = rearrange(magnitude, "b c f t -> b (c f) t")
        phase = rearrange(phase, "b c f t -> b (c f) t")
        # Get diffusion phase loss while conditioning on magnitude (/pi [-1,1] range)
        return self.diffusion(phase / pi, channels_list=[magnitude], **kwargs)

    def sample(self, magnitude: Tensor, **kwargs):  # type: ignore
        b, c, f, t, device = *magnitude.shape, magnitude.device
        magnitude_flat = rearrange(magnitude, "b c f t -> b (c f) t")
        noise = torch.randn((b, c * f, t), device=device)
        default_kwargs = dict(channels_list=[magnitude_flat])
        phase_flat = super().sample(noise, **{**default_kwargs, **kwargs})  # type: ignore # noqa
        phase = rearrange(phase_flat, "b (c f) t -> b c f t", c=c)
        wave = self.stft.decode(magnitude, phase * pi)
        return wave


class DiffusionUpphaser1d(DiffusionUpsampler1d):
    def __init__(self, **kwargs):
        stft_kwargs, kwargs = groupby("stft_", kwargs)
        super().__init__(**kwargs)
        self.stft = STFT(**stft_kwargs)

    def random_rephase(self, x: Tensor) -> Tensor:
        magnitude, phase = self.stft.encode(x)
        phase_random = (torch.rand_like(phase) - 0.5) * 2 * pi
        wave = self.stft.decode(magnitude, phase_random)
        return wave

    def forward(self, x: Tensor, **kwargs) -> Tensor:
        rephased = self.random_rephase(x)
        resampled, factors = self.random_reupsample(rephased)
        features = self.to_features(factors) if self.use_conditioning else None
        return self.diffusion(x, channels_list=[resampled], features=features, **kwargs)


class DiffusionAR1d(Model1d):
    def __init__(
        self,
        in_channels: int,
        chunk_length: int,
        upsample: int = 0,
        dropout: float = 0.05,
        verbose: int = 0,
        **kwargs,
    ):
        self.in_channels = in_channels
        self.chunk_length = chunk_length
        self.dropout = dropout
        self.upsample = upsample
        self.verbose = verbose
        super().__init__(
            in_channels=in_channels,
            context_channels=[in_channels * (2 if upsample > 0 else 1)],
            **kwargs,
        )

    def reupsample(self, x: Tensor) -> Tensor:
        x = x.clone()
        x = downsample(x, factor=self.upsample)
        x = upsample(x, factor=self.upsample)
        return x

    def forward(self, x: Tensor, **kwargs) -> Tensor:
        b, _, t, device = *x.shape, x.device
        cl, num_chunks = self.chunk_length, t // self.chunk_length
        assert num_chunks >= 2, "Input tensor length must be >= chunk_length * 2"

        # Get prev and current target chunks
        chunk_index = randint(0, num_chunks - 2)
        chunk_pos = cl * (chunk_index + 1)
        chunk_prev = x[:, :, cl * chunk_index : chunk_pos]
        chunk_curr = x[:, :, chunk_pos : cl * (chunk_index + 2)]

        # Randomly dropout source chunks to allow for zero AR start
        if self.dropout > 0:
            batch_mask = rand_bool(shape=(b, 1, 1), proba=self.dropout, device=device)
            chunk_zeros = torch.zeros_like(chunk_prev)
            chunk_prev = torch.where(batch_mask, chunk_zeros, chunk_prev)

        # Condition on previous chunk and reupsampled current if required
        if self.upsample > 0:
            chunk_reupsampled = self.reupsample(chunk_curr)
            channels_list = [torch.cat([chunk_prev, chunk_reupsampled], dim=1)]
        else:
            channels_list = [chunk_prev]

        # Diffuse current current chunk
        return self.diffusion(chunk_curr, channels_list=channels_list, **kwargs)

    def sample(self, x: Tensor, start: Optional[Tensor] = None, **kwargs) -> Tensor:  # type: ignore # noqa
        noise = x

        if self.upsample > 0:
            # In this case we assume that x is the downsampled audio instead of noise
            upsampled = upsample(x, factor=self.upsample)
            noise = torch.randn_like(upsampled)

        b, c, t, device = *noise.shape, noise.device
        cl, num_chunks = self.chunk_length, t // self.chunk_length
        assert c == self.in_channels
        assert t % cl == 0, "noise must be divisible by chunk_length"

        # Initialize previous chunk
        if exists(start):
            chunk_prev = start[:, :, -cl:]
        else:
            chunk_prev = torch.zeros(b, c, cl).to(device)

        # Computed chunks
        chunks = []

        for i in tqdm(range(num_chunks), disable=(self.verbose == 0)):
            # Chunk noise
            chunk_start, chunk_end = cl * i, cl * (i + 1)
            noise_curr = noise[:, :, chunk_start:chunk_end]

            # Condition on previous chunk and artifically upsampled current if required
            if self.upsample > 0:
                chunk_upsampled = upsampled[:, :, chunk_start:chunk_end]
                channels_list = [torch.cat([chunk_prev, chunk_upsampled], dim=1)]
            else:
                channels_list = [chunk_prev]
            default_kwargs = dict(channels_list=channels_list)

            # Sample current chunk
            chunk_curr = super().sample(noise_curr, **{**default_kwargs, **kwargs})

            # Save chunk and use current as prev
            chunks += [chunk_curr]
            chunk_prev = chunk_curr

        return rearrange(chunks, "l b c t -> b c (l t)")


"""
Audio Diffusion Classes (specific for 1d audio data)
"""


def get_default_model_kwargs():
    return dict(
        channels=128,
        patch_size=16,
        multipliers=[1, 2, 4, 4, 4, 4, 4],
        factors=[4, 4, 4, 2, 2, 2],
        num_blocks=[2, 2, 2, 2, 2, 2],
        attentions=[0, 0, 0, 1, 1, 1, 1],
        attention_heads=8,
        attention_features=64,
        attention_multiplier=2,
        attention_use_rel_pos=False,
        diffusion_type="v",
        diffusion_sigma_distribution=UniformDistribution(),
    )


def get_default_sampling_kwargs():
    return dict(sigma_schedule=LinearSchedule(), sampler=VSampler(), clamp=True)


class AudioDiffusionModel(Model1d):
    def __init__(self, **kwargs):
        super().__init__(**{**get_default_model_kwargs(), **kwargs})

    def sample(self, *args, **kwargs):
        return super().sample(*args, **{**get_default_sampling_kwargs(), **kwargs})


class AudioDiffusionUpsampler(DiffusionUpsampler1d):
    def __init__(self, in_channels: int, **kwargs):
        default_kwargs = dict(
            **get_default_model_kwargs(),
            in_channels=in_channels,
            context_channels=[in_channels],
        )
        super().__init__(**{**default_kwargs, **kwargs})  # type: ignore

    def sample(self, *args, **kwargs):
        return super().sample(*args, **{**get_default_sampling_kwargs(), **kwargs})


class AudioDiffusionAE(DiffusionAE1d):
    def __init__(self, in_channels: int, *args, **kwargs):
        default_kwargs = dict(
            **get_default_model_kwargs(),
            in_channels=in_channels,
            encoder=Encoder1d(
                in_channels=in_channels,
                patch_size=16,
                channels=16,
                multipliers=[1, 2, 4, 4, 4, 4, 4],
                factors=[4, 4, 4, 2, 2, 2],
                num_blocks=[2, 2, 2, 2, 2, 2],
                out_channels=64,
            ),
            encoder_inject_depth=6,
        )
        super().__init__(*args, **{**default_kwargs, **kwargs})  # type: ignore

    def decode(self, *args, **kwargs):
        return super().decode(*args, **{**get_default_sampling_kwargs(), **kwargs})


class AudioDiffusionConditional(Model1d):
    def __init__(
        self,
        embedding_features: int,
        embedding_max_length: int,
        embedding_mask_proba: float = 0.1,
        **kwargs,
    ):
        self.embedding_mask_proba = embedding_mask_proba
        default_kwargs = dict(
            **get_default_model_kwargs(),
            unet_type="cfg",
            context_embedding_features=embedding_features,
            context_embedding_max_length=embedding_max_length,
        )
        super().__init__(**{**default_kwargs, **kwargs})

    def forward(self, *args, **kwargs):
        default_kwargs = dict(embedding_mask_proba=self.embedding_mask_proba)
        return super().forward(*args, **{**default_kwargs, **kwargs})

    def sample(self, *args, **kwargs):
        default_kwargs = dict(
            **get_default_sampling_kwargs(),
            embedding_scale=5.0,
        )
        return super().sample(*args, **{**default_kwargs, **kwargs})


class AudioDiffusionVocoder(DiffusionVocoder1d):
    def __init__(self, in_channels: int, **kwargs):
        default_kwargs = dict(
            in_channels=in_channels,
            stft_num_fft=1023,
            stft_hop_length=256,
            channels=512,
            multipliers=[3, 2, 1, 1, 1, 1, 1, 1],
            factors=[1, 2, 2, 2, 2, 2, 2],
            num_blocks=[1, 1, 1, 1, 1, 1, 1],
            attentions=[0, 0, 0, 0, 1, 1, 1],
            attention_heads=8,
            attention_features=64,
            attention_multiplier=2,
            attention_use_rel_pos=False,
            diffusion_type="v",
            diffusion_sigma_distribution=UniformDistribution(),
        )
        super().__init__(**{**default_kwargs, **kwargs})  # type: ignore

    def sample(self, *args, **kwargs):
        default_kwargs = dict(**get_default_sampling_kwargs())
        return super().sample(*args, **{**default_kwargs, **kwargs})


class AudioDiffusionUpphaser(DiffusionUpphaser1d):
    def __init__(self, in_channels: int, **kwargs):
        default_kwargs = dict(
            **get_default_model_kwargs(),
            in_channels=in_channels,
            context_channels=[in_channels],
            factor=1,
        )
        super().__init__(**{**default_kwargs, **kwargs})  # type: ignore

    def sample(self, *args, **kwargs):
        return super().sample(*args, **{**get_default_sampling_kwargs(), **kwargs})
