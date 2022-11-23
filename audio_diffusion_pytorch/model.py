from math import pi
from typing import Any, Optional, Sequence, Tuple, Union

import torch
from audio_encoders_pytorch import Bottleneck, Encoder1d
from einops import rearrange
from torch import Tensor, nn

from .diffusion import LinearSchedule, UniformDistribution, VSampler, XDiffusion
from .modules import STFT, SinusoidalEmbedding, UNet1d, UNetConditional1d
from .utils import (
    closest_power_2,
    default,
    downsample,
    exists,
    groupby,
    prefix_dict,
    prod,
    to_list,
    upsample,
)

"""
Diffusion Classes (generic for 1d data)
"""


class Model1d(nn.Module):
    def __init__(
        self, diffusion_type: str, use_classifier_free_guidance: bool = False, **kwargs
    ):
        super().__init__()
        diffusion_kwargs, kwargs = groupby("diffusion_", kwargs)

        UNet = UNetConditional1d if use_classifier_free_guidance else UNet1d
        self.unet = UNet(**kwargs)

        self.diffusion = XDiffusion(
            type=diffusion_type, net=self.unet, **diffusion_kwargs
        )

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


class DiffusionAutoencoder1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        encoder_inject_depth: int,
        encoder_channels: int,
        encoder_factors: Sequence[int],
        encoder_multipliers: Sequence[int],
        diffusion_type: str,
        encoder_patch_size: int = 1,
        bottleneck: Union[Bottleneck, Sequence[Bottleneck]] = [],
        bottleneck_channels: Optional[int] = None,
        **kwargs,
    ):
        super().__init__()
        self.in_channels = in_channels

        encoder_kwargs, kwargs = groupby("encoder_", kwargs)
        diffusion_kwargs, kwargs = groupby("diffusion_", kwargs)

        # Compute context channels
        context_channels = [0] * encoder_inject_depth
        if exists(bottleneck_channels):
            context_channels += [bottleneck_channels]
        else:
            context_channels += [encoder_channels * encoder_multipliers[-1]]

        self.unet = UNet1d(
            in_channels=in_channels, context_channels=context_channels, **kwargs
        )

        self.diffusion = XDiffusion(
            type=diffusion_type, net=self.unet, **diffusion_kwargs
        )

        self.encoder = Encoder1d(
            in_channels=in_channels,
            channels=encoder_channels,
            patch_size=encoder_patch_size,
            factors=encoder_factors,
            multipliers=encoder_multipliers,
            out_channels=bottleneck_channels,
            **encoder_kwargs,
        )

        self.encoder_downsample_factor = encoder_patch_size * prod(encoder_factors)
        self.bottleneck_channels = bottleneck_channels
        self.bottlenecks = nn.ModuleList(to_list(bottleneck))

    def encode(
        self, x: Tensor, with_info: bool = False
    ) -> Union[Tensor, Tuple[Tensor, Any]]:
        latent, info = self.encoder(x, with_info=True)
        # Apply bottlenecks if present
        for bottleneck in self.bottlenecks:
            latent, info_bottleneck = bottleneck(latent, with_info=True)
            info = {**info, **prefix_dict("bottleneck_", info_bottleneck)}
        return (latent, info) if with_info else latent

    def forward(  # type: ignore
        self, x: Tensor, with_info: bool = False, **kwargs
    ) -> Union[Tensor, Tuple[Tensor, Any]]:
        latent, info = self.encode(x, with_info=True)
        loss = self.diffusion(x, channels_list=[latent], **kwargs)
        return (loss, info) if with_info else loss

    def decode(self, latent: Tensor, **kwargs) -> Tensor:
        b = latent.shape[0]
        length = latent.shape[2] * self.encoder_downsample_factor
        # Compute noise by inferring shape from latent length
        noise = torch.randn(b, self.in_channels, length, device=latent.device)
        # Compute context form latent
        default_kwargs = dict(channels_list=[latent])
        # Decode by sampling while conditioning on latent channels
        return self.sample(noise, **{**default_kwargs, **kwargs})

    def sample(self, *args, **kwargs) -> Tensor:
        return self.diffusion.sample(*args, **kwargs)


class DiffusionMAE1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        encoder_inject_depth: int,
        encoder_channels: int,
        encoder_factors: Sequence[int],
        encoder_multipliers: Sequence[int],
        diffusion_type: str,
        stft_num_fft: int,
        stft_hop_length: int,
        stft_use_complex: bool,
        stft_window_length: Optional[int] = None,
        encoder_patch_size: int = 1,
        bottleneck: Union[Bottleneck, Sequence[Bottleneck]] = [],
        bottleneck_channels: Optional[int] = None,
        **kwargs,
    ):
        super().__init__()
        self.in_channels = in_channels

        encoder_kwargs, kwargs = groupby("encoder_", kwargs)
        diffusion_kwargs, kwargs = groupby("diffusion_", kwargs)
        stft_kwargs, kwargs = groupby("stft_", kwargs)

        # Compute context channels
        context_channels = [0] * encoder_inject_depth
        if exists(bottleneck_channels):
            context_channels += [bottleneck_channels]
        else:
            context_channels += [encoder_channels * encoder_multipliers[-1]]

        self.spectrogram_channels = stft_num_fft // 2 + 1
        self.stft_hop_length = stft_hop_length

        self.encoder_stft = STFT(
            num_fft=stft_num_fft,
            hop_length=stft_hop_length,
            window_length=stft_window_length,
            use_complex=False,  # Magnitude encoding
        )

        self.unet = UNet1d(
            in_channels=in_channels,
            context_channels=context_channels,
            use_stft=True,
            stft_use_complex=stft_use_complex,
            stft_num_fft=stft_num_fft,
            stft_hop_length=stft_hop_length,
            stft_window_length=stft_window_length,
            **kwargs,
        )

        self.diffusion = XDiffusion(
            type=diffusion_type, net=self.unet, **diffusion_kwargs
        )

        self.encoder = Encoder1d(
            in_channels=in_channels * self.spectrogram_channels,
            channels=encoder_channels,
            patch_size=encoder_patch_size,
            factors=encoder_factors,
            multipliers=encoder_multipliers,
            out_channels=bottleneck_channels,
            **encoder_kwargs,
        )

        self.encoder_downsample_factor = encoder_patch_size * prod(encoder_factors)
        self.bottleneck_channels = bottleneck_channels
        self.bottlenecks = nn.ModuleList(to_list(bottleneck))

    def encode(
        self, x: Tensor, with_info: bool = False
    ) -> Union[Tensor, Tuple[Tensor, Any]]:
        # Extract magnitude and encode
        magnitude, _ = self.encoder_stft.encode(x)
        magnitude_flat = rearrange(magnitude, "b c f t -> b (c f) t")
        latent, info = self.encoder(magnitude_flat, with_info=True)
        # Apply bottlenecks if present
        for bottleneck in self.bottlenecks:
            latent, info_bottleneck = bottleneck(latent, with_info=True)
            info = {**info, **prefix_dict("bottleneck_", info_bottleneck)}
        return (latent, info) if with_info else latent

    def forward(  # type: ignore
        self, x: Tensor, with_info: bool = False, **kwargs
    ) -> Union[Tensor, Tuple[Tensor, Any]]:
        latent, info = self.encode(x, with_info=True)
        loss = self.diffusion(x, channels_list=[latent], **kwargs)
        return (loss, info) if with_info else loss

    def decode(self, latent: Tensor, **kwargs) -> Tensor:
        b = latent.shape[0]
        length = closest_power_2(
            self.stft_hop_length * latent.shape[2] * self.encoder_downsample_factor
        )
        # Compute noise by inferring shape from latent length
        noise = torch.randn(b, self.in_channels, length, device=latent.device)
        # Compute context form latent
        default_kwargs = dict(channels_list=[latent])
        # Decode by sampling while conditioning on latent channels
        return self.sample(noise, **{**default_kwargs, **kwargs})  # type: ignore

    def sample(self, *args, **kwargs) -> Tensor:
        return self.diffusion.sample(*args, **kwargs)


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


"""
Audio Diffusion Classes (specific for 1d audio data)
"""


def get_default_model_kwargs():
    return dict(
        channels=128,
        patch_factor=16,
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


class AudioDiffusionAutoencoder(DiffusionAutoencoder1d):
    def __init__(self, *args, **kwargs):
        default_kwargs = dict(
            **get_default_model_kwargs(),
            encoder_inject_depth=6,
            encoder_channels=16,
            encoder_patch_size=16,
            encoder_multipliers=[1, 2, 4, 4, 4, 4, 4],
            encoder_factors=[4, 4, 4, 2, 2, 2],
            encoder_num_blocks=[2, 2, 2, 2, 2, 2],
            bottleneck_channels=64,
        )
        super().__init__(*args, **{**default_kwargs, **kwargs})  # type: ignore

    def decode(self, *args, **kwargs):
        return super().decode(*args, **{**get_default_sampling_kwargs(), **kwargs})


class AudioDiffusionMAE(DiffusionMAE1d):
    def __init__(self, *args, **kwargs):
        default_kwargs = dict(
            diffusion_type="v",
            diffusion_sigma_distribution=UniformDistribution(),
            stft_num_fft=1023,
            stft_hop_length=256,
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
            context_embedding_features=embedding_features,
            context_embedding_max_length=embedding_max_length,
            use_classifier_free_guidance=True,
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


""" Pretrained Models Helper """

REVISION = {"dmae1d-ATC64-v1": "07885065867977af43b460bb9c1422bdc90c29a0"}


class AudioModel:
    @staticmethod
    def from_pretrained(name: str) -> nn.Module:
        from transformers import AutoModel

        return AutoModel.from_pretrained(
            f"archinetai/{name}", trust_remote_code=True, revision=REVISION[name]
        )
