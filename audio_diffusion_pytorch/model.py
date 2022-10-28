from math import pi
from typing import Any, Optional, Sequence, Tuple, Union

import torch
from audio_encoders_pytorch import Bottleneck, Encoder1d
from einops import rearrange
from torch import Tensor, nn

from .diffusion import (
    DiffusionSampler,
    KDiffusion,
    LinearSchedule,
    Sampler,
    Schedule,
    UniformDistribution,
    VDiffusion,
    VKDiffusion,
    VSampler,
)
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

        # Check valid diffusion type
        diffusion_classes = [VDiffusion, KDiffusion, VKDiffusion]
        aliases = [t.alias for t in diffusion_classes]  # type: ignore
        message = f"diffusion_type='{diffusion_type}' must be one of {*aliases,}"
        assert diffusion_type in aliases, message

        for XDiffusion in diffusion_classes:
            if XDiffusion.alias == diffusion_type:  # type: ignore
                self.diffusion = XDiffusion(net=self.unet, **diffusion_kwargs)

    def forward(self, x: Tensor, **kwargs) -> Tensor:
        return self.diffusion(x, **kwargs)

    def sample(
        self,
        noise: Tensor,
        num_steps: int,
        sigma_schedule: Schedule,
        sampler: Sampler,
        clamp: bool,
        **kwargs,
    ) -> Tensor:
        diffusion_sampler = DiffusionSampler(
            diffusion=self.diffusion,
            sampler=sampler,
            sigma_schedule=sigma_schedule,
            num_steps=num_steps,
            clamp=clamp,
        )
        return diffusion_sampler(noise, **kwargs)


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


class DiffusionAutoencoder1d(Model1d):
    def __init__(
        self,
        in_channels: int,
        channels: int,
        patch_blocks: int,
        patch_factor: int,
        multipliers: Sequence[int],
        factors: Sequence[int],
        num_blocks: Sequence[int],
        resnet_groups: int,
        kernel_multiplier_downsample: int,
        encoder_depth: int,
        encoder_num_blocks: Optional[Sequence[int]] = None,
        bottleneck: Union[Bottleneck, Sequence[Bottleneck]] = [],
        bottleneck_channels: Optional[int] = None,
        use_stft: bool = False,
        **kwargs,
    ):
        self.in_channels = in_channels
        encoder_num_blocks = default(encoder_num_blocks, num_blocks)
        assert_message = "The number of encoder_num_blocks must match encoder_depth"
        assert len(encoder_num_blocks) >= encoder_depth, assert_message
        assert patch_blocks == 1, "patch_blocks != 1 not supported"
        assert not use_stft, "use_stft not supported"
        self.factor = patch_factor * prod(factors[0:encoder_depth])

        context_channels = [0] * encoder_depth
        if exists(bottleneck_channels):
            context_channels += [bottleneck_channels]
        else:
            context_channels += [channels * multipliers[encoder_depth]]

        super().__init__(
            in_channels=in_channels,
            channels=channels,
            patch_blocks=patch_blocks,
            patch_factor=patch_factor,
            multipliers=multipliers,
            factors=factors,
            num_blocks=num_blocks,
            resnet_groups=resnet_groups,
            kernel_multiplier_downsample=kernel_multiplier_downsample,
            context_channels=context_channels,
            **kwargs,
        )

        self.bottlenecks = nn.ModuleList(to_list(bottleneck))
        self.encoder = Encoder1d(
            in_channels=in_channels,
            channels=channels,
            patch_size=patch_factor,
            multipliers=multipliers[0 : encoder_depth + 1],
            factors=factors[0:encoder_depth],
            num_blocks=encoder_num_blocks[0:encoder_depth],
            resnet_groups=resnet_groups,
            out_channels=bottleneck_channels,
        )

    def encode(
        self, x: Tensor, with_info: bool = False
    ) -> Union[Tensor, Tuple[Tensor, Any]]:
        latent, info = self.encoder(x, with_info=True)
        for bottleneck in self.bottlenecks:
            x, info_bottleneck = bottleneck(x, with_info=True)
            info = {**info, **prefix_dict("bottleneck_", info_bottleneck)}
        return (latent, info) if with_info else latent

    def forward(  # type: ignore
        self, x: Tensor, with_info: bool = False, **kwargs
    ) -> Union[Tensor, Tuple[Tensor, Any]]:
        latent, info = self.encode(x, with_info=True)
        loss = self.diffusion(x, channels_list=[latent], **kwargs)
        return (loss, info) if with_info else loss

    def decode(self, latent: Tensor, **kwargs) -> Tensor:
        b, length = latent.shape[0], latent.shape[2] * self.factor
        # Compute noise by inferring shape from latent length
        noise = torch.randn(b, self.in_channels, length).to(latent)
        # Compute context form latent
        default_kwargs = dict(channels_list=[latent])
        # Decode by sampling while conditioning on latent channels
        return super().sample(noise, **{**default_kwargs, **kwargs})  # type: ignore


class DiffusionVocoder1d(Model1d):
    def __init__(self, in_channels: int, stft_num_fft: int, **kwargs):
        self.stft_num_fft = stft_num_fft
        spectrogram_channels = stft_num_fft // 2 + 1
        default_kwargs = dict(
            in_channels=in_channels,
            use_stft=True,
            stft_num_fft=stft_num_fft,
            context_channels=[in_channels * spectrogram_channels],
        )
        super().__init__(**{**default_kwargs, **kwargs})  # type: ignore

    def forward(self, x: Tensor, **kwargs) -> Tensor:
        # Get magnitude spectrogram from true wave
        magnitude, _ = self.unet.stft.encode(x)
        magnitude = rearrange(magnitude, "b c f t -> b (c f) t")
        # Get diffusion loss while conditioning on magnitude
        return self.diffusion(x, channels_list=[magnitude], **kwargs)

    def sample(self, spectrogram: Tensor, **kwargs):  # type: ignore
        b, c, _, t, device = *spectrogram.shape, spectrogram.device
        magnitude = rearrange(spectrogram, "b c f t -> b (c f) t")
        timesteps = closest_power_2(self.unet.stft.hop_length * t)
        noise = torch.randn((b, c, timesteps), device=device)
        default_kwargs = dict(channels_list=[magnitude])
        return super().sample(noise, **{**default_kwargs, **kwargs})  # type: ignore # noqa


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
        patch_blocks=1,
        patch_factor=16,
        multipliers=[1, 2, 4, 4, 4, 4, 4],
        factors=[4, 4, 4, 2, 2, 2],
        num_blocks=[2, 2, 2, 2, 2, 2],
        attentions=[0, 0, 0, 1, 1, 1, 1],
        attention_heads=8,
        attention_features=64,
        attention_multiplier=2,
        attention_use_rel_pos=False,
        resnet_groups=8,
        kernel_multiplier_downsample=2,
        use_nearest_upsample=False,
        use_skip_scale=True,
        use_context_time=True,
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
            **get_default_model_kwargs(), encoder_depth=4, encoder_channels=64
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
            channels=64,
            patch_blocks=1,
            patch_factor=1,
            multipliers=[48, 32, 16, 8, 8, 8, 8],
            factors=[2, 2, 2, 1, 1, 1],
            num_blocks=[1, 1, 1, 1, 1, 1],
            attentions=[0, 0, 0, 1, 1, 1],
            attention_heads=8,
            attention_features=64,
            attention_multiplier=2,
            attention_use_rel_pos=False,
            resnet_groups=8,
            kernel_multiplier_downsample=2,
            use_nearest_upsample=False,
            use_skip_scale=True,
            use_context_time=True,
            use_magnitude_channels=False,
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
