from typing import Any, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor, nn

from .diffusion import (
    AEulerSampler,
    Diffusion,
    DiffusionSampler,
    KarrasSchedule,
    KDiffusion,
    Sampler,
    Schedule,
    VDiffusion,
    VDistribution,
)
from .modules import (
    Bottleneck,
    MultiEncoder1d,
    SinusoidalEmbedding,
    UNet1d,
    UNetConditional1d,
)
from .utils import default, downsample, exists, groupby_kwargs_prefix, to_list, upsample

"""
Diffusion Classes (generic for 1d data)
"""


class Model1d(nn.Module):
    def __init__(
        self, diffusion_type: str, use_classifier_free_guidance: bool = False, **kwargs
    ):
        super().__init__()
        diffusion_kwargs, kwargs = groupby_kwargs_prefix("diffusion_", kwargs)

        UNet = UNetConditional1d if use_classifier_free_guidance else UNet1d
        self.unet = UNet(**kwargs)

        if diffusion_type == "v":
            self.diffusion: Diffusion = VDiffusion(net=self.unet, **diffusion_kwargs)
        elif diffusion_type == "k":
            self.diffusion = KDiffusion(net=self.unet, **diffusion_kwargs)
        else:
            raise ValueError(f"diffusion_type must be v or k, found {diffusion_type}")

    def forward(self, x: Tensor, **kwargs) -> Tensor:
        return self.diffusion(x, **kwargs)

    def sample(
        self,
        noise: Tensor,
        num_steps: int,
        sigma_schedule: Schedule,
        sampler: Sampler,
        **kwargs,
    ) -> Tensor:
        diffusion_sampler = DiffusionSampler(
            diffusion=self.diffusion,
            sampler=sampler,
            sigma_schedule=sigma_schedule,
            num_steps=num_steps,
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
        encoder_channels: int,
        bottleneck: Optional[Bottleneck] = None,
        encoder_num_blocks: Optional[Sequence[int]] = None,
        encoder_out_layers: int = 0,
        **kwargs,
    ):
        self.in_channels = in_channels
        encoder_num_blocks = default(encoder_num_blocks, num_blocks)
        assert_message = "The number of encoder_num_blocks must match encoder_depth"
        assert len(encoder_num_blocks) >= encoder_depth, assert_message

        multiencoder = MultiEncoder1d(
            in_channels=in_channels,
            channels=channels,
            patch_blocks=patch_blocks,
            patch_factor=patch_factor,
            num_layers=encoder_depth,
            num_layers_out=encoder_out_layers,
            latent_channels=encoder_channels,
            multipliers=multipliers,
            factors=factors,
            num_blocks=encoder_num_blocks,
            kernel_multiplier_downsample=kernel_multiplier_downsample,
            resnet_groups=resnet_groups,
        )

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
            context_channels=multiencoder.channels_list,
            **kwargs,
        )

        self.bottleneck = bottleneck
        self.multiencoder = multiencoder

    def forward(  # type: ignore
        self, x: Tensor, with_info: bool = False, **kwargs
    ) -> Union[Tensor, Tuple[Tensor, Any]]:
        if with_info:
            latent, info = self.encode(x, with_info=True)
        else:
            latent = self.encode(x)

        channels_list = self.multiencoder.decode(latent)
        loss = self.diffusion(x, channels_list=channels_list, **kwargs)
        return (loss, info) if with_info else loss

    def encode(
        self, x: Tensor, with_info: bool = False
    ) -> Union[Tensor, Tuple[Tensor, Any]]:
        latent = self.multiencoder.encode(x)
        latent = torch.tanh(latent)
        # Apply bottleneck if provided (e.g. quantization module)
        if exists(self.bottleneck):
            latent, info = self.bottleneck(latent)
            return (latent, info) if with_info else latent
        return latent

    def decode(self, latent: Tensor, **kwargs) -> Tensor:
        b, length = latent.shape[0], latent.shape[2] * self.multiencoder.factor
        # Compute noise by inferring shape from latent length
        noise = torch.randn(b, self.in_channels, length).to(latent)
        # Compute context form latent
        channels_list = self.multiencoder.decode(latent)
        default_kwargs = dict(channels_list=channels_list)
        # Decode by sampling while conditioning on latent channels
        return super().sample(noise, **{**default_kwargs, **kwargs})  # type: ignore


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
        resnet_groups=8,
        kernel_multiplier_downsample=2,
        use_nearest_upsample=False,
        use_skip_scale=True,
        use_context_time=True,
        use_magnitude_channels=False,
        diffusion_type="v",
        diffusion_sigma_distribution=VDistribution(),
    )


def get_default_sampling_kwargs():
    return dict(
        sigma_schedule=KarrasSchedule(sigma_min=0.0001, sigma_max=3.0, rho=9.0),
        sampler=AEulerSampler(),
    )


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
