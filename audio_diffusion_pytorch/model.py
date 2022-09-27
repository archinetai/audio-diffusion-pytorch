from typing import Any, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor, nn

from .diffusion import (
    ADPM2Sampler,
    Diffusion,
    DiffusionSampler,
    Distribution,
    KarrasSchedule,
    LogNormalDistribution,
    Sampler,
    Schedule,
)
from .modules import Bottleneck, MultiEncoder1d, UNet1d, UNetConditional1d
from .utils import default, downsample, exists, to_list, upsample

"""
Diffusion Classes (generic for 1d data)
"""


class Model1d(nn.Module):
    def __init__(
        self,
        diffusion_sigma_distribution: Distribution,
        diffusion_sigma_data: int,
        diffusion_dynamic_threshold: float,
        use_classifier_free_guidance: bool = False,
        **kwargs
    ):
        super().__init__()

        UNet = UNetConditional1d if use_classifier_free_guidance else UNet1d

        self.unet = UNet(**kwargs)

        self.diffusion = Diffusion(
            net=self.unet,
            sigma_distribution=diffusion_sigma_distribution,
            sigma_data=diffusion_sigma_data,
            dynamic_threshold=diffusion_dynamic_threshold,
        )

    def forward(self, x: Tensor, **kwargs) -> Tensor:
        return self.diffusion(x, **kwargs)

    def sample(
        self,
        noise: Tensor,
        num_steps: int,
        sigma_schedule: Schedule,
        sampler: Sampler,
        **kwargs
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
        self, factor: Union[int, Sequence[int]], in_channels: int, *args, **kwargs
    ):
        self.factors = to_list(factor)
        default_kwargs = dict(
            in_channels=in_channels,
            context_channels=[in_channels],
        )
        super().__init__(*args, **{**default_kwargs, **kwargs})  # type: ignore

    def random_reupsample(self, x: Tensor) -> Tensor:
        batch_size, factors = x.shape[0], self.factors
        # Pick random factor for each batch element
        factor_batch_idx = torch.randint(0, len(factors), (batch_size,))
        x = x.clone()

        for i, factor in enumerate(factors):
            # Pick random items with current factor, skip if 0
            n = torch.count_nonzero(factor_batch_idx == i)
            if n > 0:
                waveforms = x[factor_batch_idx == i]
                # Downsample and reupsample items
                downsampled = downsample(waveforms, factor=factor)
                reupsampled = upsample(downsampled, factor=factor)
                # Save reupsampled version in place
                x[factor_batch_idx == i] = reupsampled
        return x

    def forward(self, x: Tensor, **kwargs) -> Tensor:
        channels = self.random_reupsample(x)
        return self.diffusion(x, channels_list=[channels], **kwargs)

    def sample(  # type: ignore
        self, undersampled: Tensor, factor: Optional[int] = None, *args, **kwargs
    ):
        # Either user provides factor or we pick the first
        factor = default(factor, self.factors[0])
        # Upsample channels
        channels = upsample(undersampled, factor=factor)
        noise = torch.randn_like(channels)
        default_kwargs = dict(channels_list=[channels])
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
        **kwargs
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
        diffusion_sigma_distribution=LogNormalDistribution(mean=-3.0, std=1.0),
        diffusion_sigma_data=0.1,
        diffusion_dynamic_threshold=0.0,
    )


def get_default_sampling_kwargs():
    return dict(
        sigma_schedule=KarrasSchedule(sigma_min=0.0001, sigma_max=3.0, rho=9.0),
        sampler=ADPM2Sampler(rho=1.0),
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
        **kwargs
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
