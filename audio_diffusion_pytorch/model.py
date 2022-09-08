import random
from typing import Optional, Sequence, Union

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
from .modules import Encoder1d, ResnetBlock1d, UNet1d
from .utils import default, prod, to_list

""" Diffusion Classes (generic for 1d data) """


class Model1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        channels: int,
        patch_size: int,
        kernel_sizes_init: Sequence[int],
        multipliers: Sequence[int],
        factors: Sequence[int],
        num_blocks: Sequence[int],
        attentions: Sequence[bool],
        attention_heads: int,
        attention_features: int,
        attention_multiplier: int,
        use_attention_bottleneck: bool,
        resnet_groups: int,
        kernel_multiplier_downsample: int,
        use_nearest_upsample: bool,
        use_skip_scale: bool,
        diffusion_sigma_distribution: Distribution,
        diffusion_sigma_data: int,
        diffusion_dynamic_threshold: float,
        out_channels: Optional[int] = None,
        context_channels: Optional[Sequence[int]] = None,
    ):
        super().__init__()

        self.unet = UNet1d(
            in_channels=in_channels,
            channels=channels,
            patch_size=patch_size,
            kernel_sizes_init=kernel_sizes_init,
            multipliers=multipliers,
            factors=factors,
            num_blocks=num_blocks,
            attentions=attentions,
            attention_heads=attention_heads,
            attention_features=attention_features,
            attention_multiplier=attention_multiplier,
            use_attention_bottleneck=use_attention_bottleneck,
            resnet_groups=resnet_groups,
            kernel_multiplier_downsample=kernel_multiplier_downsample,
            use_nearest_upsample=use_nearest_upsample,
            use_skip_scale=use_skip_scale,
            out_channels=out_channels,
            context_channels=context_channels,
        )

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
        self.factor = to_list(factor)
        default_kwargs = dict(
            in_channels=in_channels,
            context_channels=[in_channels],
        )
        super().__init__(*args, **{**default_kwargs, **kwargs})  # type: ignore

    def forward(self, x: Tensor, factor: Optional[int] = None, **kwargs) -> Tensor:
        # Either user provides factor or we pick one at random
        factor = default(factor, random.choice(self.factor))
        # Downsample by picking every `factor` item
        downsampled = x[:, :, ::factor]
        # Upsample by interleaving to get context
        context = torch.repeat_interleave(downsampled, repeats=factor, dim=2)
        return self.diffusion(x, context=[context], **kwargs)

    def sample(  # type: ignore
        self, undersampled: Tensor, factor: Optional[int] = None, *args, **kwargs
    ):
        # Either user provides factor or we pick the first
        factor = default(factor, self.factor[0])
        # Upsample context by interleaving
        context = torch.repeat_interleave(undersampled, repeats=factor, dim=2)
        noise = torch.randn_like(context)
        default_kwargs = dict(context=[context])
        return super().sample(noise, **{**default_kwargs, **kwargs})  # type: ignore


class DiffusionAutoencoder1d(Model1d):
    def __init__(
        self,
        in_channels: int,
        channels: int,
        patch_size: int,
        kernel_sizes_init: Sequence[int],
        multipliers: Sequence[int],
        factors: Sequence[int],
        num_blocks: Sequence[int],
        resnet_groups: int,
        kernel_multiplier_downsample: int,
        encoder_depth: int,
        encoder_channels: int,
        context_channels: int,
        **kwargs
    ):
        super().__init__(
            in_channels=in_channels,
            channels=channels,
            patch_size=patch_size,
            kernel_sizes_init=kernel_sizes_init,
            multipliers=multipliers,
            factors=factors,
            num_blocks=num_blocks,
            resnet_groups=resnet_groups,
            kernel_multiplier_downsample=kernel_multiplier_downsample,
            context_channels=[0] * encoder_depth + [context_channels],
            **kwargs
        )

        self.in_channels = in_channels
        self.encoder_factor = patch_size * prod(factors[0:encoder_depth])

        self.encoder = Encoder1d(
            in_channels=in_channels,
            channels=channels,
            patch_size=patch_size,
            kernel_sizes_init=kernel_sizes_init,
            multipliers=multipliers,
            factors=factors,
            num_blocks=num_blocks,
            resnet_groups=resnet_groups,
            kernel_multiplier_downsample=kernel_multiplier_downsample,
            extract_channels=[0] * (encoder_depth - 1) + [encoder_channels],
        )

        self.to_context = ResnetBlock1d(
            in_channels=encoder_channels,
            out_channels=context_channels,
            num_groups=resnet_groups,
        )

    def forward(self, x: Tensor, **kwargs) -> Tensor:
        latent = self.encode(x)
        context = self.to_context(latent)
        return self.diffusion(x, context=[context], **kwargs)

    def encode(self, x: Tensor) -> Tensor:
        x = self.encoder(x)[-1]
        latent = torch.tanh(x)
        return latent

    def decode(self, latent: Tensor, **kwargs) -> Tensor:
        b, length = latent.shape[0], latent.shape[2] * self.encoder_factor
        # Compute noise by inferring shape from latent length
        noise = torch.randn(b, self.in_channels, length).to(latent)
        # Compute context form latent
        context = self.to_context(latent)
        default_kwargs = dict(context=[context])
        # Decode by sampling while conditioning on latent context
        return super().sample(noise, **{**default_kwargs, **kwargs})  # type: ignore


""" Audio Diffusion Classes (specific for 1d audio data) """


class AudioDiffusionModel(Model1d):
    def __init__(self, *args, **kwargs):
        default_kwargs = dict(
            channels=128,
            patch_size=16,
            kernel_sizes_init=[1, 3, 7],
            multipliers=[1, 2, 4, 4, 4, 4, 4],
            factors=[4, 4, 4, 2, 2, 2],
            num_blocks=[2, 2, 2, 2, 2, 2],
            attentions=[False, False, False, True, True, True],
            attention_heads=8,
            attention_features=64,
            attention_multiplier=2,
            use_attention_bottleneck=True,
            resnet_groups=8,
            kernel_multiplier_downsample=2,
            use_nearest_upsample=False,
            use_skip_scale=True,
            diffusion_sigma_distribution=LogNormalDistribution(mean=-3.0, std=1.0),
            diffusion_sigma_data=0.1,
            diffusion_dynamic_threshold=0.0,
        )

        super().__init__(*args, **{**default_kwargs, **kwargs})

    def sample(self, *args, **kwargs):
        default_kwargs = dict(
            sigma_schedule=KarrasSchedule(sigma_min=0.0001, sigma_max=3.0, rho=9.0),
            sampler=ADPM2Sampler(rho=1.0),
        )
        return super().sample(*args, **{**default_kwargs, **kwargs})


class AudioDiffusionUpsampler(DiffusionUpsampler1d):
    def __init__(self, in_channels: int, *args, **kwargs):
        default_kwargs = dict(
            in_channels=in_channels,
            channels=128,
            patch_size=16,
            kernel_sizes_init=[1, 3, 7],
            multipliers=[1, 2, 4, 4, 4, 4, 4],
            factors=[4, 4, 4, 2, 2, 2],
            num_blocks=[2, 2, 2, 2, 2, 2],
            attentions=[False, False, False, True, True, True],
            attention_heads=8,
            attention_features=64,
            attention_multiplier=2,
            use_attention_bottleneck=True,
            resnet_groups=8,
            kernel_multiplier_downsample=2,
            use_nearest_upsample=False,
            use_skip_scale=True,
            diffusion_sigma_distribution=LogNormalDistribution(mean=-3.0, std=1.0),
            diffusion_sigma_data=0.1,
            diffusion_dynamic_threshold=0.0,
            context_channels=[in_channels],
        )

        super().__init__(*args, **{**default_kwargs, **kwargs})  # type: ignore

    def sample(self, *args, **kwargs):
        default_kwargs = dict(
            sigma_schedule=KarrasSchedule(sigma_min=0.0001, sigma_max=3.0, rho=9.0),
            sampler=ADPM2Sampler(rho=1.0),
        )
        return super().sample(*args, **{**default_kwargs, **kwargs})


class AudioDiffusionAutoencoder(DiffusionAutoencoder1d):
    def __init__(self, *args, **kwargs):
        default_kwargs = dict(
            channels=128,
            patch_size=16,
            kernel_sizes_init=[1, 3, 7],
            multipliers=[1, 2, 4, 4, 4, 4, 4],
            factors=[4, 4, 4, 2, 2, 2],
            num_blocks=[2, 2, 2, 2, 2, 2],
            attentions=[False, False, False, True, True, True],
            attention_heads=8,
            attention_features=64,
            attention_multiplier=2,
            use_attention_bottleneck=True,
            resnet_groups=8,
            kernel_multiplier_downsample=2,
            use_nearest_upsample=False,
            use_skip_scale=True,
            encoder_depth=4,
            encoder_channels=32,
            context_channels=512,
            diffusion_sigma_distribution=LogNormalDistribution(mean=-3.0, std=1.0),
            diffusion_sigma_data=0.1,
            diffusion_dynamic_threshold=0.0,
        )

        super().__init__(*args, **{**default_kwargs, **kwargs})  # type: ignore

    def decode(self, *args, **kwargs):
        default_kwargs = dict(
            sigma_schedule=KarrasSchedule(sigma_min=0.0001, sigma_max=3.0, rho=9.0),
            sampler=ADPM2Sampler(rho=1.0),
        )
        return super().decode(*args, **{**default_kwargs, **kwargs})
