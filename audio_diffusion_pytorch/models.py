from typing import Any, Callable, Optional, Sequence, Tuple, Union

import torch
from audio_encoders_pytorch import Encoder1d
from torch import Generator, Tensor, nn

from .components import AppendChannelsPlugin, UNetV0
from .diffusion import ARVDiffusion, ARVSampler, VDiffusion, VSampler
from .utils import closest_power_2, downsample, groupby, randn_like, upsample


class DiffusionModel(nn.Module):
    def __init__(
        self,
        dim: int = 1,
        net_t: Callable = UNetV0,
        diffusion_t: Callable = VDiffusion,
        sampler_t: Callable = VSampler,
        **kwargs,
    ):
        super().__init__()
        diffusion_kwargs, kwargs = groupby("diffusion_", kwargs)
        sampler_kwargs, kwargs = groupby("sampler_", kwargs)

        self.net = net_t(dim=dim, **kwargs)
        self.diffusion = diffusion_t(net=self.net, **diffusion_kwargs)
        self.sampler = sampler_t(net=self.net, **sampler_kwargs)

    def forward(self, *args, **kwargs) -> Tensor:
        return self.diffusion(*args, **kwargs)

    def sample(self, *args, **kwargs) -> Tensor:
        return self.sampler(*args, **kwargs)


class DiffusionAE(DiffusionModel):
    """Diffusion Auto Encoder"""

    def __init__(
        self,
        in_channels: int,
        channels: Sequence[int],
        encoder: Encoder1d,
        inject_depth: int,
        **kwargs,
    ):
        context_channels = [0] * len(channels)
        context_channels[inject_depth] = encoder.out_channels
        super().__init__(
            in_channels=in_channels,
            channels=channels,
            context_channels=context_channels,
            **kwargs,
        )
        self.in_channels = in_channels
        self.encoder = encoder
        self.inject_depth = inject_depth

    def forward(  # type: ignore
        self, x: Tensor, with_info: bool = False, **kwargs
    ) -> Union[Tensor, Tuple[Tensor, Any]]:
        latent, info = self.encode(x, with_info=True)
        channels = [None] * self.inject_depth + [latent]
        loss = super().forward(x, channels=channels, **kwargs)
        return (loss, info) if with_info else loss

    def encode(self, *args, **kwargs):
        return self.encoder(*args, **kwargs)

    def decode(
        self, latent: Tensor, generator: Optional[Generator] = None, **kwargs
    ) -> Tensor:
        b = latent.shape[0]
        length = closest_power_2(latent.shape[2] * self.encoder.downsample_factor)
        # Compute noise by inferring shape from latent length
        noise = torch.randn(
            (b, self.in_channels, length),
            device=latent.device,
            dtype=latent.dtype,
            generator=generator,
        )
        # Compute context from latent
        channels = [None] * self.inject_depth + [latent]  # type: ignore
        default_kwargs = dict(channels=channels)
        # Decode by sampling while conditioning on latent channels
        return super().sample(noise, **{**default_kwargs, **kwargs})


class DiffusionUpsampler(DiffusionModel):
    def __init__(
        self,
        in_channels: int,
        upsample_factor: int,
        net_t: Callable = UNetV0,
        **kwargs,
    ):
        self.upsample_factor = upsample_factor
        super().__init__(
            net_t=AppendChannelsPlugin(net_t, channels=in_channels),
            in_channels=in_channels,
            **kwargs,
        )

    def reupsample(self, x: Tensor) -> Tensor:
        x = x.clone()
        x = downsample(x, factor=self.upsample_factor)
        x = upsample(x, factor=self.upsample_factor)
        return x

    def forward(self, x: Tensor, *args, **kwargs) -> Tensor:  # type: ignore
        reupsampled = self.reupsample(x)
        return super().forward(x, *args, append_channels=reupsampled, **kwargs)

    def sample(  # type: ignore
        self, downsampled: Tensor, generator: Optional[Generator] = None, **kwargs
    ) -> Tensor:
        reupsampled = upsample(downsampled, factor=self.upsample_factor)
        noise = randn_like(reupsampled, generator=generator)
        return super().sample(noise, append_channels=reupsampled, **kwargs)


class DiffusionAR(DiffusionModel):
    def __init__(
        self,
        in_channels: int,
        length: int,
        num_splits: int,
        diffusion_t: Callable = ARVDiffusion,
        sampler_t: Callable = ARVSampler,
        **kwargs,
    ):
        super().__init__(
            in_channels=in_channels + 1,
            out_channels=in_channels,
            diffusion_t=diffusion_t,
            diffusion_length=length,
            diffusion_num_splits=num_splits,
            sampler_t=sampler_t,
            sampler_in_channels=in_channels,
            sampler_length=length,
            sampler_num_splits=num_splits,
            use_time_conditioning=False,
            use_modulation=False,
            **kwargs,
        )
