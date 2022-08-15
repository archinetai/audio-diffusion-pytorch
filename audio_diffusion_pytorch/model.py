from typing import Optional, Sequence

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
from .modules import AutoEncoder1d, UNet1d
from .utils import exists


class Model1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        channels: int,
        patch_size: int,
        resnet_groups: int,
        kernel_multiplier_downsample: int,
        kernel_sizes_init: Sequence[int],
        multipliers: Sequence[int],
        factors: Sequence[int],
        num_blocks: Sequence[int],
        attentions: Sequence[bool],
        attention_heads: int,
        attention_features: int,
        attention_multiplier: int,
        use_learned_time_embedding: bool,
        use_nearest_upsample: bool,
        use_skip_scale: bool,
        use_attention_bottleneck: bool,
        diffusion_sigma_distribution: Distribution,
        diffusion_sigma_data: int,
        diffusion_dynamic_threshold: float,
        out_channels: Optional[int] = None,
        use_autoencoder: bool = False,
        autoencoder: Optional[AutoEncoder1d] = None,
        autoencoder_scale: float = 1.0,
    ):
        super().__init__()

        self.use_autoencoder = use_autoencoder

        if use_autoencoder:
            assert exists(autoencoder)
            self.autoencoder_scale = autoencoder_scale
            self.autoencoder = autoencoder

        self.unet = UNet1d(
            in_channels=in_channels,
            channels=channels,
            patch_size=patch_size,
            resnet_groups=resnet_groups,
            kernel_multiplier_downsample=kernel_multiplier_downsample,
            kernel_sizes_init=kernel_sizes_init,
            multipliers=multipliers,
            factors=factors,
            num_blocks=num_blocks,
            attentions=attentions,
            attention_heads=attention_heads,
            attention_features=attention_features,
            attention_multiplier=attention_multiplier,
            use_learned_time_embedding=use_learned_time_embedding,
            use_nearest_upsample=use_nearest_upsample,
            use_skip_scale=use_skip_scale,
            use_attention_bottleneck=use_attention_bottleneck,
            out_channels=out_channels,
        )

        self.diffusion = Diffusion(
            net=self.unet,
            sigma_distribution=diffusion_sigma_distribution,
            sigma_data=diffusion_sigma_data,
            dynamic_threshold=diffusion_dynamic_threshold,
        )

    def forward(self, x: Tensor) -> Tensor:
        if self.use_autoencoder:
            x = self.autoencoder_scale * self.autoencoder.encode(x)  # type: ignore
        return self.diffusion(x)

    def sample(
        self, noise: Tensor, num_steps: int, sigma_schedule: Schedule, sampler: Sampler
    ) -> Tensor:
        diffusion_sampler = DiffusionSampler(
            diffusion=self.diffusion,
            sampler=sampler,
            sigma_schedule=sigma_schedule,
            num_steps=num_steps,
        )
        x = diffusion_sampler(noise)

        if self.use_autoencoder:
            x = (1.0 / self.autoencoder_scale) * self.autoencoder.decode(x)

        return x


class AudioAutoEncoderModel(AutoEncoder1d):
    def __init__(self, *args, **kwargs):
        default_kwargs = dict(
            in_channels=1,
            bottleneck_channels=128,
            channels=128,
            patch_size=16,
            multipliers=[1, 1, 1, 1, 1],
            factors=[1, 4, 4, 4],
            num_blocks=[2, 2, 2, 2],
            resnet_groups=8,
            kernel_multiplier_downsample=2,
            loss_kl_weight=1e-8,
        )
        super().__init__(*args, **{**default_kwargs, **kwargs})


class AudioDiffusionModel(Model1d):
    def __init__(self, *args, **kwargs):
        default_kwargs = dict(
            channels=128,
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
            diffusion_sigma_distribution=LogNormalDistribution(mean=-3.0, std=1.0),
        )

        model_kwargs = None

        if "autoencoder" in kwargs:
            sigma_data = 0.2
            model_kwargs = dict(
                in_channels=128,
                patch_size=1,
                multipliers=[1, 4, 4, 4],
                factors=[2, 2, 2],
                num_blocks=[2, 2, 2],
                attentions=[True, True, True],
                diffusion_sigma_data=sigma_data,
                diffusion_dynamic_threshold=0.0,
                use_autoencoder=True,
                autoencoder_scale=sigma_data,
            )
        else:
            model_kwargs = dict(
                in_channels=1,
                patch_size=16,
                multipliers=[1, 2, 4, 4, 4, 4, 4],
                factors=[4, 4, 4, 2, 2, 2],
                num_blocks=[2, 2, 2, 2, 2, 2],
                attentions=[False, False, False, True, True, True],
                diffusion_sigma_data=0.1,
                diffusion_dynamic_threshold=0.95,
                use_autoencoder=False,
            )
        super().__init__(*args, **{**default_kwargs, **model_kwargs, **kwargs})

    def sample(self, *args, **kwargs):
        default_kwargs = dict(
            sigma_schedule=KarrasSchedule(sigma_min=0.0001, sigma_max=3.0, rho=9.0),
            sampler=ADPM2Sampler(rho=1.0),
        )
        return super().sample(*args, **{**default_kwargs, **kwargs})
