from math import pi
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import Tensor
from tqdm import tqdm

""" Distributions """


class Distribution:
    """Interface used by different distributions"""

    def __call__(self, num_samples: int, device: torch.device):
        raise NotImplementedError()


class UniformDistribution(Distribution):
    def __init__(self, vmin: float = 0.0, vmax: float = 1.0):
        super().__init__()
        self.vmin, self.vmax = vmin, vmax

    def __call__(self, num_samples: int, device: torch.device = torch.device("cpu")):
        vmax, vmin = self.vmax, self.vmin
        return (vmax - vmin) * torch.rand(num_samples, device=device) + vmin


""" Diffusion Methods """


def pad_dims(x: Tensor, ndim: int) -> Tensor:
    # Pads additional ndims to the right of the tensor
    return x.view(*x.shape, *((1,) * ndim))


def clip(x: Tensor, dynamic_threshold: float = 0.0):
    if dynamic_threshold == 0.0:
        return x.clamp(-1.0, 1.0)
    else:
        # Dynamic thresholding
        # Find dynamic threshold quantile for each batch
        x_flat = rearrange(x, "b ... -> b (...)")
        scale = torch.quantile(x_flat.abs(), dynamic_threshold, dim=-1)
        # Clamp to a min of 1.0
        scale.clamp_(min=1.0)
        # Clamp all values and scale
        scale = pad_dims(scale, ndim=x.ndim - scale.ndim)
        x = x.clamp(-scale, scale) / scale
        return x


def extend_dim(x: Tensor, dim: int):
    # e.g. if dim = 4: shape [b] => [b, 1, 1, 1],
    return x.view(*x.shape + (1,) * (dim - x.ndim))


class Diffusion(nn.Module):
    """Interface used by different diffusion methods"""

    pass


class VDiffusion(Diffusion):
    def __init__(
        self, net: nn.Module, sigma_distribution: Distribution = UniformDistribution()
    ):
        super().__init__()
        self.net = net
        self.sigma_distribution = sigma_distribution

    def get_alpha_beta(self, sigmas: Tensor) -> Tuple[Tensor, Tensor]:
        angle = sigmas * pi / 2
        alpha, beta = torch.cos(angle), torch.sin(angle)
        return alpha, beta

    def forward(self, x: Tensor, **kwargs) -> Tensor:  # type: ignore
        batch_size, device = x.shape[0], x.device
        # Sample amount of noise to add for each batch element
        sigmas = self.sigma_distribution(num_samples=batch_size, device=device)
        sigmas_batch = extend_dim(sigmas, dim=x.ndim)
        # Get noise
        noise = torch.randn_like(x)
        # Combine input and noise weighted by half-circle
        alphas, betas = self.get_alpha_beta(sigmas_batch)
        x_noisy = alphas * x + betas * noise
        v_target = alphas * noise - betas * x
        # Predict velocity and return loss
        v_pred = self.net(x_noisy, sigmas, **kwargs)
        return F.mse_loss(v_pred, v_target)


class ARVDiffusion(Diffusion):
    def __init__(self, net: nn.Module, length: int, num_splits: int):
        super().__init__()
        assert length % num_splits == 0, "length must be divisible by num_splits"
        self.net = net
        self.length = length
        self.num_splits = num_splits
        self.split_length = length // num_splits

    def get_alpha_beta(self, sigmas: Tensor) -> Tuple[Tensor, Tensor]:
        angle = sigmas * pi / 2
        alpha, beta = torch.cos(angle), torch.sin(angle)
        return alpha, beta

    def forward(self, x: Tensor, **kwargs) -> Tensor:
        """Returns diffusion loss of v-objective with different noises per split"""
        b, _, t, device, dtype = *x.shape, x.device, x.dtype
        assert t == self.length, "input length must match length"
        # Sample amount of noise to add for each split
        sigmas = torch.rand((b, 1, self.num_splits), device=device, dtype=dtype)
        sigmas = repeat(sigmas, "b 1 n -> b 1 (n l)", l=self.split_length)
        # Get noise
        noise = torch.randn_like(x)
        # Combine input and noise weighted by half-circle
        alphas, betas = self.get_alpha_beta(sigmas)
        x_noisy = alphas * x + betas * noise
        v_target = alphas * noise - betas * x
        # Sigmas will be provided as additional channel
        channels = torch.cat([x_noisy, sigmas], dim=1)
        # Predict velocity and return loss
        v_pred = self.net(channels, **kwargs)
        return F.mse_loss(v_pred, v_target)


""" Schedules """


class Schedule(nn.Module):
    """Interface used by different sampling schedules"""

    def forward(self, num_steps: int, device: torch.device) -> Tensor:
        raise NotImplementedError()


class LinearSchedule(Schedule):
    def __init__(self, start: float = 1.0, end: float = 0.0):
        super().__init__()
        self.start, self.end = start, end

    def forward(self, num_steps: int, device: Any) -> Tensor:
        return torch.linspace(self.start, self.end, num_steps, device=device)


""" Samplers """


class Sampler(nn.Module):
    pass


class VSampler(Sampler):

    diffusion_types = [VDiffusion]

    def __init__(self, net: nn.Module, schedule: Schedule = LinearSchedule()):
        super().__init__()
        self.net = net
        self.schedule = schedule

    def get_alpha_beta(self, sigmas: Tensor) -> Tuple[Tensor, Tensor]:
        angle = sigmas * pi / 2
        alpha, beta = torch.cos(angle), torch.sin(angle)
        return alpha, beta

    def forward(  # type: ignore
        self, x_noisy: Tensor, num_steps: int, show_progress: bool = False, **kwargs
    ) -> Tensor:
        b = x_noisy.shape[0]
        sigmas = self.schedule(num_steps + 1, device=x_noisy.device)
        sigmas = repeat(sigmas, "i -> i b", b=b)
        sigmas_batch = extend_dim(sigmas, dim=x_noisy.ndim + 1)
        alphas, betas = self.get_alpha_beta(sigmas_batch)
        progress_bar = tqdm(range(num_steps), disable=not show_progress)

        for i in progress_bar:
            v_pred = self.net(x_noisy, sigmas[i], **kwargs)
            x_pred = alphas[i] * x_noisy - betas[i] * v_pred
            noise_pred = betas[i] * x_noisy + alphas[i] * v_pred
            x_noisy = alphas[i + 1] * x_pred + betas[i + 1] * noise_pred
            progress_bar.set_description(f"Sampling (noise={sigmas[i+1,0]:.2f})")

        return x_noisy


class ARVSampler(Sampler):
    def __init__(self, net: nn.Module, in_channels: int, length: int, num_splits: int):
        super().__init__()
        assert length % num_splits == 0, "length must be divisible by num_splits"
        self.length = length
        self.in_channels = in_channels
        self.num_splits = num_splits
        self.split_length = length // num_splits
        self.net = net

    @property
    def device(self):
        return next(self.net.parameters()).device

    def get_alpha_beta(self, sigmas: Tensor) -> Tuple[Tensor, Tensor]:
        angle = sigmas * pi / 2
        alpha = torch.cos(angle)
        beta = torch.sin(angle)
        return alpha, beta

    def get_sigmas_ladder(self, num_items: int, num_steps_per_split: int) -> Tensor:
        b, n, l, i = num_items, self.num_splits, self.split_length, num_steps_per_split
        n_half = n // 2  # Only half ladder, rest is zero, to leave some context
        sigmas = torch.linspace(1, 0, i * n_half, device=self.device)
        sigmas = repeat(sigmas, "(n i) -> i b 1 (n l)", b=b, l=l, n=n_half)
        sigmas = torch.flip(sigmas, dims=[-1])  # Lowest noise level first
        sigmas = F.pad(sigmas, pad=[0, 0, 0, 0, 0, 0, 0, 1])  # Add index i+1
        sigmas[-1, :, :, l:] = sigmas[0, :, :, :-l]  # Loop back at index i+1
        return torch.cat([torch.zeros_like(sigmas), sigmas], dim=-1)

    def sample_loop(
        self, current: Tensor, sigmas: Tensor, show_progress: bool = False, **kwargs
    ) -> Tensor:
        num_steps = sigmas.shape[0] - 1
        alphas, betas = self.get_alpha_beta(sigmas)
        progress_bar = tqdm(range(num_steps), disable=not show_progress)

        for i in progress_bar:
            channels = torch.cat([current, sigmas[i]], dim=1)
            v_pred = self.net(channels, **kwargs)
            x_pred = alphas[i] * current - betas[i] * v_pred
            noise_pred = betas[i] * current + alphas[i] * v_pred
            current = alphas[i + 1] * x_pred + betas[i + 1] * noise_pred
            progress_bar.set_description(f"Sampling (noise={sigmas[i+1,0,0,0]:.2f})")

        return current

    def sample_start(self, num_items: int, num_steps: int, **kwargs) -> Tensor:
        b, c, t = num_items, self.in_channels, self.length
        # Same sigma schedule over all chunks
        sigmas = torch.linspace(1, 0, num_steps + 1, device=self.device)
        sigmas = repeat(sigmas, "i -> i b 1 t", b=b, t=t)
        noise = torch.randn((b, c, t), device=self.device) * sigmas[0]
        # Sample start
        return self.sample_loop(current=noise, sigmas=sigmas, **kwargs)

    def forward(
        self,
        num_items: int,
        num_chunks: int,
        num_steps: int,
        start: Optional[Tensor] = None,
        show_progress: bool = False,
        **kwargs,
    ) -> Tensor:
        assert_message = f"required at least {self.num_splits} chunks"
        assert num_chunks >= self.num_splits, assert_message

        # Sample initial chunks
        start = self.sample_start(num_items=num_items, num_steps=num_steps, **kwargs)
        # Return start if only num_splits chunks
        if num_chunks == self.num_splits:
            return start

        # Get sigmas for autoregressive ladder
        b, n = num_items, self.num_splits
        assert num_steps >= n, "num_steps must be greater than num_splits"
        sigmas = self.get_sigmas_ladder(
            num_items=b,
            num_steps_per_split=num_steps // self.num_splits,
        )
        alphas, betas = self.get_alpha_beta(sigmas)

        # Noise start to match ladder and set starting chunks
        start_noise = alphas[0] * start + betas[0] * torch.randn_like(start)
        chunks = list(start_noise.chunk(chunks=n, dim=-1))

        # Loop over ladder shifts
        num_shifts = num_chunks  # - self.num_splits
        progress_bar = tqdm(range(num_shifts), disable=not show_progress)

        for j in progress_bar:
            # Decrease ladder noise of last n chunks
            updated = self.sample_loop(
                current=torch.cat(chunks[-n:], dim=-1), sigmas=sigmas, **kwargs
            )
            # Update chunks
            chunks[-n:] = list(updated.chunk(chunks=n, dim=-1))
            # Add fresh noise chunk
            shape = (b, self.in_channels, self.split_length)
            chunks += [torch.randn(shape, device=self.device)]

        return torch.cat(chunks[:num_chunks], dim=-1)
