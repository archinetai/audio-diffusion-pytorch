from math import pi
from typing import Any, List, Tuple, Type

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
    def __call__(self, num_samples: int, device: torch.device = torch.device("cpu")):
        return torch.rand(num_samples, device=device)


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


class Diffusion(nn.Module):
    """Interface used by different diffusion methods"""

    def forward(self, *args, **kwargs) -> Tensor:
        raise NotImplementedError("Diffusion class missing forward function")


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
        sigmas_batch = rearrange(sigmas, "b -> b 1 1")
        # Get noise
        noise = torch.randn_like(x)
        # Combine input and noise weighted by half-circle
        alphas, betas = self.get_alpha_beta(sigmas_batch)
        x_noisy = alphas * x + betas * noise
        v_target = alphas * noise - betas * x
        # Predict velocity and return loss
        v_pred = self.net(x_noisy, sigmas, **kwargs)
        return F.mse_loss(v_pred, v_target)


""" Schedules """


class Schedule(nn.Module):
    """Interface used by different sampling schedules"""

    def forward(self, num_steps: int, device: torch.device) -> Tensor:
        raise NotImplementedError()


class LinearSchedule(Schedule):
    def forward(self, num_steps: int, device: Any) -> Tensor:
        return torch.linspace(1.0, 0.0, num_steps, device=device)


""" Samplers """


class Sampler(nn.Module):
    """Interface used by different samplers"""

    diffusion_types: List[Type] = []

    def forward(*args, **kwargs) -> Tensor:
        raise NotImplementedError()


class VSampler(Sampler):

    diffusion_types = [VDiffusion]

    def __init__(self, net: nn.Module, schedule: Schedule = LinearSchedule()):
        super().__init__()
        self.net = net
        self.schedule = schedule

    @property
    def device(self):
        return next(self.net.parameters()).device

    def get_alpha_beta(self, sigmas: Tensor) -> Tuple[Tensor, Tensor]:
        angle = sigmas * pi / 2
        alpha = torch.cos(angle)
        beta = torch.sin(angle)
        return alpha, beta

    def forward(  # type: ignore
        self, noise: Tensor, num_steps: int, show_progress: bool = False, **kwargs
    ) -> Tensor:
        b = noise.shape[0]
        sigmas = self.schedule(num_steps + 1, device=self.device)
        sigmas = repeat(sigmas, "i -> i b", b=b)
        sigmas_batch = rearrange(sigmas, "i b -> i b 1 1")
        alphas, betas = self.get_alpha_beta(sigmas_batch)
        x_noisy = noise * sigmas_batch[0]
        progress_bar = tqdm(range(num_steps), disable=not show_progress)

        for i in progress_bar:
            v_pred = self.net(x_noisy, sigmas[i], **kwargs)
            x_pred = alphas[i] * x_noisy - betas[i] * v_pred
            noise_pred = betas[i] * x_noisy + alphas[i] * v_pred
            x_noisy = alphas[i + 1] * x_pred + betas[i + 1] * noise_pred
            progress_bar.set_description(f"Sampling (noise={sigmas[i+1,0]:.2f})")

        return x_noisy
