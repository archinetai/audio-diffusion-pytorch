from math import log, sqrt
from typing import Any, Callable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from torch import Tensor

from ..utils import default, exists

""" Samplers and sigma schedules """


class SigmaSampler:
    def __call__(self, num_samples: int, device: str):
        raise NotImplementedError()


class LogNormalSampler(SigmaSampler):
    def __init__(self, mean: float, std: float):
        self.mean = mean
        self.std = std

    def __call__(self, num_samples, device: Any = "cpu"):
        normal = self.mean + self.std * torch.randn((num_samples,), device=device)
        return normal.exp()


class SigmaSchedule(nn.Module):
    """Interface used by different sampling sigma schedules"""

    def forward(self, num_steps: int, device: Any) -> Tensor:
        raise NotImplementedError()


class KerrasSchedule(SigmaSchedule):
    """https://arxiv.org/abs/2206.00364 eq. (5)"""

    def __init__(self, sigma_min: float, sigma_max: float, rho: float = 7.0):
        super().__init__()
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho

    def forward(self, num_steps: int, device: Any) -> Tensor:
        rho_inv = 1.0 / self.rho
        steps = torch.arange(num_steps, device=device, dtype=torch.float32)
        sigmas = (
            self.sigma_max ** rho_inv
            + (steps / (num_steps - 1))
            * (self.sigma_min ** rho_inv - self.sigma_max ** rho_inv)
        ) ** self.rho
        sigmas = F.pad(sigmas, pad=(0, 1), value=0.0)
        return sigmas


class Diffusion(nn.Module):
    """Elucidated Diffusion: https://arxiv.org/abs/2206.00364"""

    def __init__(
        self,
        net: nn.Module,
        *,
        sigma_sampler: Sampler,
        sigma_data: float,  # data distribution standard deviation
    ):
        super().__init__()

        self.net = net
        self.sigma_data = sigma_data
        self.sigma_sampler = sigma_sampler

    def c_skip(self, sigmas: Tensor) -> Tensor:
        return (self.sigma_data ** 2) / (sigmas ** 2 + self.sigma_data ** 2)

    def c_out(self, sigmas: Tensor) -> Tensor:
        return sigmas * self.sigma_data * (self.sigma_data ** 2 + sigmas ** 2) ** -0.5

    def c_in(self, sigmas: Tensor) -> Tensor:
        return 1 * (sigmas ** 2 + self.sigma_data ** 2) ** -0.5

    def c_noise(self, sigmas: Tensor) -> Tensor:
        return torch.log(sigmas) * 0.25

    def denoise_fn(
        self,
        x_noisy: Tensor,
        sigmas: Tensor = None,
        sigma: float = None,
        clamp: bool = False,
    ) -> Tensor:
        batch, device = x_noisy.shape[0], x_noisy.device

        assert exists(sigmas) or exists(
            sigma
        ), "Either sigmas or sigma must be provided"

        # If sigma provided use the same for all batch items (used for sampling)
        if exists(sigma):
            sigmas = torch.full(size=(batch,), fill_value=sigma).to(device)

        sigmas_padded = rearrange(sigmas, "b -> b 1 1")

        # Predict network output and add skip connection
        x_pred = self.net(self.c_in(sigmas_padded) * x_noisy, self.c_noise(sigmas))
        x_denoised = (
            self.c_skip(sigmas_padded) * x_noisy + self.c_out(sigmas_padded) * x_pred
        )
        x_denoised = x_denoised.clamp(-1.0, 1) if clamp else x_denoised

        return x_denoised

    def loss_weight(self, sigmas: Tensor) -> Tensor:
        # Computes weight depending on data distribution
        return (sigmas ** 2 + self.sigma_data ** 2) * (sigmas * self.sigma_data) ** -2

    def forward(self, x: Tensor, noise: Tensor = None) -> Tensor:
        batch, device = x.shape[0], x.device

        # Sample amount of noise to add for each batch element
        sigmas = self.sigma_sampler(num_samples=batch, device=device)
        sigmas_padded = rearrange(sigmas, "b -> b 1 1")

        # Add noise to input
        noise = default(noise, lambda: torch.randn_like(x))
        x_noisy = x + sigmas_padded * noise

        # Compute denoised value using
        x_denoised = self.denoise_fn(x_noisy, sigmas=sigmas)

        # Compute weighted loss
        losses = F.mse_loss(x_denoised, x, reduction="none")
        losses = reduce(losses, "b ... -> b", "mean")
        losses = losses * self.loss_weight(sigmas)
        loss = losses.mean()

        return loss


class DiffusionSampler(nn.Module):
    def __init__(
        self,
        diffusion: Diffusion,
        *,
        num_steps: int,
        sigma_schedule: SigmaSchedule,
        s_tmin: float = 0,
        s_tmax: float = float("inf"),
        s_churn: float = 0.0,
        s_noise: float = 1.0,
    ):
        super().__init__()
        self.denoise_fn = diffusion.denoise_fn
        self.num_steps = num_steps
        self.sigma_schedule = sigma_schedule
        self.s_tmin = s_tmin
        self.s_tmax = s_tmax
        self.s_noise = s_noise
        self.gamma = min(s_churn / num_steps, sqrt(2) - 1)

    def step(self, x: Tensor, sigma: Tensor, sigma_next: Tensor, clamp: bool = True):
        """Algorithm 2 (step)"""
        gamma = self.gamma if self.s_tmin <= sigma <= self.s_tmax else 0.0
        # Select temporarily increased noise level
        sigma_hat = sigma + gamma * sigma
        # Add noise to move from sigma to sigma_hat
        epsilon = self.s_noise * torch.randn_like(x)
        x_hat = x + sqrt(sigma_hat ** 2 - sigma ** 2) * epsilon
        # Evaluate ∂x/∂sigma at sigma_hat
        d = (x_hat - self.denoise_fn(x_hat, sigma=sigma_hat, clamp=clamp)) / sigma_hat
        # Take euler step from sigma_hat to sigma_next
        x_next = x_hat + (sigma_next - sigma_hat) * d
        # Second order correction
        if sigma_next != 0:
            model_out_next = self.denoise_fn(x_next, sigma=sigma_next, clamp=clamp)
            d_prime = (x_next - model_out_next) / sigma_next
            x_next = x_hat + 0.5 * (sigma - sigma_hat) * (d + d_prime)
        return x_next

    def forward(self, x: Tensor, num_steps: int = None) -> Tensor:
        b, device = x.shape[0], x.device
        num_steps = default(num_steps, self.num_steps)
        # Compute sigmas using schedule
        sigmas = self.sigma_schedule(num_steps, device)
        # Sample from first sigma distribution
        x = sigmas[0] * x
        # Denoise x
        for i in range(num_steps - 1):
            x = self.step(x, sigma=sigmas[i], sigma_next=sigmas[i + 1])

        x = x.clamp(-1.0, 1.0)
        return x
