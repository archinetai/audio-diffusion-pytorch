import math
from typing import Callable, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from torch import Tensor

from .utils import default


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def cosine_beta_schedule(timesteps, s=0.008):
    """cosine schedule as proposed in https://openreview.net/forum?id=-NEXDKk8gZ"""
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


class Diffusion(nn.Module):
    def __init__(
        self,
        denoise_fn: nn.Module,
        *,
        num_timesteps=1000,
        loss_fn: Callable = F.l1_loss,
        loss_weight_gamma=0.5,  # https://openreview.net/pdf?id=-NEXDKk8gZ page 5
        loss_weight_k=1
    ):
        super().__init__()

        self.denoise_fn = denoise_fn
        self.loss_fn = loss_fn
        self.num_timesteps = num_timesteps

        self.register("betas", cosine_beta_schedule(timesteps=num_timesteps))
        self.register("alphas", 1.0 - self.betas)
        self.register("alphas_cumprod", torch.cumprod(self.alphas, axis=0))
        self.register("sqrt_alphas_cumprod", torch.sqrt(self.alphas_cumprod))
        self.register(
            "sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - self.alphas_cumprod)
        )
        self.register(
            "loss_weight",
            (loss_weight_k + self.alphas_cumprod / (1 - self.alphas_cumprod))
            ** -loss_weight_gamma,
        )

    def register(self, name: str, tensor: Tensor):
        self.register_buffer(name, tensor.to(torch.float32), persistent=False)

    def q_sample(self, x_0: Tensor, t: Tensor, noise):
        """Adds t steps of noise to x_0, i.e. samples x_t ~ q(.|x_0,t)"""
        return (
            extract(self.sqrt_alphas_cumprod, t, x_0.shape) * x_0
            + extract(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape) * noise
        )

    def forward(self, x_0: Tensor, noise: Tensor = None):
        """Addes t steps of noise to x_0 and denoises one step using denoise_fn, returns the loss."""
        b, device = x_0.shape[0], x_0.device
        # Number of noise steps for each batch item
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        # Pick noise and add to x_0
        noise = default(noise, lambda: torch.randn_like(x_0))
        x_t = self.q_sample(x_0=x_0, t=t, noise=noise)
        # Denoise x_0 with the model
        noise_pred = self.denoise_fn(x_t, t=t)
        # Compute loss
        loss = self.loss_fn(noise_pred, noise, reduction="none")
        loss = reduce(loss, "b ... -> b (...)", "mean")
        loss = loss * extract(self.loss_weight, t, loss.shape)
        return loss.mean()


class DiffusionSampler(nn.Module):
    def __init__(self, diffusion: Diffusion):
        super().__init__()

        self.denoise_fn = diffusion.denoise_fn
        self.num_timesteps = diffusion.num_timesteps

        betas = diffusion.betas
        alphas_cumprod = diffusion.alphas_cumprod
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], pad=(1, 0), value=1.0)

        self.register("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod))
        self.register(
            "sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1)
        )

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register("posterior_variance", posterior_variance)
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register(
            "posterior_log_variance_clipped",
            torch.log(posterior_variance.clamp(min=1e-20)),
        )
        self.register(
            "posterior_mean_coef1",
            betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )
        self.register(
            "posterior_mean_coef2",
            (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod),
        )

    def register(self, name: str, tensor: Tensor):
        self.register_buffer(name, tensor.to(torch.float32), persistent=False)

    def q_posterior_mean_variance(
        self, x_t: Tensor, x_0: Tensor, t: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Returns mean and variance of q(x_{t-1}|x_t,x_0,t)"""
        mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_0
            + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        var = extract(self.posterior_variance, t, x_t.shape)
        log_var_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return mean, var, log_var_clipped

    def p_mean_variance(self, x_t: Tensor, t: Tensor, clip_denoised: bool):
        """Returns mean and variance of p(x_{t-1}|x_t,t)"""
        noise = self.denoise_fn(x_t, t)
        x_0 = (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )
        if clip_denoised:
            x_0.clamp_(min=-1.0, max=1.0)
        # Since q(x_{t-1}|x_t,x_0,t) ≈ p(x_{t-1}|x_t,t) we can use q_posterior values
        return self.q_posterior_mean_variance(x_t, x_0, t)

    @torch.no_grad()
    def p_sample(self, x_t: Tensor, t: Tensor, clip_denoised: bool = True):
        """Samples a single denoised step sample, i.e. x_{t-1} ~ p(.|x_t,t)"""
        b = x_t.shape[0]
        mean, _, log_var = self.p_mean_variance(x_t, t, clip_denoised)
        # Sample from normal distribution
        noise = torch.randn_like(x_t)
        # no noise when t == 0
        shape = [b] + [1] * (x_t.ndim - 1)
        nonzero_mask = (1 - (t == 0).type(torch.float32)).view(*shape)
        return mean + nonzero_mask * (0.5 * log_var).exp() * noise

    def forward(self, x: Tensor) -> Tensor:
        """Sample loop for p, i.e. returns x_0 ~ p(.|x_T = x)"""
        b, device = x.shape[0], x.device

        for i in reversed(range(self.num_timesteps)):
            t = torch.full(size=(b,), fill_value=i, device=device, dtype=torch.int64)
            x = self.p_sample(x_t=x, t=t)

        return x
