""" Audio DDSP inspired by https://github.com/acids-ircam/RAVE """

from math import ceil, log2, pi, prod
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from scipy.optimize import fmin
from scipy.signal import firwin, kaiserord
from torch import Tensor
from torch.nn import functional as F

from .modules import Conv1d, ConvBlock1d


def reverse_half(x: Tensor) -> Tensor:
    mask = torch.ones_like(x)
    mask[..., 1::2, ::2] = -1
    return x * mask


def center_pad_next_pow_2(x: Tensor) -> Tensor:
    next_2 = 2 ** ceil(log2(x.shape[-1]))
    pad = next_2 - x.shape[-1]
    return F.pad(x, (pad // 2, pad // 2 + int(pad % 2)))


def get_qmf_bank(h: Tensor, nun_bands: int) -> Tensor:
    """
    Modulates an input protoype filter h into a bank of cosine modulated filters
    """
    k = torch.arange(nun_bands).reshape(-1, 1)
    N = h.shape[-1]
    t = torch.arange(-(N // 2), N // 2 + 1)

    p = (-1) ** k * pi / 4

    mod = torch.cos((2 * k + 1) * pi / (2 * nun_bands) * t + p)
    hk = 2 * h * mod

    return hk


def kaiser_filter(wc: float, attenuation: float) -> np.ndarray:
    """
    wc: Angular frequency
    attenuation: Attenuation (dB, positive)
    """
    N, beta = kaiserord(attenuation, wc / np.pi)
    N = 2 * (N // 2) + 1
    h = firwin(N, wc, window=("kaiser", beta), scale=False, nyq=np.pi)
    return h


def loss_wc(wc: float, attenuation: float, num_bands: int) -> np.ndarray:
    """
    Computes the objective described in https://ieeexplore.ieee.org/document/681427
    """
    h = kaiser_filter(wc, attenuation)
    g = np.convolve(h, h[::-1], "full")  # type: ignore
    start_idx = g.shape[-1] // 2
    stride = 2 * num_bands
    g = abs(g[start_idx::stride][1:])
    return np.max(g)


def get_prototype(attenuation: float, num_bands: int) -> np.ndarray:
    """
    Returns the corresponding lowpass filter
    """
    wc = fmin(lambda w: loss_wc(w, attenuation, num_bands), 1.0 / num_bands, disp=0)[0]
    return kaiser_filter(wc, attenuation)


def polyphase_forward(x: Tensor, hk: Tensor) -> Tensor:
    """
    x: [b, 1, t]
    hk: filter bank [m, t]
    """
    x = rearrange(x, "b c (t m) -> b (c m) t", m=hk.shape[0])
    hk = rearrange(hk, "c (t m) -> c m t", m=hk.shape[0])
    x = F.conv1d(x, hk, padding=hk.shape[-1] // 2)[..., :-1]
    return x


def polyphase_inverse(x: Tensor, hk: Tensor) -> Tensor:
    """
    x: signal to synthesize from [b, 1, t]
    hk: filter bank [m, t]
    """
    m = hk.shape[0]

    hk = hk.flip(-1)
    hk = rearrange(hk, "c (t m) -> m c t", m=m)  # polyphase

    pad = hk.shape[-1] // 2 + 1
    x = F.conv1d(x, hk, padding=int(pad))[..., :-1] * m

    x = x.flip(1)
    x = rearrange(x, "b (c m) t -> b c (t m)", m=m)
    start_idx = 2 * hk.shape[1]
    x = x[..., start_idx:]
    return x


def amp_to_impulse_response(amp: Tensor, target_size: int) -> Tensor:
    """
    Transforms frequecny amps to ir on the last dimension
    """
    # Set complex part to zero
    amp = torch.stack([amp, torch.zeros_like(amp)], -1)
    amp = torch.view_as_complex(amp)
    # Compute irrt i.e. fourier domain => real-valued amplitude domain
    amp = torch.fft.irfft(amp)
    #
    filter_size = amp.shape[-1]
    amp = torch.roll(amp, filter_size // 2, -1)

    win = torch.hann_window(filter_size, dtype=amp.dtype, device=amp.device)
    amp = amp * win

    amp = F.pad(amp, (0, int(target_size) - int(filter_size)))
    amp = torch.roll(amp, -filter_size // 2, -1)

    return amp


def fft_convolve(signal: Tensor, kernel: Tensor) -> Tensor:
    """
    Convolves signal by kernel on the last dimension
    """
    signal = F.pad(signal, (0, signal.shape[-1]))
    kernel = F.pad(kernel, (kernel.shape[-1], 0))

    output = torch.fft.irfft(torch.fft.rfft(signal) * torch.fft.rfft(kernel))
    start_idx = output.shape[-1] // 2
    output = output[..., start_idx:]

    return output


def scaled_simgoid(x: Tensor) -> Tensor:
    return 2 * torch.sigmoid(x) ** 2.3 + 1e-7


class PQMF(nn.Module):
    def __init__(self, attenuation: float, num_bands: int):
        super().__init__()
        self.num_bands = num_bands
        assert log2(num_bands).is_integer(), "num_bands must be a power of 2"

        h = get_prototype(attenuation, num_bands)
        hk = get_qmf_bank(torch.from_numpy(h).float(), num_bands)
        hk = center_pad_next_pow_2(hk)
        self.register_buffer("hk", hk)

    def forward(self, x):
        b, _, _ = x.shape
        x = rearrange(x, "b c t -> (b c) 1 t")
        x = polyphase_forward(x, self.hk)
        x = reverse_half(x)
        x = rearrange(x, "(b c) k t -> b (c k) t", b=b)
        return x

    def inverse(self, x):
        b, k = x.shape[0], self.num_bands
        x = rearrange(x, "b (c k) t -> (b c) k t", k=k)
        x = reverse_half(x)
        x = polyphase_inverse(x, self.hk)
        x = rearrange(x, "(b c) 1 t -> b c t", b=b)
        return x


class AudioProcessor(nn.Module):
    def __init__(
        self,
        in_channels: int,
        channels: int,
        pqmf_bands: int,
        pqmf_attenuation: float,
        noise_bands: int,
        noise_ratios: Sequence[int],
    ):
        super().__init__()

        pqmf_channels = in_channels * pqmf_bands
        amp_channels = [channels] * len(noise_ratios) + [pqmf_channels * noise_bands]

        self.noise_bands = noise_bands
        self.noise_multiplier = prod(noise_ratios)

        self.pqmf = PQMF(num_bands=pqmf_bands, attenuation=pqmf_attenuation)

        # Input processing

        self.to_in = Conv1d(
            in_channels=pqmf_channels, out_channels=channels, kernel_size=1
        )

        # Output processing

        self.to_wave = Conv1d(
            in_channels=channels, out_channels=pqmf_channels, kernel_size=1
        )

        self.to_loudness = Conv1d(
            in_channels=channels, out_channels=pqmf_channels, kernel_size=1
        )

        self.to_amp = nn.Sequential(
            *[
                ConvBlock1d(
                    in_channels=amp_channels[i],
                    out_channels=amp_channels[i + 1],
                    stride=noise_ratios[i],
                )
                for i in range(len(amp_channels) - 1)
            ]
        )

    def encode(self, x: Tensor) -> Tensor:
        x = self.pqmf(x)
        x = self.to_in(x)
        return x

    def decode(self, x: Tensor) -> Tensor:
        n = self.noise_bands
        wave, loudness, amp = self.to_wave(x), self.to_loudness(x), self.to_amp(x)

        # Convert computed amp to noise
        amp = rearrange(scaled_simgoid(amp - 5), "b (c n) t -> b t c n", n=n)
        impulse_response = amp_to_impulse_response(amp, self.noise_multiplier)
        noise = torch.rand_like(impulse_response) * 2 - 1
        noise = fft_convolve(noise, impulse_response)
        noise = rearrange(noise, "b t c n -> b c (t n)")

        x = torch.tanh(wave) * scaled_simgoid(loudness) + noise
        x = self.pqmf.inverse(x)
        return x
