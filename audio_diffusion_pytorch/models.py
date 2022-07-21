from math import log, pi
from typing import Any, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange
from einops_exts import rearrange_many, repeat_many
from einops_exts.torch import EinopsToAndFrom
from torch import Tensor, einsum
from torch.nn import functional as F

from .utils import default, exists


def Conv1d(*args, **kwargs) -> nn.Module:
    return nn.Conv1d(*args, **kwargs)


def ConvTranspose1d(*args, **kwargs) -> nn.Module:
    return nn.ConvTranspose1d(*args, **kwargs)


class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        half_dim = self.dim // 2
        factor = log(10000) / (half_dim - 1)
        embedding = torch.exp(torch.arange(half_dim, device=x.device) * -factor)
        embedding = rearrange(x, "i -> i 1") * rearrange(embedding, "j -> 1 j")
        return torch.cat((embedding.sin(), embedding.cos()), dim=-1)


class LearnedPositionalEmbedding(nn.Module):
    """Used for continuous time"""

    def __init__(self, dim: int):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim))

    def forward(self, x: Tensor) -> Tensor:
        x = rearrange(x, "b -> b 1")
        freqs = x * rearrange(self.weights, "d -> 1 d") * 2 * pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        fouriered = torch.cat((x, fouriered), dim=-1)
        return fouriered


def TimePositionalEmbedding(
    dim: int, out_features: int, use_learned: bool
) -> nn.Module:
    return nn.Sequential(
        LearnedPositionalEmbedding(dim)
        if use_learned
        else SinusoidalPositionalEmbedding(dim),
        nn.Linear(
            in_features=dim + 1 if use_learned else dim, out_features=out_features
        ),
    )


def Downsample1d(
    in_channels: int,
    out_channels: int,
    factor: int,
    kernel_multiplier: int,
) -> nn.Module:
    assert kernel_multiplier % 2 == 0, "Kernel multiplier must be even"

    return Conv1d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=factor * kernel_multiplier + 1,
        stride=factor,
        padding=factor * (kernel_multiplier // 2),
        groups=in_channels // 4,
    )


def Upsample1d(
    in_channels: int, out_channels: int, factor: int, use_nearest: bool = False
) -> nn.Module:

    if factor == 1:
        return Conv1d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1
        )

    if use_nearest:
        return nn.Sequential(
            nn.Upsample(scale_factor=factor, mode="nearest"),
            Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
            ),
        )
    else:
        return ConvTranspose1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=factor * 2,
            stride=factor,
            padding=factor // 2 + factor % 2,
            output_padding=factor % 2,
        )


def scale_and_shift(x: Tensor, scale: Tensor, shift: Tensor) -> Tensor:
    return x * (scale + 1) + shift


class ConvBlock1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        dilation: int = 1,
        num_groups: int = 8,
        use_norm: bool = True,
    ) -> None:
        super().__init__()

        self.groupnorm = (
            nn.GroupNorm(num_groups=num_groups, num_channels=in_channels)
            if use_norm
            else nn.Identity()
        )
        self.activation = nn.SiLU()
        self.project = Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
        )

    def forward(
        self, x: Tensor, scale_shift: Optional[Tuple[Tensor, Tensor]] = None
    ) -> Tensor:
        x = self.groupnorm(x)
        if exists(scale_shift):
            x = scale_and_shift(x, scale=scale_shift[0], shift=scale_shift[1])
        x = self.activation(x)
        return self.project(x)


class ResnetBlock1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        dilation: int = 1,
        num_groups: int,
        time_context_features: Optional[int] = None,
    ) -> None:
        super().__init__()

        self.to_time_embedding = (
            nn.Sequential(
                nn.SiLU(),
                nn.Linear(
                    in_features=time_context_features, out_features=out_channels * 2
                ),
            )
            if exists(time_context_features)
            else None
        )

        self.block1 = ConvBlock1d(
            in_channels=in_channels,
            out_channels=out_channels,
            num_groups=num_groups,
            dilation=dilation,
        )

        self.block2 = ConvBlock1d(
            in_channels=out_channels, out_channels=out_channels, num_groups=num_groups
        )

        self.to_out = (
            Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: Tensor, time_context: Tensor = None) -> Tensor:

        h = self.block1(x)

        # Compute scale and shift from time_context
        scale_shift = None
        if exists(self.to_time_embedding) and exists(time_context):
            time_embedding = self.to_time_embedding(time_context)
            time_embedding = rearrange(time_embedding, "b c -> b c 1")
            scale_shift = time_embedding.chunk(2, dim=1)

        h = self.block2(h, scale_shift=scale_shift)

        return h + self.to_out(x)


class InsertNullTokens(nn.Module):
    def __init__(self, head_features: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.tokens = nn.Parameter(torch.randn(2, head_features))

    def forward(
        self, k: Tensor, v: Tensor, *, mask: Tensor = None
    ) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
        b = k.shape[0]
        nk, nv = repeat_many(
            self.tokens.unbind(dim=-2), "d -> b h 1 d", h=self.num_heads, b=b
        )
        k = torch.cat((nk, k), dim=-2)
        v = torch.cat((nv, v), dim=-2)
        mask = F.pad(mask, pad=(1, 0), value=True) if exists(mask) else None
        return k, v, mask


class LayerNorm(nn.Module):
    def __init__(self, features: int, *, bias: bool = True, eps: float = 1e-5):
        super().__init__()
        self.bias = bias
        self.eps = eps
        self.g = nn.Parameter(torch.ones(features))
        self.b = nn.Parameter(torch.zeros(features)) if bias else None

    def forward(self, x: Tensor) -> Tensor:
        var = torch.var(x, dim=-1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=-1, keepdim=True)
        norm = (x - mean) * (var + self.eps).rsqrt() * self.g
        return norm + self.b if self.bias else norm


def FeedForward(features: int, multiplier: int = 2) -> nn.Module:
    mid_features = int(features * multiplier)
    return nn.Sequential(
        LayerNorm(features, bias=False),
        nn.Linear(in_features=features, out_features=mid_features, bias=False),
        nn.GELU(),
        LayerNorm(mid_features, bias=False),
        nn.Linear(in_features=mid_features, out_features=features, bias=False),
    )


def attention_mask(
    sim: Tensor,
    mask: Tensor,
) -> Tensor:
    mask = rearrange(mask, "b j -> b 1 1 j")
    max_neg_value = -torch.finfo(sim.dtype).max
    sim = sim.masked_fill(~mask, max_neg_value)
    return sim


class AttentionBase(nn.Module):
    def __init__(
        self,
        features: int,
        *,
        head_features: int = 64,
        num_heads: int = 8,
        use_null_tokens: bool = True,
    ):
        super().__init__()
        self.scale = head_features ** -0.5
        self.num_heads = num_heads
        self.use_null_tokens = use_null_tokens
        mid_features = head_features * num_heads

        self.insert_null_tokens = InsertNullTokens(
            head_features=head_features, num_heads=num_heads
        )
        self.to_out = nn.Sequential(
            nn.Linear(in_features=mid_features, out_features=features, bias=False),
            LayerNorm(features=features, bias=False),
        )

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        *,
        mask: Optional[Tensor] = None,
        attention_bias: Optional[Tensor] = None,
    ) -> Tensor:

        # Split heads, scale queries
        q, k, v = rearrange_many((q, k, v), "b n (h d) -> b h n d", h=self.num_heads)
        q = q * self.scale

        # Insert null tokens
        if self.use_null_tokens:
            k, v, mask = self.insert_null_tokens(k, v, mask=mask)

        # Compute similarity matrix with bias and mask
        sim = einsum("... n d, ... m d -> ... n m", q, k)
        sim = sim + attention_bias if exists(attention_bias) else sim
        sim = attention_mask(sim, mask) if exists(mask) else sim

        # Get attention matrix with softmax
        attn = sim.softmax(dim=-1, dtype=torch.float32)

        # Compute values
        out = einsum("... n j, ... j d -> ... n d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class Attention(nn.Module):
    def __init__(
        self,
        features: int,
        *,
        head_features: int = 64,
        num_heads: int = 8,
    ):
        super().__init__()
        mid_features = head_features * num_heads

        self.norm = LayerNorm(features, bias=False)
        self.to_q = nn.Linear(
            in_features=features, out_features=mid_features, bias=False
        )
        self.to_kv = nn.Linear(
            in_features=features, out_features=mid_features * 2, bias=False
        )
        self.attention = AttentionBase(
            features,
            num_heads=num_heads,
            head_features=head_features,
            use_null_tokens=False,
        )

    def forward(self, x: Tensor, *, mask: Optional[Tensor] = None) -> Tensor:
        x = self.norm(x)
        q, k, v = (self.to_q(x), *torch.chunk(self.to_kv(x), chunks=2, dim=-1))
        x = self.attention(q, k, v, mask=mask)
        return x


class CrossAttention(nn.Module):
    def __init__(
        self,
        features: int,
        *,
        context_features: int = None,
        head_features: int = 64,
        num_heads: int = 8,
    ):
        super().__init__()
        mid_features = head_features * num_heads
        context_features = default(context_features, features)

        self.norm_in = LayerNorm(features=features, bias=False)
        self.norm_context = LayerNorm(features=features, bias=False)

        self.to_q = nn.Linear(
            in_features=features, out_features=mid_features, bias=False
        )
        self.to_kv = nn.Linear(
            in_features=context_features, out_features=mid_features * 2, bias=False
        )
        self.attention = AttentionBase(
            features,
            num_heads=num_heads,
            head_features=head_features,
            use_null_tokens=False,
        )

    def forward(self, x: Tensor, context: Tensor, mask: Tensor = None) -> Tensor:
        b, n, d = x.shape
        x = self.norm_in(x)
        context = self.norm_context(context)
        # Queries form x, k and v from context
        q, k, v = (self.to_q(x), *torch.chunk(self.to_kv(context), chunks=2, dim=-1))
        x = self.attention(q, k, v, mask=mask)
        return x


class PerceiverAttention(nn.Module):
    """https://arxiv.org/pdf/2103.03206.pdf"""

    def __init__(
        self,
        features: int,
        *,
        head_features: int = 64,
        num_heads: int = 8,
    ):
        super().__init__()
        self.scale = head_features ** -0.5
        self.num_heads = num_heads
        mid_features = head_features * num_heads

        self.norm_in = nn.LayerNorm(features)
        self.norm_byte = nn.LayerNorm(features)

        self.to_q = nn.Linear(
            in_features=features, out_features=mid_features, bias=False
        )

        self.to_kv = nn.Linear(
            in_features=features, out_features=mid_features * 2, bias=False
        )

        self.attention = AttentionBase(
            features,
            num_heads=num_heads,
            head_features=head_features,
            use_null_tokens=True,
        )

    def forward(self, x: Tensor, byte: Tensor, *, mask: Tensor = None) -> Tensor:
        n = x.shape[-2]
        x = self.norm_in(x)  # latents
        byte = self.norm_byte(byte)
        context = torch.cat([x, byte], dim=-2)
        q, k, v = (self.to_q(x), *torch.chunk(self.to_kv(context), chunks=2, dim=-1))
        mask = F.pad(mask, pad=(0, n), value=True) if exists(mask) else None
        x = self.attention(q, k, v, mask=mask)
        return x


class PerceiverTransformerBlock(nn.Module):
    def __init__(
        self,
        features: int,
        *,
        head_features: int = 64,
        num_heads: int = 8,
        multiplier: int = 2,
    ):
        super().__init__()

        self.attention = PerceiverAttention(
            features=features, head_features=head_features, num_heads=num_heads
        )

        self.feed_forward = FeedForward(features=features, multiplier=multiplier)

    def forward(self, x: Tensor, byte: Tensor, *, mask: Tensor = None) -> Tensor:
        x = self.attention(x, byte, mask=mask) + x
        x = self.feed_forward(x) + x
        return x


class PerceiverTransformer(nn.Module):
    def __init__(
        self,
        *,
        features: int,
        num_blocks: int,
        head_features: int = 64,
        num_heads: int = 8,
        multiplier: int = 4,
    ):
        super().__init__()

        self.blocks = nn.ModuleList(
            [
                PerceiverTransformerBlock(
                    features=features,
                    head_features=head_features,
                    num_heads=num_heads,
                    multiplier=multiplier,
                )
                for i in range(num_blocks)
            ]
        )

    def forward(self, x: Tensor, byte: Tensor, *, mask: Tensor = None) -> Tensor:
        for block in self.blocks:
            x = block(x, byte, mask=mask)
        return x


class MeanPooler(nn.Module):
    def __init__(self, features: int, *, num_tokens: int):
        super().__init__()

        self.to_tokens = nn.Sequential(
            LayerNorm(features=features, bias=False),
            nn.Linear(in_features=features, out_features=features * num_tokens),
            Rearrange("b (n d) -> b n d", n=num_tokens),
        )

    def forward(self, x: Tensor) -> Tensor:
        mean_token = reduce(x, "b n d -> b d", "mean")
        tokens = self.to_tokens(mean_token)
        return tokens


class PerceiverResampler(nn.Module):
    def __init__(
        self,
        *,
        features: int,
        num_latents: int,
        num_pooled: int,
        num_tokens: int,
        num_blocks: int,
        attention_head_features: int = 64,
        attention_num_heads: int = 8,
        attention_multiplier: int = 4,
    ):
        super().__init__()

        self.positional_embedding = nn.Parameter(torch.randn(1, num_tokens, features))

        self.latent_tokens = nn.Parameter(torch.randn(num_latents, features))

        self.to_mean_pooled = MeanPooler(features=features, num_tokens=num_pooled)

        self.transformer = PerceiverTransformer(
            features=features,
            num_blocks=num_blocks,
            head_features=attention_head_features,
            num_heads=attention_num_heads,
            multiplier=attention_multiplier,
        )

    def forward(self, tokens: Tensor, *, mask: Tensor = None) -> Tensor:
        b, n, d = tokens.shape
        # Add positional embedding to tokens
        tokens = tokens + self.positional_embedding
        # Repeat latent tokens over all batch elements
        latent_tokens = repeat(self.latent_tokens, "l d -> b l d", b=b)
        # Concat mean pooled tokens to latent tokens
        latent_tokens = torch.cat((self.to_mean_pooled(tokens), latent_tokens), dim=-2)
        # Resample tokens with transformer (returns num_latent+num_pooled tokens)
        return self.transformer(latent_tokens, tokens, mask=mask)


def rand_bool(shape: Any, proba: float, device: Any = None) -> Tensor:
    if proba == 1:
        return torch.ones(shape, device=device, dtype=torch.bool)
    elif proba == 0:
        return torch.zeros(shape, device=device, dtype=torch.bool)
    else:
        return torch.bernoulli(torch.full(shape, proba, device=device)).to(torch.bool)


class LearnedRandomMasker(nn.Module):
    """Masks random batches and masked tokens with fixed leared tokens."""

    def __init__(
        self,
        features: int,
        num_tokens: int,
    ):
        super().__init__()
        self.fixed_tokens = nn.Parameter(torch.randn(1, num_tokens, features))

    def forward(
        self, tokens: Tensor, proba_keep_batch: float, *, mask: Tensor = None
    ) -> Tensor:
        b, device = tokens.shape[0], tokens.device
        batch_mask = rand_bool(shape=(b, 1, 1), proba=proba_keep_batch, device=device)
        full_mask = batch_mask
        if exists(mask):
            full_mask = batch_mask & rearrange(mask, "b n -> b n 1")
        return torch.where(full_mask, tokens, self.fixed_tokens)


def crop_or_pad_tokens(tokens: Tensor, num_tokens: int, pad_value: Any = 0) -> Tensor:
    n = tokens.shape[1]
    tokens = tokens[:, :num_tokens]
    if n < num_tokens:
        tokens = F.pad(tokens, (0, 0, 0, num_tokens - n), value=pad_value)
    return tokens


class TokenConditiner(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_tokens: int,
        use_resampler: bool,
        resampling_num_latents: Optional[int] = None,
        resampling_num_pooled: Optional[int] = None,
        resampling_num_blocks: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_tokens = num_tokens
        self.use_resampler = use_resampler

        self.to_cond = nn.Linear(in_features=in_features, out_features=out_features)

        if use_resampler:
            assert (
                exists(resampling_num_latents)
                and exists(resampling_num_pooled)
                and resampling_num_blocks
            )

            self.resample = PerceiverResampler(
                features=out_features,
                num_latents=resampling_num_latents,
                num_pooled=resampling_num_pooled,
                num_tokens=num_tokens,
                num_blocks=resampling_num_blocks,
            )

        self.to_mask = LearnedRandomMasker(features=out_features, num_tokens=num_tokens)

    def forward(
        self,
        tokens: Tensor,
        *,
        mask: Optional[Tensor] = None,
        proba_mask_batch: float = 0.0,
    ) -> Tensor:
        b, n, d = tokens.shape
        assert d == self.in_features

        cond_tokens = self.to_cond(tokens)  # (b, n, out_features)
        cond_tokens = crop_or_pad_tokens(cond_tokens, num_tokens=self.num_tokens)

        if exists(mask):
            # mask = rearrange(mask, 'b n -> b n 1')
            mask = crop_or_pad_tokens(mask, num_tokens=self.num_tokens, pad_value=False)

        cond_tokens = self.to_mask(
            cond_tokens, mask=mask, proba_keep_batch=1 - proba_mask_batch
        )

        if self.use_resampler:
            cond_tokens = self.resample(cond_tokens, mask=mask)

        return cond_tokens


class LayerNorm1d(nn.Module):
    def __init__(self, channels: int, *, bias: bool = True, eps: float = 1e-5):
        super().__init__()
        self.bias = bias
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, channels, 1))
        self.b = nn.Parameter(torch.zeros(1, channels, 1)) if bias else None

    def forward(self, x: Tensor) -> Tensor:
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        norm = (x - mean) * (var + self.eps).rsqrt() * self.g
        return norm + self.b if self.bias else norm


def FeedForward1d(channels: int, multiplier: int = 2):
    mid_channels = int(channels * multiplier)
    return nn.Sequential(
        LayerNorm1d(channels=channels, bias=False),
        Conv1d(
            in_channels=channels, out_channels=mid_channels, kernel_size=1, bias=False
        ),
        nn.GELU(),
        LayerNorm1d(channels=mid_channels, bias=False),
        Conv1d(
            in_channels=mid_channels, out_channels=channels, kernel_size=1, bias=False
        ),
    )


class TransformerBlock1d(nn.Module):
    def __init__(
        self,
        channels: int,
        *,
        num_heads: int = 8,
        head_features: int = 32,
        multiplier: int = 2,
    ):
        super().__init__()
        self.attention = EinopsToAndFrom(
            "b c l",
            "b l c",
            Attention(
                features=channels, num_heads=num_heads, head_features=head_features
            ),
        )
        self.feed_forward = FeedForward1d(channels=channels, multiplier=multiplier)

    def forward(self, x: Tensor) -> Tensor:
        x = self.attention(x) + x
        x = self.feed_forward(x) + x
        return x


class CrossEmbed1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        *,
        kernel_sizes: Sequence[int],
        stride: int,
        out_channels: Optional[int] = None,
    ):
        super().__init__()
        assert all([*map(lambda t: (t % 2) == (stride % 2), kernel_sizes)])
        out_channels = default(out_channels, in_channels)

        kernel_sizes = sorted(kernel_sizes)
        num_scales = len(kernel_sizes)

        channels_list = [int(out_channels / (2 ** i)) for i in range(1, num_scales)]
        channels_list = [*channels_list, out_channels - sum(channels_list)]

        self.convs = nn.ModuleList([])
        for kernel_size, channels in zip(kernel_sizes, channels_list):
            self.convs += [
                Conv1d(
                    in_channels=in_channels,
                    out_channels=channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=(kernel_size - stride) // 2,
                )
            ]

    def forward(self, x):
        out_list = tuple(map(lambda conv: conv(x), self.convs))
        return torch.cat(out_list, dim=1)


class DownsampleBlock1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        factor: int,
        kernel_multiplier: int,
        dilations: Sequence[int],
        time_context_features: int,
        num_groups: int,
        use_pre_downsample: bool,
        use_attention: bool,
        attention_heads: Optional[int] = None,
        attention_features: Optional[int] = None,
        attention_multiplier: Optional[int] = None,
    ):
        super().__init__()

        self.use_pre_downsample = use_pre_downsample
        self.use_attention = use_attention

        channels = out_channels if use_pre_downsample else in_channels

        self.blocks = nn.ModuleList(
            [
                ResnetBlock1d(
                    in_channels=channels,
                    out_channels=channels,
                    dilation=dilation,
                    num_groups=num_groups,
                    time_context_features=time_context_features,
                )
                for dilation in dilations
            ]
        )

        if use_attention:
            assert (
                exists(attention_heads)
                and exists(attention_features)
                and exists(attention_multiplier)
            )
            self.transformer = TransformerBlock1d(
                channels=channels,
                num_heads=attention_heads,
                head_features=attention_features,
                multiplier=attention_multiplier,
            )

        self.downsample = Downsample1d(
            in_channels=in_channels,
            out_channels=out_channels,
            factor=factor,
            kernel_multiplier=kernel_multiplier,
        )

    def forward(self, x: Tensor, t: Tensor) -> Tuple[Tensor, List[Tensor]]:

        if self.use_pre_downsample:
            x = self.downsample(x)

        skips = []
        for block in self.blocks:
            x = block(x, t)
            skips += [x]

        if self.use_attention:
            x = self.transformer(x)
            skips += [x]

        if not self.use_pre_downsample:
            x = self.downsample(x)

        return x, skips


class UpsampleBlock1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        *,
        factor: int,
        use_nearest: bool,
        dilations: Sequence[int],
        time_context_features: int,
        num_groups: int,
        use_pre_upsample: bool,
        use_skip_scale: bool,
        use_attention: bool,
        attention_heads: Optional[int] = None,
        attention_features: Optional[int] = None,
        attention_multiplier: Optional[int] = None,
    ):
        super().__init__()

        assert (not use_attention) or (
            exists(attention_heads)
            and exists(attention_features)
            and exists(attention_multiplier)
        )

        self.use_pre_upsample = use_pre_upsample
        self.use_attention = use_attention
        self.num_layers = len(dilations)
        self.skip_scale = 2 ** -0.5 if use_skip_scale else 1.0

        channels = out_channels if use_pre_upsample else in_channels

        self.blocks = nn.ModuleList(
            [
                ResnetBlock1d(
                    in_channels=channels + skip_channels,
                    out_channels=channels,
                    dilation=dilation,
                    num_groups=num_groups,
                    time_context_features=time_context_features,
                )
                for dilation in dilations
            ]
        )

        if use_attention:
            assert (
                exists(attention_heads)
                and exists(attention_features)
                and exists(attention_multiplier)
            )
            self.transformer = TransformerBlock1d(
                channels=channels,
                num_heads=attention_heads,
                head_features=attention_features,
                multiplier=attention_multiplier,
            )

        self.upsample = Upsample1d(
            in_channels=in_channels,
            out_channels=out_channels,
            factor=factor,
            use_nearest=use_nearest,
        )

    def add_skip(self, x: Tensor, skip: Tensor) -> Tensor:
        return torch.cat([x, skip * self.skip_scale], dim=1)

    def forward(self, x: Tensor, skips: List[Tensor], t: Tensor) -> Tensor:

        if self.use_pre_upsample:
            x = self.upsample(x)

        for block in self.blocks:
            x = self.add_skip(x, skip=skips.pop())
            x = block(x, t)

        if self.use_attention:
            x = self.transformer(x)

        if not self.use_pre_upsample:
            x = self.upsample(x)

        return x


class BottleneckBlock1d(nn.Module):
    def __init__(
        self,
        channels: int,
        *,
        time_context_features: int,
        num_groups: int,
        use_attention: bool,
        attention_heads: Optional[int] = None,
        attention_features: Optional[int] = None,
    ):
        super().__init__()

        assert (not use_attention) or (
            exists(attention_heads) and exists(attention_features)
        )

        self.use_attention = use_attention

        self.pre_block = ResnetBlock1d(
            in_channels=channels,
            out_channels=channels,
            num_groups=num_groups,
            time_context_features=time_context_features,
        )

        if use_attention:
            assert exists(attention_heads) and exists(attention_features)
            self.attention = EinopsToAndFrom(
                "b c l",
                "b l c",
                Attention(
                    features=channels,
                    num_heads=attention_heads,
                    head_features=attention_features,
                ),
            )

        self.post_block = ResnetBlock1d(
            in_channels=channels,
            out_channels=channels,
            num_groups=num_groups,
            time_context_features=time_context_features,
        )

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        x = self.pre_block(x, t)
        if self.use_attention:
            x = self.attention(x)
        x = self.post_block(x, t)
        return x


class UNet1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        channels: int,
        multipliers: Sequence[int],
        factors: Sequence[int],
        dilations: Sequence[Sequence[int]],
        attentions: Sequence[bool],
        attention_heads: int,
        attention_features: int,
        attention_multiplier: int,
        resnet_groups: int,
        kernel_multiplier_downsample: int,
        kernel_sizes_init: Sequence[int],
        use_learned_time_embedding: bool,
        use_nearest_upsample: bool,
        use_skip_scale: bool,
        use_attention_bottleneck: bool,
        out_channels: Optional[int] = None,
        patch_size: int = 1,
    ):
        super().__init__()

        out_channels = default(out_channels, in_channels)
        time_context_features = channels * 4
        num_layers = len(multipliers) - 1

        assert (
            len(factors) == num_layers
            and len(attentions) == num_layers
            and len(dilations) == num_layers
        )

        self.to_in = nn.Sequential(
            Rearrange("b c (l p) -> b (c p) l", p=patch_size),
            CrossEmbed1d(
                in_channels=in_channels * patch_size,
                out_channels=channels,
                kernel_sizes=kernel_sizes_init,
                stride=1,
            ),
        )

        self.to_time = nn.Sequential(
            TimePositionalEmbedding(
                dim=channels,
                out_features=time_context_features,
                use_learned=use_learned_time_embedding,
            ),
            nn.SiLU(),
            nn.Linear(
                in_features=time_context_features, out_features=time_context_features
            ),
        )

        self.downsamples = nn.ModuleList(
            [
                DownsampleBlock1d(
                    in_channels=channels * multipliers[i],
                    out_channels=channels * multipliers[i + 1],
                    time_context_features=time_context_features,
                    dilations=dilations[i],
                    factor=factors[i],
                    kernel_multiplier=kernel_multiplier_downsample,
                    num_groups=resnet_groups,
                    use_pre_downsample=True,
                    use_attention=attentions[i],
                    attention_heads=attention_heads,
                    attention_features=attention_features,
                    attention_multiplier=attention_multiplier,
                )
                for i in range(num_layers)
            ]
        )

        self.bottleneck = BottleneckBlock1d(
            channels=channels * multipliers[-1],
            time_context_features=time_context_features,
            num_groups=resnet_groups,
            use_attention=use_attention_bottleneck,
            attention_heads=attention_heads,
            attention_features=attention_features,
        )

        self.upsamples = nn.ModuleList(
            [
                UpsampleBlock1d(
                    in_channels=channels * multipliers[i + 1],
                    skip_channels=channels * multipliers[i + 1],
                    out_channels=channels * multipliers[i],
                    time_context_features=time_context_features,
                    dilations=(1,) * (len(dilations[i]) + (1 if attentions[i] else 0)),
                    factor=factors[i],
                    use_nearest=use_nearest_upsample,
                    num_groups=resnet_groups,
                    use_skip_scale=use_skip_scale,
                    use_pre_upsample=False,
                    use_attention=attentions[i],
                    attention_heads=attention_heads,
                    attention_features=attention_features,
                    attention_multiplier=attention_multiplier,
                )
                for i in reversed(range(num_layers))
            ]
        )

        self.to_out = nn.Sequential(
            ResnetBlock1d(
                in_channels=channels,
                out_channels=channels,
                num_groups=resnet_groups,
                time_context_features=time_context_features,
            ),
            Conv1d(
                in_channels=channels,
                out_channels=out_channels * patch_size,
                kernel_size=1,
            ),
            Rearrange("b (c p) l -> b c (l p)", p=patch_size),
        )

    def forward(self, x: Tensor, t: Tensor):

        x = self.to_in(x)
        t = self.to_time(t)
        skips_list = []

        for downsample in self.downsamples:
            x, skips = downsample(x, t)
            skips_list += [skips]

        x = self.bottleneck(x, t)

        for upsample in self.upsamples:
            skips = skips_list.pop()
            x = upsample(x, skips, t)

        x = self.to_out(x)

        return x


class UNet1dAlpha(UNet1d):
    def __init__(self, *args, **kwargs):
        default_kwargs = dict(
            in_channels=1,
            patch_size=1,
            channels=128,
            multipliers=[1, 2, 4, 4, 4, 4, 4],
            factors=[4, 4, 4, 4, 2, 2],
            attentions=[False, False, False, False, True, True],
            attention_heads=8,
            attention_features=64,
            attention_multiplier=2,
            dilations=[
                [1, 1, 1],
                [1, 1, 1],
                [1, 1, 1],
                [1, 1, 1],
                [1, 1, 1],
                [1, 1, 1],
            ],
            resnet_groups=8,
            kernel_multiplier_downsample=2,
            kernel_sizes_init=[1, 3, 7],
            use_nearest_upsample=False,
            use_skip_scale=True,
            use_attention_bottleneck=True,
            use_learned_time_embedding=True,
        )
        super().__init__(*args, **{**default_kwargs, **kwargs})


class UNet1dBravo(UNet1d):
    def __init__(self, *args, **kwargs):
        default_kwargs = dict(
            in_channels=1,
            patch_size=8,
            channels=128,
            multipliers=[1, 2, 4, 4, 4, 4, 4],
            factors=[4, 4, 4, 2, 2, 2],
            attentions=[False, False, False, True, True, True],
            attention_heads=8,
            attention_features=64,
            attention_multiplier=2,
            dilations=[[1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1]],
            resnet_groups=8,
            kernel_multiplier_downsample=2,
            kernel_sizes_init=[1, 3, 7],
            use_nearest_upsample=False,
            use_skip_scale=True,
            use_attention_bottleneck=True,
            use_learned_time_embedding=True,
        )
        super().__init__(*args, **{**default_kwargs, **kwargs})
