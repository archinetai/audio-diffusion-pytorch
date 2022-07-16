from math import log, pi
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from einops import rearrange
from einops.layers.torch import Rearrange
from einops_exts import rearrange_many, repeat_many
from einops_exts.torch import EinopsToAndFrom
from torch import Tensor, einsum
from torch.nn import functional as F

from .utils import default, exists


def Conv1d(*args, **kwargs):
    return nn.Conv1d(*args, **kwargs)


def ConvTranspose1d(*args, **kwargs):
    return nn.ConvTranspose1d(*args, **kwargs)


class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        half_dim = self.dim // 2
        emb = log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=x.device) * -emb)
        emb = rearrange(x, "i -> i 1") * rearrange(emb, "j -> 1 j")
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


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


def TimePositionalEmbedding(dim: int, out_features: int, use_learned: bool):
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
            else nn.Identity()
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

    def forward(self, k: Tensor, v: Tensor) -> Tuple[Tensor, Tensor]:
        b = k.shape[0]
        nk, nv = repeat_many(
            self.tokens.unbind(dim=-2), "d -> b h 1 d", h=self.num_heads, b=b
        )
        k = torch.cat((nk, k), dim=-2)
        v = torch.cat((nv, v), dim=-2)
        return k, v


def attention_mask(sim: Tensor, mask: Tensor) -> Tensor:
    mask = F.pad(mask, pad=(1, 0), value=True)
    mask = rearrange(mask, "b j -> b 1 1 j")
    max_neg_value = -torch.finfo(sim.dtype).max
    sim = sim.masked_fill(~mask, max_neg_value)
    return sim


class CenteredLayerNorm(nn.Module):
    def __init__(self, features: int):
        super().__init__()
        # Learned variance (gamma), fixed mean (bias)
        self.gamma = nn.Parameter(torch.ones(features))
        self.register_buffer("beta", torch.zeros(features))

    def forward(self, x: Tensor) -> Tensor:
        shape = (x.shape[-1],)
        return F.layer_norm(
            x, normalized_shape=shape, weight=self.gamma, bias=self.beta
        )


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
            CenteredLayerNorm(features=features),
        )

    def forward(
        self, q: Tensor, k: Tensor, v: Tensor, mask: Tensor = None, attention_bias=None
    ) -> Tensor:

        # Split heads, scale queries, insert null tokens
        q, k, v = rearrange_many((q, k, v), "b n (h d) -> b h n d", h=self.num_heads)
        q = q * self.scale
        k, v = self.insert_null_tokens(k, v) if self.use_null_tokens else (k, v)

        # Compute similarity matrix
        sim = einsum("b h i d, b h j d -> b h i j", q, k)
        sim = sim + attention_bias if exists(attention_bias) else sim
        sim = attention_mask(sim, mask) if exists(mask) else sim

        # Attend with stable softmax
        attn = sim.softmax(dim=-1, dtype=torch.float32)

        # Compute values
        out = einsum("b h i j, b h j d -> b h i d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


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
        self.scale = head_features ** -0.5
        self.num_heads = num_heads
        mid_features = head_features * num_heads
        context_features = default(context_features, features)

        self.norm = CenteredLayerNorm(features=features)
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
        x = self.norm(x)
        # Queries form x, k and v from context
        q, k, v = (self.to_q(x), *self.to_kv(context).chunk(2, dim=-1))
        x = self.attention(q, k, v, mask)
        return x


class Attention(CrossAttention):
    def forward(self, x: Tensor, *args, **kwargs) -> Tensor:
        return super().forward(x, context=x, *args, **kwargs)


class LayerNorm1d(nn.Module):
    def __init__(self, channels: int, *, bias: bool = True, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.bias = bias
        self.g = nn.Parameter(torch.ones(1, channels, 1))
        self.b = nn.Parameter(torch.zeros(1, channels, 1)) if bias else None

    def forward(self, x: Tensor) -> Tensor:
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        norm = (x - mean) / (var + self.eps).sqrt() * self.g
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

        assert (not use_attention) or (
            exists(attention_heads)
            and exists(attention_features)
            and exists(attention_multiplier)
        )

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

        self.transformer = (
            TransformerBlock1d(
                channels=channels,
                num_heads=attention_heads,
                head_features=attention_features,
                multiplier=attention_multiplier,
            )
            if use_attention
            else nn.Identity()
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
        use_nearest: int,
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

        self.transformer = (
            TransformerBlock1d(
                channels=channels,
                num_heads=attention_heads,
                head_features=attention_features,
                multiplier=attention_multiplier,
            )
            if use_attention
            else nn.Identity()
        )

        self.upsample = Upsample1d(
            in_channels=in_channels,
            out_channels=out_channels,
            factor=factor,
            use_nearest=use_nearest,
        )

    def add_skip(self, x: Tensor, skip: Tensor) -> Tensor:
        return torch.cat([x, skip * self.skip_scale], dim=1)

    def forward(self, x: Tensor, skips: Sequence[Tensor], t: Tensor) -> Tensor:

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

        self.attention = (
            EinopsToAndFrom(
                "b c l",
                "b l c",
                Attention(
                    features=channels,
                    num_heads=attention_heads,
                    head_features=attention_features,
                ),
            )
            if use_attention
            else nn.Identity()
        )

        self.post_block = ResnetBlock1d(
            in_channels=channels,
            out_channels=channels,
            num_groups=num_groups,
            time_context_features=time_context_features,
        )

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        x = self.pre_block(x, t)
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
        use_nearest_upsample: int,
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
            patch_size=1,
        )
        super().__init__(*args, **{**default_kwargs, **kwargs})


class UNet1dBravo(UNet1d):
    def __init__(self, *args, **kwargs):
        default_kwargs = dict(
            in_channels=1,
            patch_size=4,
            channels=128,
            multipliers=[1, 2, 4, 4, 4, 4, 4],
            factors=[4, 4, 4, 4, 2, 2],
            attentions=[False, False, False, False, True, True],
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
