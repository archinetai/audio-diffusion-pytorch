from math import log, pi
from typing import List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
from einops import rearrange, reduce
from einops.layers.torch import Rearrange
from einops_exts import rearrange_many, repeat_many
from einops_exts.torch import EinopsToAndFrom
from torch import Tensor, einsum
from torch.nn import functional as F

from .utils import default, exists

"""
Convolutional Blocks
"""


def Conv1d(*args, **kwargs) -> nn.Module:
    return nn.Conv1d(*args, **kwargs)


def ConvTranspose1d(*args, **kwargs) -> nn.Module:
    return nn.ConvTranspose1d(*args, **kwargs)


def Downsample1d(
    in_channels: int, out_channels: int, factor: int, kernel_multiplier: int = 2
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
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
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
            kernel_size=kernel_size,
            stride=stride,
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
        num_groups: int,
        dilation: int = 1,
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

    def forward(self, x: Tensor, time_context: Optional[Tensor] = None) -> Tensor:

        h = self.block1(x)

        # Compute scale and shift from time_context
        scale_shift = None
        if exists(self.to_time_embedding) and exists(time_context):
            time_embedding = self.to_time_embedding(time_context)
            time_embedding = rearrange(time_embedding, "b c -> b c 1")
            scale_shift = time_embedding.chunk(2, dim=1)

        h = self.block2(h, scale_shift=scale_shift)

        return h + self.to_out(x)


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


"""
Norms
"""


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


"""
Attention Components
"""


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
        out_features: Optional[int] = None,
    ):
        super().__init__()
        self.scale = head_features ** -0.5
        self.num_heads = num_heads
        self.use_null_tokens = use_null_tokens
        mid_features = head_features * num_heads
        out_features = out_features if exists(out_features) else features

        self.insert_null_tokens = InsertNullTokens(
            head_features=head_features, num_heads=num_heads
        )
        self.to_out = nn.Sequential(
            nn.Linear(in_features=mid_features, out_features=out_features, bias=False),
            LayerNorm(features=out_features, bias=False),
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


"""
Attention Blocks
"""


class Attention(nn.Module):
    def __init__(
        self,
        features: int,
        *,
        head_features: int = 64,
        num_heads: int = 8,
        out_features: Optional[int] = None,
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
            out_features=out_features,
        )

    def forward(self, x: Tensor, *, mask: Optional[Tensor] = None) -> Tensor:
        x = self.norm(x)
        q, k, v = (self.to_q(x), *torch.chunk(self.to_kv(x), chunks=2, dim=-1))
        x = self.attention(q, k, v, mask=mask)
        return x


"""
Transformer Blocks
"""


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


"""
Time Embeddings
"""


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


"""
Encoder/Decoder Components
"""


class DownsampleBlock1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        factor: int,
        num_groups: int,
        num_layers: int,
        kernel_multiplier: int = 2,
        use_pre_downsample: bool = True,
        use_skip: bool = False,
        use_attention: bool = False,
        attention_heads: Optional[int] = None,
        attention_features: Optional[int] = None,
        attention_multiplier: Optional[int] = None,
        time_context_features: Optional[int] = None,
    ):
        super().__init__()
        self.use_pre_downsample = use_pre_downsample
        self.use_skip = use_skip
        self.use_attention = use_attention

        channels = out_channels if use_pre_downsample else in_channels

        self.downsample = Downsample1d(
            in_channels=in_channels,
            out_channels=out_channels,
            factor=factor,
            kernel_multiplier=kernel_multiplier,
        )

        self.blocks = nn.ModuleList(
            [
                ResnetBlock1d(
                    in_channels=channels,
                    out_channels=channels,
                    num_groups=num_groups,
                    time_context_features=time_context_features,
                )
                for _ in range(num_layers)
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

    def forward(
        self, x: Tensor, t: Optional[Tensor] = None
    ) -> Union[Tuple[Tensor, List[Tensor]], Tensor]:

        if self.use_pre_downsample:
            x = self.downsample(x)

        skips = []
        for block in self.blocks:
            x = block(x, t)
            skips += [x] if self.use_skip else []

        if self.use_attention:
            x = self.transformer(x)
            skips += [x] if self.use_skip else []

        if not self.use_pre_downsample:
            x = self.downsample(x)

        return (x, skips) if self.use_skip else x


class UpsampleBlock1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        factor: int,
        num_layers: int,
        num_groups: int,
        use_nearest: bool = False,
        use_pre_upsample: bool = False,
        use_skip: bool = False,
        skip_channels: int = 0,
        use_skip_scale: bool = False,
        use_attention: bool = False,
        attention_heads: Optional[int] = None,
        attention_features: Optional[int] = None,
        attention_multiplier: Optional[int] = None,
        time_context_features: Optional[int] = None,
    ):
        super().__init__()

        assert (not use_attention) or (
            exists(attention_heads)
            and exists(attention_features)
            and exists(attention_multiplier)
        )

        self.use_pre_upsample = use_pre_upsample
        self.use_attention = use_attention
        self.use_skip = use_skip
        self.skip_scale = 2 ** -0.5 if use_skip_scale else 1.0

        channels = out_channels if use_pre_upsample else in_channels

        self.blocks = nn.ModuleList(
            [
                ResnetBlock1d(
                    in_channels=channels + skip_channels,
                    out_channels=channels,
                    num_groups=num_groups,
                    time_context_features=time_context_features,
                )
                for _ in range(num_layers)
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

    def forward(
        self,
        x: Tensor,
        skips: Optional[List[Tensor]] = None,
        t: Optional[Tensor] = None,
    ) -> Tensor:

        if self.use_pre_upsample:
            x = self.upsample(x)

        for block in self.blocks:
            x = self.add_skip(x, skip=skips.pop()) if exists(skips) else x
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
        num_groups: int,
        use_attention: bool = False,
        attention_heads: Optional[int] = None,
        attention_features: Optional[int] = None,
        time_context_features: Optional[int] = None,
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

    def forward(self, x: Tensor, t: Optional[Tensor] = None) -> Tensor:
        x = self.pre_block(x, t)
        if self.use_attention:
            x = self.attention(x)
        x = self.post_block(x, t)
        return x


""" UNets """


class UNet1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        channels: int,
        patch_size: int,
        multipliers: Sequence[int],
        factors: Sequence[int],
        num_blocks: Sequence[int],
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
    ):
        super().__init__()

        out_channels = default(out_channels, in_channels)
        time_context_features = channels * 4
        num_layers = len(multipliers) - 1
        self.num_layers = num_layers

        assert (
            len(factors) == num_layers
            and len(attentions) == num_layers
            and len(num_blocks) == num_layers
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
                    num_layers=num_blocks[i],
                    factor=factors[i],
                    kernel_multiplier=kernel_multiplier_downsample,
                    num_groups=resnet_groups,
                    use_pre_downsample=True,
                    use_skip=True,
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
                    out_channels=channels * multipliers[i],
                    time_context_features=time_context_features,
                    num_layers=num_blocks[i] + (1 if attentions[i] else 0),
                    factor=factors[i],
                    use_nearest=use_nearest_upsample,
                    num_groups=resnet_groups,
                    use_skip_scale=use_skip_scale,
                    use_pre_upsample=False,
                    use_skip=True,
                    skip_channels=channels * multipliers[i + 1],
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

        x = self.to_out(x)  # t?

        return x


""" Autoencoders """


def gaussian_sample(mean: Tensor, logvar: Tensor) -> Tensor:
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    sample = mean + std * eps
    return sample


def kl_loss(mean: Tensor, logvar: Tensor) -> Tensor:
    losses = reduce(1 + logvar - mean ** 2 - logvar.exp(), "b ... -> b", "sum")
    loss = reduce(-0.5 * losses, "b -> 1", "mean")
    return loss


class AutoEncoder1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        bottleneck_channels: int,
        channels: int,
        patch_size: int,
        multipliers: Sequence[int],
        factors: Sequence[int],
        num_blocks: Sequence[int],
        resnet_groups: int,
        loss_kl_weight: float,
        kernel_multiplier_downsample: int = 2,
    ):
        super().__init__()

        num_layers = len(multipliers) - 1
        self.num_layers = num_layers
        self.loss_kl_weight = loss_kl_weight

        assert len(factors) == num_layers and len(num_blocks) == num_layers

        self.to_in = nn.Sequential(
            Rearrange("b c (l p) -> b (c p) l", p=patch_size),
            Conv1d(
                in_channels=in_channels * patch_size,
                out_channels=channels,
                kernel_size=1,
            ),
        )

        self.downsamples = nn.ModuleList(
            [
                DownsampleBlock1d(
                    in_channels=channels * multipliers[i],
                    out_channels=channels * multipliers[i + 1],
                    num_layers=num_blocks[i],
                    factor=factors[i],
                    kernel_multiplier=kernel_multiplier_downsample,
                    num_groups=resnet_groups,
                )
                for i in range(num_layers)
            ]
        )

        self.pre_bottleneck = Conv1d(
            in_channels=channels * multipliers[-1],
            out_channels=bottleneck_channels * 2,
            kernel_size=1,
        )

        self.post_bottleneck = Conv1d(
            in_channels=bottleneck_channels,
            out_channels=channels * multipliers[-1],
            kernel_size=1,
        )

        self.upsamples = nn.ModuleList(
            [
                UpsampleBlock1d(
                    in_channels=channels * multipliers[i + 1],
                    out_channels=channels * multipliers[i],
                    num_layers=num_blocks[i],
                    factor=factors[i],
                    num_groups=resnet_groups,
                )
                for i in reversed(range(num_layers))
            ]
        )

        self.to_out = nn.Sequential(
            Conv1d(
                in_channels=channels,
                out_channels=in_channels * patch_size,
                kernel_size=1,
            ),
            Rearrange("b (c p) l -> b c (l p)", p=patch_size),
        )

    def encode(
        self, x: Tensor, *, with_kl_loss: bool = False
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        x = self.to_in(x)

        for downsample in self.downsamples:
            x = downsample(x)

        mean_and_var = self.pre_bottleneck(x)

        # Chunk channels to mean and log variance and sample in VAE style
        mean, logvar = torch.chunk(mean_and_var, chunks=2, dim=1)
        logvar = torch.clamp(logvar, -30.0, 20.0)
        bottleneck = gaussian_sample(mean, logvar)

        if with_kl_loss:
            return bottleneck, kl_loss(mean, logvar)

        return bottleneck

    def decode(self, x: Tensor) -> Tensor:
        x = self.post_bottleneck(x)

        for upsample in self.upsamples:
            x = upsample(x)

        return self.to_out(x)

    def forward(self, x: Tensor) -> Tensor:
        """Returns autoencoding loss"""
        z, kl_loss = self.encode(x, with_kl_loss=True)
        y = self.decode(z)
        loss = F.mse_loss(x, y) + kl_loss * self.loss_kl_weight
        return loss
