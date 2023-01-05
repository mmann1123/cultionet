import typing as T

from . import model_utils

import torch
from torch_geometric import nn


class Permute(torch.nn.Module):
    def __init__(self, axis_order: T.Sequence[int]):
        super(Permute, self).__init__()
        self.axis_order = axis_order

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.permute(*self.axis_order)


class Add(torch.nn.Module):
    def __init__(self):
        super(Add, self).__init__()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return x + y


class Squeeze(torch.nn.Module):
    def __init__(self):
        super(Squeeze, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.squeeze()


class DepthwiseConv2d(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: int = 0,
        dilation: int = 1,
        bias: bool = False
    ):
        super(DepthwiseConv2d, self).__init__()

        layers = [
            torch.nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation,
                groups=in_channels
            ),
            torch.nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                padding=0,
                bias=bias
            )
        ]
        self.seq = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.seq(x)


class DepthwiseConvBlock2d(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: int = 0,
        dilation: int = 1,
        add_activation: bool = True,
        activation_type: str = 'LeakyReLU'
    ):
        super(DepthwiseConvBlock2d, self).__init__()

        layers = [
            DepthwiseConv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation,
                bias=False
            ),
            torch.nn.BatchNorm2d(out_channels)
        ]
        if add_activation:
            layers += [
                getattr(torch.nn, activation_type)(inplace=False)
            ]

        self.seq = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.seq(x)


class ConvBlock2d(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: int = 0,
        dilation: int = 1,
        add_activation: bool = True,
        activation_type: str = 'LeakyReLU'
    ):
        super(ConvBlock2d, self).__init__()

        layers = [
            torch.nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation,
                bias=False
            ),
            torch.nn.BatchNorm2d(out_channels)
        ]
        if add_activation:
            layers += [
                getattr(torch.nn, activation_type)(inplace=False)
            ]

        self.seq = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.seq(x)


class ResBlock2d(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: int = 0,
        dilation: int = 1
    ):
        super(ResBlock2d, self).__init__()

        layers = [
            torch.nn.BatchNorm2d(in_channels),
            torch.nn.LeakyReLU(inplace=False),
            torch.nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation
            )
        ]

        self.seq = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.seq(x)


class ConvBlock3d(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        in_time: int = 0,
        padding: int = 0,
        dilation: int = 1,
        add_activation: bool = True,
        activation_type: str = 'LeakyReLU',
        squeeze: bool = True
    ):
        super(ConvBlock3d, self).__init__()

        layers = [
            torch.nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation,
                bias=False
            )
        ]
        if squeeze:
            layers += [
                Squeeze(),
                torch.nn.BatchNorm2d(in_time)
            ]
        else:
            layers += [torch.nn.BatchNorm3d(out_channels)]
        if add_activation:
            layers += [
                getattr(torch.nn, activation_type)(inplace=False)
            ]

        self.seq = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.seq(x)


class AttentionGate(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int
    ):
        super(AttentionGate, self).__init__()

        conv_x = ConvBlock2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            padding=0,
            add_activation=False
        )
        conv_g = ConvBlock2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            padding=0,
            add_activation=False
        )
        conv_psi = torch.nn.Sequential(
            torch.nn.Conv2d(
                out_channels,
                1,
                kernel_size=1,
                padding=0
            ),
            torch.nn.Sigmoid()
        )
        self.up = model_utils.UpSample()

        self.seq = nn.Sequential(
            'g, x',
            [
                (conv_g, 'g -> g'),
                (conv_x, 'x -> h'),
                (Add(), 'g, h -> h'),
                (torch.nn.ReLU(inplace=False), 'h -> h'),
                (conv_psi, 'h -> h')
            ]
        )

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            g: Higher feature dimension, lower spatial resolution
            x: Lower feature dimension, higher spatial resolution
        """
        h = self.seq(g, x)

        return h * x


class TanimotoComplement(torch.nn.Module):
    """Tanimoto distance with complement

    Adapted from publications and source code below:

        CSIRO BSTD/MIT LICENSE

        Redistribution and use in source and binary forms, with or without modification, are permitted provided that
        the following conditions are met:

        1. Redistributions of source code must retain the above copyright notice, this list of conditions and the
            following disclaimer.
        2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and
            the following disclaimer in the documentation and/or other materials provided with the distribution.
        3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or
            promote products derived from this software without specific prior written permission.

        THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
        INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
        DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
        SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
        SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
        WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE
        USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

        References:
            https://www.mdpi.com/2072-4292/14/22/5738
            https://arxiv.org/abs/2009.02062
            https://github.com/waldnerf/decode/blob/main/FracTAL_ResUNet/nn/layers/ftnmt.py
    """
    def __init__(
        self,
        smooth: float = 1e-5,
        depth: int = 5,
        dim: T.Union[int, T.Sequence[int]] = 0,
        targets_are_labels: bool = True
    ):
        super(TanimotoComplement, self).__init__()

        self.smooth = smooth
        self.depth = depth
        self.dim = dim
        self.targets_are_labels = targets_are_labels

    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """Performs a single forward pass

        Args:
            inputs: Predictions from model (probabilities or labels).
            targets: Ground truth values.

        Returns:
            Tanimoto distance loss (float)
        """
        if self.depth == 1:
            scale = 1.0
        else:
            scale = 1.0 / self.depth

        def tanimoto(y: torch.Tensor, yhat: torch.Tensor) -> torch.Tensor:
            tpl = torch.sum(y * yhat, dim=self.dim, keepdim=True)
            numerator = tpl + self.smooth
            sq_sum = torch.sum(y**2 + yhat**2, dim=self.dim, keepdim=True)
            denominator = torch.zeros(1, dtype=inputs.dtype).to(device=inputs.device)
            for d in range(0, self.depth):
                a = 2**d
                b = -(2.0 * a - 1.0)
                denominator = denominator + torch.reciprocal((a * sq_sum) + (b * tpl) + self.smooth)

            return numerator * denominator * scale

        l1 = tanimoto(targets, inputs)
        l2 = tanimoto(1.0 - targets, 1.0 - inputs)
        score = (l1 + l2) * 0.5

        return score


class TanimotoDist(torch.nn.Module):
    """Tanimoto distance

    Reference:
        https://github.com/sentinel-hub/eo-flow/blob/master/eoflow/models/losses.py

    MIT License

    Copyright (c) 2017-2020 Matej Aleksandrov, Matej Batič, Matic Lubej, Grega Milčinski (Sinergise)
    Copyright (c) 2017-2020 Devis Peressutti, Jernej Puc, Anže Zupanc, Lojze Žust, Jovan Višnjić (Sinergise)
    """
    def __init__(
        self,
        smooth: float = 1e-5,
        weight: T.Optional[torch.Tensor] = None,
        dim: T.Union[int, T.Sequence[int]] = 0
    ):
        super(TanimotoDist, self).__init__()

        self.smooth = smooth
        self.weight = weight
        self.dim = dim

    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """Performs a single forward pass

        Args:
            inputs: Predictions from model (probabilities, logits or labels).
            targets: Ground truth values.

        Returns:
            Tanimoto distance loss (float)
        """
        tpl = torch.sum(targets * inputs, dim=self.dim, keepdim=True)
        sq_sum = torch.sum(targets**2 + inputs**2, dim=self.dim, keepdim=True)
        numerator = tpl + self.smooth
        denominator = (sq_sum - tpl) + self.smooth
        tanimoto = numerator / denominator

        return tanimoto.mean()


class FractalAttention(torch.nn.Module):
    """Fractal Tanimoto Attention Layer (FracTAL)

    Adapted from publications and source code below:

        CSIRO BSTD/MIT LICENSE

        Redistribution and use in source and binary forms, with or without modification, are permitted provided that
        the following conditions are met:

        1. Redistributions of source code must retain the above copyright notice, this list of conditions and the
            following disclaimer.
        2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and
            the following disclaimer in the documentation and/or other materials provided with the distribution.
        3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or
            promote products derived from this software without specific prior written permission.

        THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
        INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
        DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
        SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
        SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
        WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE
        USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

        Reference:
            https://arxiv.org/pdf/2009.02062.pdf
            https://github.com/waldnerf/decode/blob/9e922a2082e570e248eaee10f7a1f2f0bd852b42/FracTAL_ResUNet/nn/units/fractal_resnet.py
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        depth: int = 5
    ):
        super(FractalAttention, self).__init__()

        self.gamma = torch.nn.Parameter(torch.ones(1))

        self.query = torch.nn.Sequential(
            ConvBlock2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
                add_activation=False
            ),
            torch.nn.Sigmoid()
        )
        self.key = torch.nn.Sequential(
            ConvBlock2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
                add_activation=False
            ),
            torch.nn.Sigmoid()
        )
        self.values = torch.nn.Sequential(
            ConvBlock2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
                add_activation=False
            ),
            torch.nn.Sigmoid()
        )

        self.spatial_sim = TanimotoComplement(depth=depth, dim=1)
        self.channel_sim = TanimotoComplement(depth=depth, dim=[2, 3])
        self.norm = torch.nn.BatchNorm2d(out_channels)

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        q = self.query(g)
        k = self.key(g)
        v = self.values(g)

        attention_spatial = self.spatial_sim(q, k)
        v_spatial = attention_spatial * v

        attention_channel = self.channel_sim(q, k)
        v_channel = attention_channel * v

        attention = (v_spatial + v_channel) * 0.5
        attention = self.norm(attention)

        # 1 + γA
        attention = 1.0 + self.gamma * attention

        return attention * x


class ChannelAttention(torch.nn.Module):
    """Residual Channel Attention Block

    References:
        https://arxiv.org/abs/1807.02758
        https://github.com/yjn870/RCAN-pytorch
    """
    def __init__(self, channels: int):
        super(ChannelAttention, self).__init__()

        self.module = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Conv2d(channels, channels, kernel_size=1, padding=0),
            torch.nn.LeakyReLU(inplace=True),
            torch.nn.Conv2d(channels, channels, kernel_size=1, padding=0),
            torch.nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.module(x)


class AtrousSpatialPyramid(torch.nn.Module):
    """Atrous spatial pyramid pooling
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dilations: T.Sequence[int]
    ):
        super(AtrousSpatialPyramid, self).__init__()

        self.layers = [
            ConvBlock2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=dilation,
                dilation=dilation
            ) for dilation in dilations
        ]
        final_in_channels = out_channels * len(dilations)
        self.final = ConvBlock2d(
            in_channels=final_in_channels,
            out_channels=out_channels,
            kernel_size=1,
            padding=0
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.cat([layer(x) for layer in self.layers], dim=1)
        h = self.final(h)

        return h


class DoubleConv(torch.nn.Module):
    """A double convolution layer
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: int,
        dilation: int,
        depthwise_conv: bool = False
    ):
        super(DoubleConv, self).__init__()

        convolution = DepthwiseConvBlock2d if depthwise_conv else torch.nn.Conv2d

        self.seq = torch.nn.Sequential(
            convolution(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation
            ),
            convolution(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.seq(x)


class PoolConvSingle(torch.nn.Module):
    """Max pooling followed by convolution
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        pool_size: int = 2
    ):
        super(PoolConvSingle, self).__init__()

        self.seq = torch.nn.Sequential(
            torch.nn.MaxPool2d(pool_size),
            ConvBlock2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.seq(x)


class PoolConv(torch.nn.Module):
    """Max pooling with (optional) dropout
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        pool_size: int = 2,
        dropout: T.Optional[float] = None,
        depthwise_conv: T.Optional[bool] = False
    ):
        super(PoolConv, self).__init__()

        layers = [torch.nn.MaxPool2d(pool_size)]
        if dropout is not None:
            layers += [torch.nn.Dropout(dropout)]
        layers += [
            DoubleConv(
                in_channels,
                out_channels,
                depthwise_conv=depthwise_conv
            )
        ]
        self.seq = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.seq(x)


class ResidualConvInit(torch.nn.Module):
    """A residual convolution layer
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int
    ):
        super(ResidualConvInit, self).__init__()

        self.seq = ConvBlock2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1
        )
        self.skip = ConvBlock2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            add_activation=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.seq(x) + self.skip(x)


class ResidualConv(torch.nn.Module):
    """A residual convolution layer with (optional) attention
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        init_conv: bool = False,
        channel_attention: bool = False,
        dilations: T.List[int] = None
    ):
        super(ResidualConv, self).__init__()

        init_in_channels = in_channels

        layers = []
        if init_conv:
            layers += [
                ConvBlock2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    padding=1
                )
            ]
            in_channels = out_channels

        layers += [
            ResBlock2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1
            )
        ]
        if dilations is not None:
            for dilation in dilations:
                layers += [
                    ResBlock2d(
                        in_channels=out_channels,
                        out_channels=out_channels,
                        kernel_size=3,
                        padding=dilation,
                        dilation=dilation
                    )
                ]

        if channel_attention:
            layers += [ChannelAttention(channels=out_channels)]

        self.seq = torch.nn.Sequential(*layers)
        self.skip = ConvBlock2d(
            in_channels=init_in_channels,
            out_channels=out_channels,
            kernel_size=1,
            add_activation=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.seq(x) + self.skip(x)

        return out


class ResidualConvRCAB(torch.nn.Module):
    """A group of residual convolution layers with (optional) RCAB
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        channel_attention: bool = False,
        res_blocks: int = 2,
        dilations: T.List[int] = None
    ):
        super(ResidualConvRCAB, self).__init__()

        layers = [
            torch.nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1
            )
        ]

        for __ in range(0, res_blocks):
            layers += [
                ResidualConv(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    channel_attention=channel_attention,
                    dilations=dilations
                )
            ]
        layers += [
            torch.nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1
            )
        ]

        self.seq = torch.nn.Sequential(*layers)
        self.expand = ConvBlock2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            add_activation=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.seq(x) + self.expand(x)


class PoolResidualConv(torch.nn.Module):
    """Max pooling followed by a residual convolution
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        pool_size: int = 2,
        dropout: T.Optional[float] = None,
        dilations: T.List[int] = None,
        channel_attention: bool = False,
        res_blocks: int = 0
    ):
        super(PoolResidualConv, self).__init__()

        layers = [torch.nn.MaxPool2d(pool_size)]

        if dropout is not None:
            assert isinstance(dropout, float), 'The dropout arg must be a float.'
            layers += [torch.nn.Dropout(dropout)]

        if res_blocks > 0:
            layers += [
                ResidualConvRCAB(
                    in_channels,
                    out_channels,
                    channel_attention=channel_attention,
                    res_blocks=res_blocks
                ),
                torch.nn.Conv2d(
                    out_channels,
                    out_channels,
                    kernel_size=3,
                    padding=1
                )
            ]

        else:
            layers += [
                ResidualConv(
                    in_channels,
                    out_channels,
                    channel_attention=channel_attention,
                    dilations=dilations
                )
            ]

        self.seq = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.seq(x)


class SingleConv(torch.nn.Module):
    """A single convolution layer
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int
    ):
        super(SingleConv, self).__init__()

        self.seq = ConvBlock2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.seq(x)


class TemporalConv(torch.nn.Module):
    """A temporal convolution layer
    """
    def __init__(
        self,
        in_channels: int,
        in_time: int,
        out_channels: int
    ):
        super(TemporalConv, self).__init__()

        layers = [
            ConvBlock3d(
                in_channels=in_channels,
                in_time=in_time,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
                squeeze=False
            ),
            ConvBlock3d(
                in_channels=out_channels,
                in_time=in_time,
                out_channels=out_channels,
                kernel_size=3,
                padding=2,
                dilation=2,
                squeeze=True
            )
        ]
        self.seq = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.seq(x)
