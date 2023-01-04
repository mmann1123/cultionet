import typing as T

from .base_layers import Permute, Transpose

import torch


class EncoderLayer(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_k: T.Optional[int] = None,
        d_v: T.Optional[int] = None,
        d_ff: T.Optional[int] = None,
        dropout: float = 0.1,
        activation: str = 'GELU'
    ):
        super(EncoderLayer, self).__init__()

        self.multihead_attn = torch.nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            kdim=d_k,
            vdim=d_v,
            batch_first=True
        )
        self.dropout = torch.nn.Dropout(dropout)
        self.batchnorm_attn = torch.nn.Sequential(
            Transpose(axis_order=(1, 2)),
            torch.nn.BatchNorm1d(d_model),
            Transpose(axis_order=(1, 2))
        )
        # Position-wise Feed-Forward
        self.linear_net = torch.nn.Sequential(
            torch.nn.Linear(d_model, d_ff),
            torch.nn.Dropout(dropout),
            getattr(torch.nn, activation)(),
            torch.nn.Linear(d_ff, d_model)
        )
        self.batchnorm_ffn = torch.nn.Sequential(
            Transpose(axis_order=(1, 2)),
            torch.nn.BatchNorm1d(d_model),
            Transpose(axis_order=(1, 2))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input dimensions -> B x T x Embed_dim
        # Output dimensions -> (B x T x Embed_dim,)
        attn, __ = self.multihead_attn(
            query=x,
            key=x,
            value=x,
            need_weights=False,
            average_attn_weights=True
        )
        x = x + self.dropout(attn)
        x = self.batchnorm_attn(x)
        lin = self.linear_net(x)
        x = x + self.dropout(lin)
        x = self.batchnorm_ffn(x)

        return x


class Encoder(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_k: int = None,
        d_v: int = None,
        d_ff: int = None,
        dropout: float = 0.1,
        activation: str = 'GELU',
        n_layers: int = 1
    ):
        super(Encoder, self).__init__()

        layers = [
            EncoderLayer(
                d_model,
                n_heads=n_heads,
                d_k=d_k,
                d_v=d_v,
                d_ff=d_ff,
                dropout=dropout,
                activation=activation
            ) for __ in range(n_layers)
        ]
        self.layers = torch.nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)

        return x


class TST(torch.nn.Module):
    """
    References:
        https://pytorch-lightning.readthedocs.io/en/stable/notebooks/course_UvA-DL/05-transformers-and-MH-attention.html
        https://github.com/timeseriesAI/tsai/blob/main/tsai/models/TST.py#L131

    Source repository:
        https://github.com/timeseriesAI/tsai

    Source license:
        Apache-2.0 license
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        seq_len: int,
        n_layers: int = 3,
        d_model: int = 128,
        n_heads: int = 8,
        d_ff: int = 256,
        d_k: T.Optional[int] = None,
        d_v: T.Optional[int] = None,
        dropout: float = 0.1,
        activation: str = 'GELU'
    ):
        super(TST, self).__init__()

        q_len = seq_len
        self.w_p = torch.nn.Conv3d(
            in_channels,
            d_model,
            kernel_size=3,
            padding=1
        )
        # Positional encoding
        w_pos = torch.empty((q_len, d_model))
        torch.nn.init.uniform_(w_pos, -0.02, 0.02)
        self.w_pos = torch.nn.Parameter(w_pos)

        self.dropout = torch.nn.Dropout()

        self.encoder = Encoder(
            d_model,
            n_heads,
            d_k=d_k,
            d_v=d_v,
            d_ff=d_ff,
            dropout=dropout,
            activation=activation,
            n_layers=n_layers
        )

        self.final = torch.nn.Sequential(
            getattr(torch.nn, activation)(),
            torch.nn.Conv2d(
                d_model,
                out_channels,
                kernel_size=3,
                padding=1
            )
        )

    def forward(self, x: torch.Tensor) -> T.Tuple[torch.Tensor, torch.Tensor]:
        # Input encoding
        u = self.w_p(x)
        n_batch, n_channels, n_time, height, width = u.shape

        # Positional encoding
        # u dim -> B x C (d_model) x T (q_len) x H x W
        # self.w_pos dim -> q_len x d_model
        u = u.permute(0, 3, 4, 2, 1).reshape(n_batch*height*width, n_time, n_channels)
        u = self.dropout(u + self.w_pos)

        # Encoder
        z = self.encoder(u)
        # Output dimensions -> B x C x T x H x W
        z = z.reshape(
            n_batch, height, width, n_time, n_channels
        ).permute(0, 4, 3, 1, 2).mean(dim=2)
        last = self.final(z)

        return z, last
