from . import model_utils

import torch
from torch_geometric import nn


class RefineConv(torch.nn.Module):
    """A refinement convolution layer
    """
    def __init__(
        self,
        in_channels: int,
        mid_channels: int,
        out_channels: int,
        dropout: float = 0.1
    ):
        super(RefineConv, self).__init__()

        self.gc = model_utils.GraphToConv()
        self.cg = model_utils.ConvToGraph()

        conv2d_1 = torch.nn.Conv2d(
            in_channels, mid_channels, kernel_size=1, padding=0, bias=False, padding_mode='replicate'
        )
        conv2d_2 = torch.nn.Conv2d(
            mid_channels, mid_channels, kernel_size=3, padding=1, bias=False, padding_mode='replicate'
        )
        conv2d_3 = torch.nn.Conv2d(
            mid_channels, out_channels, kernel_size=1, padding=0, bias=False, padding_mode='replicate'
        )

        self.seq = nn.Sequential(
            'x',
            [
                (conv2d_1, 'x -> h'),
                (torch.nn.BatchNorm2d(mid_channels), 'h -> h'),
                (torch.nn.ELU(alpha=0.1, inplace=False), 'h -> h'),
                (conv2d_2, 'h -> h'),
                (torch.nn.BatchNorm2d(mid_channels), 'h -> h'),
                (torch.nn.ELU(alpha=0.1, inplace=False), 'h -> h'),
                (torch.nn.Dropout(dropout), 'h -> h'),
                (conv2d_2, 'h -> h'),
                (torch.nn.BatchNorm2d(mid_channels), 'h -> h'),
                (torch.nn.ReLU(inplace=False), 'h -> h'),
                (torch.nn.Dropout(dropout), 'h -> h'),
                (conv2d_3, 'h -> h')
            ]
        )

    def add_res(self, h1: torch.Tensor, h2: torch.Tensor) -> torch.Tensor:
        return h1 + h2

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(
        self,
        x: torch.Tensor,
        batch_size: torch.Tensor,
        height: int,
        width: int
    ) -> torch.Tensor:
        x = self.gc(
            x,
            batch_size,
            height,
            width
        )
        x = self.seq(x)

        return self.cg(x)
