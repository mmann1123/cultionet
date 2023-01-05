import typing as T

from . import model_utils
from .base_layers import ConvBlock2d
from .nunet import UNet3Psi, ResUNet3Psi
from .inception import InceptionNet

import torch
from torch_geometric.data import Data


class CropTypeFinal(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        out_classes: int
    ):
        super(CropTypeFinal, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.out_classes = out_classes

        self.conv1 = ConvBlock2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            padding=0,
            activation_type='ReLU'
        )
        layers1 = [
            ConvBlock2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
                activation_type='ReLU'
            ),
            torch.nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            torch.nn.BatchNorm2d(out_channels)
        ]
        self.seq = torch.nn.Sequential(*layers1)

        layers_final = [
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(
                out_channels,
                out_classes,
                kernel_size=1,
                padding=0
            )
        ]
        self.final = torch.nn.Sequential(*layers_final)

    def forward(
        self, x: torch.Tensor, crop_type_star: torch.Tensor
    ) -> torch.Tensor:
        out1 = self.conv1(x)
        out = self.seq(out1)
        out = out + out1
        out = self.final(out)
        out = out + crop_type_star

        return out


class CultioNet(torch.nn.Module):
    """The cultionet model framework

    Args:
        ds_features (int): The total number of dataset features (bands x time).
        ds_time_features (int): The number of dataset time features in each band/channel.
        filters (int): The number of output filters for each stream.
        star_rnn_hidden_dim (int): The number of hidden features for the ConvSTAR layer.
        star_rnn_n_layers (int): The number of ConvSTAR layers.
        num_classes (int): The number of output classes.
    """
    def __init__(
        self,
        ds_features: int,
        ds_time_features: int,
        filters: int = 64,
        star_rnn_hidden_dim: int = 64,
        star_rnn_n_layers: int = 4,
        num_classes: int = 2,
        model_type: str = 'UNet3Psi'
    ):
        super(CultioNet, self).__init__()

        # Total number of features (time x bands/indices/channels)
        self.ds_num_features = ds_features
        # Total number of time features
        self.ds_num_time = ds_time_features
        # Total number of bands
        self.ds_num_bands = int(self.ds_num_features / self.ds_num_time)
        self.filters = filters
        self.num_classes = num_classes

        self.gc = model_utils.GraphToConv()
        self.cg = model_utils.ConvToGraph()

        # Star RNN layer crop classes
        num_classes_l2 = 0
        if self.num_classes > 2:
            # Crop-type classes
            # Loss on background:0, crop-type:1, edge:2
            num_classes_l2 = 3
            # Loss on background:0, crop-type:N
            num_classes_last = self.num_classes
            out_mask_channels = self.num_classes
            base_in_channels = star_rnn_hidden_dim * 2
        else:
            # Loss on background:0, crop-type:1, edge:2
            num_classes_last = 3
            out_mask_channels = 2
            base_in_channels = star_rnn_hidden_dim

        self.inception_model = InceptionNet(
            in_channels=self.ds_num_bands,
            out_channels=star_rnn_hidden_dim,
            num_classes_last=num_classes_last
        )
        if model_type == 'UNet3Psi':
            self.mask_model = UNet3Psi(
                in_channels=base_in_channels,
                out_dist_channels=1,
                out_edge_channels=2,
                out_mask_channels=2,
                init_filter=self.filters
            )
        elif model_type == 'ResUNet3Psi':
            self.mask_model = ResUNet3Psi(
                in_channels=base_in_channels,
                out_dist_channels=1,
                out_edge_channels=2,
                out_mask_channels=out_mask_channels,
                init_filter=self.filters,
                attention=True,
                attention_weights='gate'
            )
        else:
            raise NameError('Model type not supported.')

    def forward(
        self, data: Data
    ) -> T.Dict[str, torch.Tensor]:
        height = int(data.height) if data.batch is None else int(data.height[0])
        width = int(data.width) if data.batch is None else int(data.width[0])
        batch_size = 1 if data.batch is None else data.batch.unique().size(0)

        time_stream = self.gc(
            data.x, batch_size, height, width
        )
        # Reshape from (B x C x H x W) -> (B x C x T x H x W)
        # nbatch, ntime, height, width
        nbatch, __, height, width = time_stream.shape
        time_stream = time_stream.reshape(
            nbatch, self.ds_num_bands, self.ds_num_time, height, width
        )
        # (1) InceptionNet
        # Crop/Non-crop and Crop types
        if self.num_classes > 2:
            logits_time_h, logits_time_l2, logits_time_last = self.inception_model(time_stream)
            logits_time_l2 = self.cg(logits_time_l2)
        else:
            logits_time_h, logits_time_last = self.inception_model(time_stream)

        logits_time_h = self.cg(logits_time_h)
        logits_time_last = self.cg(logits_time_last)

        # (3) Main stream
        logits = self.mask_model(
            self.gc(
                logits_time_h, batch_size, height, width
            )
        )
        logits_distance = self.cg(logits['dist'])
        logits_edges = self.cg(logits['edge'])
        logits_crop = self.cg(logits['mask'])

        out = {
            'dist': logits_distance,
            'edge': logits_edges,
            'crop': None,
            'crop_star': None,
            'crop_type_star': None,
            'crop_type': None
        }
        if self.num_classes > 2:
            # Crop|non-crop plus edge
            out['crop_star'] = logits_time_l2
            # Crop-type
            out['crop_type_star'] = logits_time_last
            # Crop-type (final)
            out['crop_type'] = logits_crop
        else:
            out['crop'] = logits_crop
            # Crop|non-crop plus edge
            out['crop_star'] = logits_time_last

        return out
