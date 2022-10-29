import typing as T

from . import model_utils
from .nunet import TemporalNestedUNet2
from .convstar import StarRNN
from .crop import CropFinal
from .regression import RegressionLayer
from .tcn import GraphTransformer

import torch
from torch_geometric.data import Data


class CultioGraphNet(torch.nn.Module):
    """The cultionet graph network model framework

    Args:
        ds_features (int): The total number of dataset features (bands x time).
        ds_time_features (int): The number of dataset time features in each band/channel.
        filters (int): The number of output filters for each stream.
        num_classes (int): The number of output classes.
        dropout (Optional[float]): The dropout fraction for the transformer stream.
    """
    def __init__(
        self,
        ds_features: int,
        ds_time_features: int,
        filters: int = 64,
        star_rnn_hidden_dim: int = 64,
        star_rnn_n_layers: int = 3,
        num_classes: int = 2,
        dropout: T.Optional[float] = 0.1
    ):
        super(CultioGraphNet, self).__init__()

        # Total number of features (time x bands/indices/channels)
        self.ds_num_features = ds_features
        # Total number of time features
        self.ds_num_time = ds_time_features
        # Total number of bands
        self.ds_num_bands = int(self.ds_num_features / self.ds_num_time)
        self.filters = filters
        num_distances = 2
        num_index_streams = 1
        # Temporal Convolution Network outputs
        tcn_heads = 2
        tcn_out_channels = self.filters
        # Temporal UNet++
        nunet_out_channels = self.filters
        # RNN stream = 2 + num_classes
        rnn_stream_local_out_channels = star_rnn_hidden_dim
        self.crop_type_layer = True if num_classes > 2 else False
        rnn_stream_out_channels = star_rnn_hidden_dim if not self.crop_type_layer else 0
        base_in_channels = (
            tcn_out_channels
            + nunet_out_channels
            + rnn_stream_local_out_channels
            + rnn_stream_out_channels
        )

        self.gc = model_utils.GraphToConv()
        self.cg = model_utils.ConvToGraph()

        self.transformer = GraphTransformer(
            num_features=self.ds_num_features,
            in_channels=self.ds_num_time,
            out_channels=self.filters,
            heads=tcn_heads,
            dropout=dropout
        )
        # Nested UNet (+self.filters x self.ds_num_bands)
        self.nunet = TemporalNestedUNet2(
            num_features=self.ds_num_features,
            in_channels=self.ds_num_time,
            mid_channels=self.filters,
            out_channels=nunet_out_channels
        )
        # Star RNN layer
        self.star_rnn = StarRNN(
            input_dim=self.ds_num_bands,
            hidden_dim=star_rnn_hidden_dim,
            n_layers=star_rnn_n_layers
        )
        # Boundary distance orientations (+1)
        self.dist_layer_ori = RegressionLayer(
            in_channels=base_in_channels,
            mid_channels=self.filters,
            out_channels=1
        )
        # Boundary distances (+1)
        self.dist_layer = RegressionLayer(
            in_channels=base_in_channels+1,
            mid_channels=self.filters,
            out_channels=1
        )
        # Edges (+2)
        self.edge_layer = CropFinal(
            in_channels=base_in_channels+num_distances,
            mid_channels=self.filters,
            out_channels=2
        )
        # Crops (+2)
        self.crop_layer = CropFinal(
            in_channels=base_in_channels+num_distances+2,
            mid_channels=self.filters,
            out_channels=2
        )

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(
        self, data: Data
    ) -> T.Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, T.Union[torch.Tensor, None]
    ]:
        height = int(data.height) if data.batch is None else int(data.height[0])
        width = int(data.width) if data.batch is None else int(data.width[0])
        batch_size = 1 if data.batch is None else data.batch.unique().size(0)

        transformer_stream = self.transformer(
            data.x,
            data.edge_index,
            data.edge_attrs
        )
        # Nested UNet on each band time series
        nunet_stream = self.nunet(
            data.x,
            data.edge_index,
            data.edge_attrs[:, 1],
            data.batch,
            height,
            width
        )
        # RNN ConvStar
        star_stream = self.gc(
            data.x, batch_size, height, width
        )
        # nbatch, ntime, height, width
        nbatch, __, height, width = star_stream.shape
        # Reshape from (B x C x H x W) -> (B x C x T x H x W)
        star_stream = star_stream.reshape(
            nbatch, self.ds_num_bands, self.ds_num_time, height, width
        )
        # Crop/Non-crop and Crop types
        star_stream_local_2, star_stream = self.star_rnn(star_stream)
        star_stream_local_2 = self.cg(star_stream_local_2)
        star_stream = self.cg(star_stream)
        # Concatenate time series streams
        h = torch.cat(
            [
                transformer_stream,
                nunet_stream,
                star_stream_local_2
            ],
            dim=1
        )
        if not self.crop_type_layer:
            h = torch.cat([h, star_stream], dim=1)
            star_stream = None

        # Estimate distance orientations
        logits_distances_ori = self.dist_layer_ori(
            h,
            data.edge_index,
            data.edge_attrs,
            data.batch,
            height,
            width
        )
        # Concatenate streams + distance orientations
        h = torch.cat([h, logits_distances_ori], dim=1)

        # Estimate distance from edges
        logits_distances = self.dist_layer(
            h,
            data.edge_index,
            data.edge_attrs,
            data.batch,
            height,
            width
        )
        # Concatenate streams + distance orientations + distances
        h = torch.cat([h, logits_distances], dim=1)

        # Estimate edges
        logits_edges = self.edge_layer(
            h,
            data.edge_index,
            data.edge_attrs,
            data.batch,
            height,
            width
        )
        # Concatenate streams + distance orientations + distances + edges
        h = torch.cat([h, logits_edges], dim=1)
        # Estimate crop/non-crop
        logits_crop = self.crop_layer(
            h,
            data.edge_index,
            data.edge_attrs,
            data.batch,
            height,
            width
        )

        return (
            logits_distances_ori,
            logits_distances,
            logits_edges,
            logits_crop,
            star_stream
        )
