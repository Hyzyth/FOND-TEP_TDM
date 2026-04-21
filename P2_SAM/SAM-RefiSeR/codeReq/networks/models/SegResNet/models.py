
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from monai.networks.blocks.segresnet_block import get_conv_layer, get_upsample_layer
from monai.networks.layers.factories import Dropout
from monai.networks.layers.utils import get_act_layer, get_norm_layer
from monai.utils import UpsampleMode
from buillding_block import ResidualBlock

class SegResNet(nn.Module):

    
    def __init__(self, 
                 spatial_dims: int = 3,
                 init_kernels: int = 8,
                 in_channels: int =  4,
                 out_channels: int = 3,
                 dropout_prob: Optional[float] = 0.3,
                 act: Union[Tuple, str] = ("RELU", {"inplace": True}),
                 norm: Union[Tuple, str] = ("GROUP", {"num_groups": 4}),
                 norm_name: str = "",
                 num_groups: int = 4,
                 use_conv_final: bool = True,
                 blocks_down: tuple = (1, 2, 2, 4),
                 blocks_up: tuple = (1, 1, 1),
                 upsample_mode: Union[UpsampleMode, str] = UpsampleMode.NONTRAINABLE) -> None:

                 super().__init__()
                 if spatial_dims not in (2, 3):
                    raise ValueError("spatial dimension should either 2 or 3.")

                 self.spatial_dims = spatial_dims
                 self.init_filters = init_kernels
                 self.in_channels = in_channels
                 self.out_channels = out_channels
                 self.dropout_prob = dropout_prob
                 self.activation = act
                 self.activation_mode = get_act_layer(self.activation)
                 self.blocks_down = blocks_down
                 self.blocks_up = blocks_up

                 if norm_name:
                    if norm_name.lower() != "group":
                        raise ValueError(f"Deprecating option 'norm_name={norm_name}', please use 'norm' instead.")
                    norm = ("group", {"num_groups": num_groups})

                 self.norm = norm
                 self.upsample_mode = UpsampleMode(upsample_mode)
                 self.use_conv_final = use_conv_final
                 self.initial_conv = get_conv_layer(spatial_dims, in_channels, init_kernels, 
                                                    kernel_size = 3, stride = 1, bias = True)

                 self.down_layers = self._make_down_layers()
                 self.up_layers, self.up_samples = self._make_up_layers()
                 self.conv_final = self._make_final_conv(out_channels)

                 if dropout_prob is not None:
                    self.dropout = Dropout[Dropout.DROPOUT, spatial_dims](dropout_prob)
               
   
    def _make_down_layers(self):

        down_layers = nn.ModuleList()
        blocks_down, spatial_dims, filters, norm = (self.blocks_down, self.spatial_dims, self.init_filters, self.norm)
        for i, item in enumerate(blocks_down):
            layer_in_channels = filters * 2**i
            pre_conv = (get_conv_layer(spatial_dims, layer_in_channels // 2, layer_in_channels, stride=2) if i > 0 else nn.Identity())
            down_layer = nn.Sequential(pre_conv, *[ResidualBlock(spatial_dims, layer_in_channels, norm) for j in range(item)])
            down_layers.append(down_layer)
        return down_layers
    
    def _make_up_layers(self):
        up_layers, up_samples = nn.ModuleList(), nn.ModuleList()
        upsample_mode, blocks_up, spatial_dims, filters, norm = (
            self.upsample_mode,
            self.blocks_up,
            self.spatial_dims,
            self.init_filters,
            self.norm)

        n_up = len(blocks_up)
        for i in range(n_up):
            sample_in_channels = filters * 2 ** (n_up - i)
            up_layers.append(
                nn.Sequential(
                    *[
                        ResidualBlock(spatial_dims, sample_in_channels // 2, norm=norm, act=self.activation)
                        for _ in range(blocks_up[i])
                    ]))

            up_samples.append(
                nn.Sequential(
                    *[
                        get_conv_layer(spatial_dims, sample_in_channels, sample_in_channels // 2, kernel_size=1),
                        get_upsample_layer(spatial_dims, sample_in_channels // 2, upsample_mode=upsample_mode),
                    ]))
        
        return up_layers, up_samples
    
    def _make_final_conv(self, out_channels: int):
        return nn.Sequential(
            get_conv_layer(self.spatial_dims, self.init_filters, out_channels, kernel_size=1, bias=True),
            get_norm_layer(name=self.norm, spatial_dims=self.spatial_dims, channels=self.init_filters),
            self.activation_mode)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        x = self.initial_conv(x)
        if self.dropout_prob is not None:
            x = self.dropout(x)

        down_x = []

        for down in self.down_layers:
            x = down(x)
            down_x.append(x)

        return x, down_x

    def decode(self, x: torch.Tensor, down_x: List[torch.Tensor]) -> Tuple[torch.Tensor, list]:
        shapes = []
        for i, (up, upl) in enumerate(zip(self.up_samples, self.up_layers)):
            shapes.append(x.shape)
            shapes.append(down_x[i + 1].shape)
            x = up(x) + down_x[i + 1]
            x = upl(x)

        if self.use_conv_final:
            x = self.conv_final(x)

        return x,shapes


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, down_x = self.encode(x)
        down_x.reverse()
        x,shapes = self.decode(x, down_x)
        return x, shapes
