import numpy as np
import torch
from typing import Tuple, Union
import torch.nn as nn
from convolutions import convolution
from monai.networks.layers.utils import get_act_layer, get_norm_layer

def get_conv_layer(
    spatial_dims: int, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, bias: bool = False
):

    return convolution(
        spatial_dims, in_channels, out_channels, strides=stride, kernel_size=kernel_size, bias=bias, conv_only=True
    )



class ResidualBlock(nn.Module):

    def __init__(self, 
                 spatial_dims: int,
                 in_channels: int,
                 norm: Union[Tuple, str] = ("GROUP", {"num_groups": 2}),
                 kernel_size: int = 3,
                 act: Union[Tuple, str] = ("RELU", {"inplace": True})) -> None:


                super().__init__()
                if kernel_size % 2 != 1:
                    raise AssertionError("Kernel size should be an odd number.")
                self.norm = get_norm_layer(name = norm, spatial_dims = spatial_dims, channels= in_channels)
                self.activation = get_act_layer(act)
                self.convolution = get_conv_layer(spatial_dims=spatial_dims, in_channels=in_channels, out_channels=in_channels)
                self.bottleneck = nn.Sequential(get_conv_layer(spatial_dims=spatial_dims, in_channels=in_channels,
                                                                  out_channels = int(in_channels/4), kernel_size = 1),
                                                 get_act_layer(act),

                                                get_conv_layer(spatial_dims = spatial_dims, in_channels = int(in_channels/4),
                                                                 out_channels = int(in_channels/4)),
                                                 get_act_layer(act),

                                                get_conv_layer(spatial_dims = spatial_dims, in_channels = int(in_channels/4), 
                                                                 out_channels = in_channels, kernel_size = 1))
            

    def forward_bottleneck(self, x):
          identity = x

          x = self.norm(x)
          x = self.bottleneck(x)

          x += identity
          x = self.activation(x)
          return x

    def forward(self, x):
        identity = x

        x = self.norm(x)
        x = self.convolution(x)
        x = self.activation(x)

        x = self.norm(x)
        x = self.convolution(x)
        
        x += identity
        x = self.activation(x)

        return x




               



