
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn


from monai.networks.blocks import ADN
from monai.networks.layers.convutils import same_padding, stride_minus_kernel_padding
from monai.networks.layers.factories import Conv


class convolution(nn.Sequential):


    def __init__(self, 
                spatial_dims: int,
                in_channels:  int,
                out_channels: int,
                kernel_size: Union[Sequence[int], int] = 3,
                strides: Union[Sequence[int], int] = 1,
                adn_ordering: str = "NDA",
                act: Optional[Union[tuple, str]] = "PRELU",
                dropout: Optional[Union[Tuple, str, float]] = None,
                norm: Optional[Union[Tuple, str]] = "INSTANCE",
                dropout_dim: Optional[int] = 1,
                dilation: Union[Sequence[int], int] = 1,
                groups: int = 1,
                bias: bool = True,
                conv_only: bool = False,
                is_transposed: bool = False,
                padding: Optional[Union[Sequence[int], int]] = None,
                output_padding: Optional[Union[Sequence[int], int]] = None) -> None:

                super().__init__()
                self.spatial_dims = spatial_dims
                self.in_channels = in_channels
                self.out_channels = out_channels
                self.is_transposed = is_transposed
                if padding is None:
                    padding = same_padding(kernel_size, dilation)
                conv_type = Conv[Conv.CONVTRANS if is_transposed else Conv.CONV, self.spatial_dims]

                conv: nn.Module
                if is_transposed:
                    if output_padding is None:
                        output_padding = stride_minus_kernel_padding(1, strides)
                    conv = conv_type(
                        in_channels,
                        out_channels,
                        kernel_size=kernel_size,
                        stride=strides,
                        padding=padding,
                        output_padding=output_padding,
                        groups=groups,
                        bias=bias,
                        dilation=dilation,)
                else:
                    conv = conv_type(
                        in_channels,
                        out_channels,
                        kernel_size=kernel_size,
                        stride=strides,
                        padding=padding,
                        dilation=dilation,
                        groups=groups,
                        bias=bias,)
                self.add_module("conv", conv)

                if conv_only:
                    return
                if act is None and norm is None and dropout is None:
                    return
                self.add_module(
                    "adn",
                    ADN(
                        ordering=adn_ordering,
                        in_channels=out_channels,
                        act=act,
                        norm=norm,
                        norm_dim=self.spatial_dims,
                        dropout=dropout,
                        dropout_dim=dropout_dim,),)

                
     
                                  