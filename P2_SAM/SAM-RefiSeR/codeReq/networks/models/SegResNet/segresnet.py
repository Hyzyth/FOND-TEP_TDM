from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections.abc import Sequence

from monai.networks.blocks.segresnet_block import ResBlock, get_conv_layer, get_upsample_layer
from monai.networks.layers.factories import Dropout
from monai.networks.layers.utils import get_act_layer, get_norm_layer
from monai.utils import UpsampleMode
from monai.losses import DiceLoss
from myCodeArchives.vffc import Bottleneck
from myCodeArchives.nnAdapts import norm_nd_class
from torch.utils.checkpoint import checkpoint
from torch.nn.utils import spectral_norm

# Edited SEGRESNET with the VFFC

class SegResNet(nn.Module):

    def __init__(
        self,
        spatial_dims: int = 3,
        init_filters: int = 8,
        in_channels: int = 1,
        out_channels: int = 2,
        dropout_prob: float | None = None,
        act: tuple | str = ("RELU", {"inplace": True}),
        norm: tuple | str = ("GROUP", {"num_groups": 8}),
        norm_name: str = "",
        num_groups: int = 8,
        use_conv_final: bool = True,
        blocks_down: tuple = (1, 2, 2, 4),
        blocks_up: tuple = (1, 1, 1),
        upsample_mode: UpsampleMode | str = UpsampleMode.NONTRAINABLE,
    ):
        super().__init__()

        if spatial_dims not in (2, 3):
            raise ValueError("`spatial_dims` can only be 2 or 3.")

        self.spatial_dims = spatial_dims
        self.init_filters = init_filters
        self.in_channels = in_channels
        self.blocks_down = blocks_down
        self.blocks_up = blocks_up
        self.dropout_prob = dropout_prob
        self.act = act  # input options
        self.act_mod = get_act_layer(act)
        if norm_name:
            if norm_name.lower() != "group":
                raise ValueError(f"Deprecating option 'norm_name={norm_name}', please use 'norm' instead.")
            norm = ("group", {"num_groups": num_groups})
        self.norm = norm
        self.upsample_mode = UpsampleMode(upsample_mode)
        self.use_conv_final = use_conv_final
        self.convInit = get_conv_layer(spatial_dims, in_channels, init_filters)
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
            pre_conv = (
                get_conv_layer(spatial_dims, layer_in_channels // 2, layer_in_channels, stride=2)
                if i > 0
                else nn.Identity()
            )
            down_layer = nn.Sequential(
                pre_conv, *[ResBlock(spatial_dims, layer_in_channels, norm=norm, act=self.act) for _ in range(item)]
            )
            down_layers.append(down_layer)
        return down_layers

    def _make_up_layers(self):
        up_layers, up_samples = nn.ModuleList(), nn.ModuleList()
        upsample_mode, blocks_up, spatial_dims, filters, norm = (
            self.upsample_mode,
            self.blocks_up,
            self.spatial_dims,
            self.init_filters,
            self.norm,
        )
        n_up = len(blocks_up)
        for i in range(n_up):
            sample_in_channels = filters * 2 ** (n_up - i)
            up_layers.append(
                nn.Sequential(
                    *[
                        ResBlock(spatial_dims, sample_in_channels // 2, norm=norm, act=self.act)
                        for _ in range(blocks_up[i])
                    ]
                )
            )
            up_samples.append(
                nn.Sequential(
                    *[
                        get_conv_layer(spatial_dims, sample_in_channels, sample_in_channels // 2, kernel_size=1),
                        get_upsample_layer(spatial_dims, sample_in_channels // 2, upsample_mode=upsample_mode),
                    ]
                )
            )
        return up_layers, up_samples

    def _make_final_conv(self, out_channels: int):
        return nn.Sequential(
            get_norm_layer(name=self.norm, spatial_dims=self.spatial_dims, channels=self.init_filters),
            self.act_mod,
            get_conv_layer(self.spatial_dims, self.init_filters, out_channels, kernel_size=1, bias=True),
        )
    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        x = self.convInit(x) 
        if self.dropout_prob is not None:
            x = self.dropout(x)

        down_x = []

        for down in self.down_layers:
            x = checkpoint(down, x, use_reentrant=False)
            down_x.append(x)

        return x, down_x

    def decode(self, x: torch.Tensor, down_x: list[torch.Tensor]) -> torch.Tensor:
        for i, (up, upl) in enumerate(zip(self.up_samples, self.up_layers)):
            x = up(x) + down_x[i + 1]
            x = upl(x)

        if self.use_conv_final:
            x = self.conv_final(x)

        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, down_x = self.encode(x)
        down_x.reverse() 

        x = self.decode(x, down_x)
        return x


class ResBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch, downsample=False):
        super().__init__()
        stride = 2 if downsample else 1
        self.conv1 = spectral_norm(nn.Conv3d(in_ch, out_ch, 3, stride=stride, padding=1))
        self.bn1   = nn.GroupNorm(num_groups=16, num_channels=out_ch)
        self.act1  = nn.LeakyReLU(0.2, inplace=False)

        self.conv2 = spectral_norm(nn.Conv3d(out_ch, out_ch, 3, padding=1))
        self.bn2   = nn.GroupNorm(num_groups=16, num_channels=out_ch)
        self.act2  = nn.LeakyReLU(0.2, inplace=False)

        if downsample or in_ch != out_ch:
            self.skip = spectral_norm(nn.Conv3d(in_ch, out_ch, 1, stride=stride))
        else:
            self.skip = nn.Identity()

    def forward(self, x):
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.act2(out + self.skip(x))



class FCDiscriminator3D(nn.Module):
    def __init__(self, in_channels: int, num_domains: int = 2):
        super().__init__()
        self.stem = nn.Sequential(
            spectral_norm(nn.Conv3d(in_channels, 64, 3, stride=1, padding=1)),
            nn.GroupNorm(num_groups=16, num_channels=64),
            nn.LeakyReLU(0.2, inplace=False),
        )
        self.layer1 = ResBlock3D(64, 128, downsample=True)
        self.layer2 = ResBlock3D(128, 256, downsample=True)
        self.layer3 = ResBlock3D(256, 512, downsample=True)
        self.layer4 = ResBlock3D(512, 512, downsample=True)

        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.dropout     = nn.Dropout(0.3)
        self.classifier  = nn.Sequential(
            spectral_norm(nn.Linear(256 + 512, 256)),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Dropout(0.3),
            spectral_norm(nn.Linear(256, num_domains))
        )

    def forward(self, x):
        x = self.stem(x)              # → [B,64,D,H,W]
        x = self.layer1(x)            # → [B,128,D/2,H/2,W/2]
        m = self.layer2(x)            # → [B,256,D/4,H/4,W/4]
        x = self.layer3(m)            # → [B,512,D/8,H/8,W/8]
        d = self.layer4(x)            # → [B,512,D/16,H/16,W/16]

        m_feat = self.global_pool(m).view(x.size(0), -1)  # [B,256]
        d_feat = self.global_pool(d).view(x.size(0), -1)  # [B,512]

        feat = torch.cat([m_feat, d_feat], dim=1)        # [B,768]
        feat = self.dropout(feat)

        return self.classifier(feat)                     # [B,num_domains]



###################################################


class GradReverseFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None
    
def grad_reverse(x, alpha = 1.0):
    return GradReverseFunction.apply(x, alpha)


class GRLSegResNet(SegResNet):

    def __init__(
        self,
        num_domains: int = 3,
        alpha: float = 1,
        *args,
        **kwargs
    ):
        

        super().__init__(*args, **kwargs)

        self.num_domains = num_domains
        self.alpha = alpha

        deep_feature_channels = 256
        self.domain_discriminator = FCDiscriminator3D(in_channels = deep_feature_channels, 
                                                      num_domains = num_domains, 
                                                      )

        self.seg_loss_fn = DiceLoss(sigmoid = True, to_onehot_y = False)
        self.domain_loss_fn = nn.CrossEntropyLoss()


    def forward(
        self,
        images: torch.Tensor,
        seg_labels: torch.Tensor | None = None,
        domain_labels: torch.Tensor | None = None,
        alpha: float | None = None
    ):


        if alpha is None:
            alpha = self.alpha

        x, down_x = self.encode(images)

        feat = x
        down_x.reverse()

        seg_logits = self.decode(x, down_x)

        if seg_labels is None and domain_labels is None:
            return seg_logits

        reversed_feat = grad_reverse(feat, alpha = alpha)
        domain_logits = self.domain_discriminator(reversed_feat)


        if seg_labels is not None:
            seg_loss = self.seg_loss_fn(seg_logits, seg_labels)
        else:
            seg_loss = torch.tensor(0.0, device=images.device, dtype=torch.float)

        if domain_labels is not None:
            domain_loss = self.domain_loss_fn(domain_logits, domain_labels.long())
        else:
            domain_loss = torch.tensor(0.0, device=images.device, dtype=torch.float)

        return seg_loss, domain_loss
    



class GraphGRLSegResNetBaseline(GRLSegResNet):

    def forward(
        self,
        images: torch.Tensor,
        seg_labels: torch.Tensor | None = None,
        domain_labels: torch.Tensor | None = None,
        alpha: float | None = None,
        return_features: bool = False
    ):
        if alpha is None:
            alpha = self.alpha

        x, down_x = self.encode(images)
        feat = x
        down_x.reverse()

        seg_logits = self.decode(x, down_x)

        if seg_labels is None and domain_labels is None:
            pooled_feat = F.adaptive_avg_pool3d(feat, (1, 1, 1)).view(feat.size(0), -1)
            return (seg_logits, pooled_feat) if return_features else seg_logits
                
        reversed_feat = grad_reverse(feat, alpha = alpha)
        
        domain_logits = self.domain_discriminator(reversed_feat)

        seg_loss = self.seg_loss_fn(seg_logits, seg_labels) if seg_labels is not None else torch.tensor(0.0, device = images.device)
        domain_loss = self.domain_loss_fn(domain_logits, domain_labels.long()) if domain_labels is not None else torch.tensor(0.0, device = images.device)

        if return_features:
            pooled_feat = F.adaptive_avg_pool3d(feat, (1, 1, 1)).view(feat.size(0), -1)
            return seg_loss, domain_loss, pooled_feat
        else:
            return seg_loss, domain_loss




class GraphGRLSegResNet(GRLSegResNet):

    def forward(
        self,
        images: torch.Tensor,
        seg_labels: torch.Tensor | None = None,
        domain_labels: torch.Tensor | None = None,
        alpha: float | None = None,
        return_features: bool = False
    ):
        if alpha is None:
            alpha = self.alpha

        x, down_x = self.encode(images)
        feat = x
        down_x.reverse()

        seg_logits = self.decode(x, down_x)

        if seg_labels is None and domain_labels is None:

            pooled_feat = F.adaptive_avg_pool3d(feat, (8, 8, 8)).view(feat.size(0) * 512, -1)
            return (seg_logits, pooled_feat) if return_features else seg_logits
                
        reversed_feat = grad_reverse(feat, alpha = alpha)
        
        domain_logits = self.domain_discriminator(reversed_feat)

        seg_loss = self.seg_loss_fn(seg_logits, seg_labels) if seg_labels is not None else torch.tensor(0.0, device = images.device)
        domain_loss = self.domain_loss_fn(domain_logits, domain_labels.long()) if domain_labels is not None else torch.tensor(0.0, device = images.device)

        if return_features:
            pooled_feat = F.adaptive_avg_pool3d(feat, (8, 8, 8)).view(feat.size(0) * 512, -1)
            return seg_loss, domain_loss, pooled_feat
        else:
            return seg_loss, domain_loss
        
