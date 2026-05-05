"""
model.py
========
Lightweight 3D U-Net for proposal generation
"""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_c, out_c, 3, padding=1),
            nn.InstanceNorm3d(out_c),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(out_c, out_c, 3, padding=1),
            nn.InstanceNorm3d(out_c),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class Small3DUNet(nn.Module):
    def __init__(self, in_channels=2, base=16):
        super().__init__()

        self.enc1 = ConvBlock(in_channels, base)
        self.pool1 = nn.MaxPool3d(2)

        self.enc2 = ConvBlock(base, base * 2)
        self.pool2 = nn.MaxPool3d(2)

        self.bottleneck = ConvBlock(base * 2, base * 4)

        self.up2 = nn.ConvTranspose3d(base * 4, base * 2, 2, stride=2)
        self.dec2 = ConvBlock(base * 4, base * 2)

        self.up1 = nn.ConvTranspose3d(base * 2, base, 2, stride=2)
        self.dec1 = ConvBlock(base * 2, base)

        self.out = nn.Conv3d(base, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))

        b = self.bottleneck(self.pool2(e2))

        d2 = self.up2(b)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return torch.sigmoid(self.out(d1))
