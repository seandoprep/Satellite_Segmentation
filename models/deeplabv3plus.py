import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import torch
import torch.nn as nn
import torchvision.models as models

from models.modules.deeplabv3plus_modules import ASPPModule, DecoderModule, SEModule
from typing import Any

"""
Model Code from https://github.com/mukund-ks/DeepLabV3Plus-PyTorch
"""

# Feature Extraction block
def feature_extract(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.LeakyReLU(0.01, inplace=True)
    )

# DeepLabv3+ Main Architecture
class DeepLabV3Plus(nn.Module):
    def __init__(self, in_channel, num_classes: int = 1) -> None:
        super(DeepLabV3Plus, self).__init__()

        self.feature_extractor = feature_extract(in_channel, 3)

        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

        self.backbone = nn.Sequential(*list(resnet.children())[:-2])

        in_channels = 1024
        out_channels = 256

        # Dilation Rates
        dilations = [6, 12, 18]

        # SE Module
        self.squeeze_excite = SEModule(channels=out_channels)

        # ASPP Module
        self.aspp = ASPPModule(in_channels, out_channels, dilations)

        # Decoder Module
        self.decoder = DecoderModule(out_channels, out_channels)

        # Upsampling with Bilinear Interpolation
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=4)

        # Dropout
        self.dropout = nn.Dropout(p=0.5)

        # Final 1x1 Convolution
        self.final_conv = nn.Conv2d(out_channels, num_classes, kernel_size=1)

        # Sigmoid Activation for Binary-Seg
        self.sigmoid = nn.Sigmoid()

        self.tanh = nn.Tanh()

    def forward(self, x: Any) -> Any:
        # DeepLabV3+ Forward Pass
        x = self.feature_extractor(x) 

        # Getting Low-Level Features
        x_low = self.backbone[:-3](x)
        x_low = self.squeeze_excite(x_low)

        # Getting Image Features from Backbone
        x = self.backbone[:-1](x)

        # ASPP forward pass - High-Level Features
        x = self.aspp(x)

        # Upsampling High-Level Features
        x = self.upsample(x)
        x = self.dropout(x)

        # Decoder forward pass - Concatenating Features
        x = self.decoder(x, x_low)

        # Upsampling Concatenated Features from Decoder
        x = self.upsample(x)

        # Final 1x1 Convolution for Binary-Segmentation
        x = self.final_conv(x)
        # x = self.sigmoid(x)
        x = self.tanh(x)
        normalized_x = (x + 1) * 0.5

        return normalized_x