import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import torch
import torch.nn as nn

'''
Model Code from https://github.com/usuyama/pytorch-unet
'''

# Double_convolution block 
def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )


# U-Net main architecture
class UNet(nn.Module):
    def __init__(self, in_channel: int = 3, num_classes: int = 1) -> None:
        super(UNet, self).__init__()
        self.feature_extractor = double_conv(in_channel, 3)
        self.dconv_down1 = double_conv(3, 64)
        self.dconv_down2 = double_conv(64, 128)
        self.dconv_down3 = double_conv(128, 256)
        self.dconv_down4 = double_conv(256, 512)

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)        

        self.dconv_up3 = double_conv(256 + 512, 256)
        self.dconv_up2 = double_conv(128 + 256, 128)
        self.dconv_up1 = double_conv(128 + 64, 64)

        self.conv_last = nn.Conv2d(64, num_classes, 1)

    def forward(self, x):
        x = self.feature_extractor(x) # n, 256, 256 -> 3, 256, 256 
        
        conv1 = self.dconv_down1(x) # 3, 256, 256 -> 64, 256, 256
        x = self.maxpool(conv1)  # 64, 256, 256 -> 64, 128, 128

        conv2 = self.dconv_down2(x) # 64, 128, 128 -> 128, 128, 128
        x = self.maxpool(conv2)  # 128, 128, 128 -> 128, 64, 64
        
        conv3 = self.dconv_down3(x)  # 128, 64, 64 -> 256, 64, 64
        x = self.maxpool(conv3)  # 256, 64, 64 -> 256, 32, 32

        x = self.dconv_down4(x)  # 256, 32, 32 -> 512, 32, 32

        x = self.upsample(x)  #  512, 32, 32 -> 512, 64, 64         
        x = torch.cat([x, conv3], dim=1)  # (512 + 128), 64, 64

        x = self.dconv_up3(x)
        x = self.upsample(x)        
        x = torch.cat([x, conv2], dim=1)       

        x = self.dconv_up2(x)
        x = self.upsample(x)        
        x = torch.cat([x, conv1], dim=1)   

        x = self.dconv_up1(x)

        out = self.conv_last(x)

        return out
    