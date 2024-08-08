import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import torch
import torch.nn as nn

from models.modules.mdoaunet_modules import *

"""
Model Code from https://github.com/Jichao-Wang/MDOAU-net/blob/main/MDOAU_net.py
"""


class MDOAU_net(nn.Module):
    # Fused multi-scaled convolution block
    def __init__(self, in_channel=1, num_classes=1):
        super(MDOAU_net, self).__init__()
        # offset_convolution()
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.multi_scale_1 = multi_scaled_dilation_conv_block(in_channel, 16, kernel_size=3, dilation=1)
        self.multi_scale_2 = multi_scaled_dilation_conv_block(in_channel, 16, kernel_size=5, dilation=1)
        self.multi_scale_3 = multi_scaled_dilation_conv_block(in_channel, 16, kernel_size=7, dilation=2)
        self.multi_scale_4 = multi_scaled_dilation_conv_block(in_channel, 16, kernel_size=11, dilation=2)
        self.multi_scale_5 = multi_scaled_dilation_conv_block(in_channel, 16, kernel_size=15, dilation=3)

        self.Conv1 = conv_block(ch_in=16 * 5, ch_out=8)
        self.Conv2 = conv_block(ch_in=8, ch_out=16)
        self.Conv3 = conv_block(ch_in=16, ch_out=32)
        self.Conv4 = conv_block(ch_in=32, ch_out=64)
        self.Conv5 = conv_block(ch_in=64, ch_out=128)

        self.o1 = offset_convolution(ch_in=8, ch_out=8)
        self.o2 = offset_convolution(ch_in=16, ch_out=16)
        self.o3 = offset_convolution(ch_in=32, ch_out=32)
        self.o4 = offset_convolution(ch_in=64, ch_out=64)

        self.Up5 = up_conv(ch_in=128, ch_out=64)
        self.Att5 = Attention_block(F_g=64, F_l=64, F_int=32)
        self.Up_conv5 = conv_block(ch_in=128, ch_out=64)

        self.Up4 = up_conv(ch_in=64, ch_out=32)
        self.Att4 = Attention_block(F_g=32, F_l=32, F_int=16)
        self.Up_conv4 = conv_block(ch_in=64, ch_out=32)

        self.Up3 = up_conv(ch_in=32, ch_out=16)
        self.Att3 = Attention_block(F_g=16, F_l=16, F_int=8)
        self.Up_conv3 = conv_block(ch_in=32, ch_out=16)

        self.Up2 = up_conv(ch_in=16, ch_out=8)
        self.Att2 = Attention_block(F_g=8, F_l=8, F_int=4)
        self.Up_conv2 = conv_block(ch_in=16, ch_out=8)

        self.Conv_1x1 = nn.Conv2d(8, num_classes, kernel_size=1, stride=1, padding=0)
        self.sigmoid = nn.Sigmoid()

        self.Conv_1x1_1 = nn.Conv2d(8, num_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, x, train_flag=True):
        # multi_scale_generator
        x_pre_1 = self.multi_scale_1(x)
        x_pre_2 = self.multi_scale_2(x)
        x_pre_3 = self.multi_scale_3(x)
        x_pre_4 = self.multi_scale_4(x)
        x_pre_5 = self.multi_scale_5(x)
        muti_scale_x = torch.cat((x_pre_1, x_pre_2, x_pre_3, x_pre_4, x_pre_5), dim=1)

        # encoding path
        x1 = self.Conv1(muti_scale_x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # offset convolution
        o1 = self.o1(x1)
        o2 = self.o2(x2)
        o3 = self.o3(x3)
        o4 = self.o4(x4)

        # decoding + concat path
        d5 = self.Up5(x5)
        x4 = self.Att5(g=d5, x=o4)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4, x=o3)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3, x=o2)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2, x=o1)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)
        if train_flag:
            return d1
        else:
            return self.sigmoid(d1)