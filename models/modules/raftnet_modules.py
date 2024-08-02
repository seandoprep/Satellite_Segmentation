import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import torch
import torch.nn as nn

'''
Model from https://www.mdpi.com/2072-4292/14/18/4587
'''

class Dual_Channel_Block(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Dual_Channel_Block, self).__init__()

        self.upconv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=True)
        self.upconv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=True)
        self.upconv3 = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=True)

        self.dconv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=True)
        self.dconv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=True)
        self.dconv3 = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=True)

        self.up_out_conv = nn.Conv2d(3*out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=True)
        self.d_out_conv = nn.Conv2d(3*out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self, x):
        up_channel1 = self.upconv1(x)
        up_channel2 = self.upconv2(up_channel1)
        up_channel3 = self.upconv3(up_channel2)
        up_channel = torch.cat([up_channel1,up_channel2,up_channel3], dim=1)
        up_channel = self.up_out_conv(up_channel)

        d_channel1 = self.dconv1(x)
        d_channel2 = self.dconv2(d_channel1)
        d_channel3 = self.dconv3(d_channel2)
        d_channel = torch.cat([d_channel1,d_channel2,d_channel3], dim=1)
        d_channel = self.d_out_conv(d_channel)

        out = up_channel + d_channel
        return out


class Res_HDC(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Res_HDC, self).__init__()

        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, dilation=1, padding='same', bias=True),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace = True),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, dilation=1, padding='same', bias=True),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace = True),            
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, dilation=1, padding='same', bias=True),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace = True)
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(in_channel+out_channel, out_channel, kernel_size=3, stride=1, dilation=1, padding='same', bias=True),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace = True),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, dilation=2, padding='same', bias=True),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace = True),            
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, dilation=1, padding='same', bias=True),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace = True)
        )


        self.conv_block3 = nn.Sequential(
            nn.Conv2d(in_channel+2*out_channel, out_channel, kernel_size=3, stride=1, dilation=1, padding='same', bias=True),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace = True),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, dilation=3, padding='same', bias=True),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace = True),            
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, dilation=1, padding='same', bias=True),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace = True)
        )

        self.conv_out = nn.Conv2d(in_channel+3*out_channel, out_channel, kernel_size=3, stride=1, dilation=1, padding='same', bias=True)


    def forward(self, x):
        conv_1 = self.conv_block1(x)
        conv_1 = torch.cat([conv_1, x], dim=1)
        
        conv_2 = self.conv_block2(conv_1)
        conv_2 = torch.cat([conv_1, conv_2], dim=1)

        conv_3 = self.conv_block3(conv_2)
        conv_3 = torch.cat([conv_2, conv_3], dim=1)

        out = self.conv_out(conv_3)
        return out

