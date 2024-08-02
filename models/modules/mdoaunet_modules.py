import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import torch
import torch.nn as nn
from einops import rearrange, repeat
import numpy as np

''' 
Modules for MDOAU-net
https://github.com/Jichao-Wang/MDOAU-net/blob/main/MDOAU_net.py
'''

class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=11, stride=1, padding=10, dilation=2, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x
    
    
class multi_scaled_dilation_conv_block(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size, dilation=1):
        super(multi_scaled_dilation_conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.ReflectionPad2d(int((kernel_size - 1) / 2 * dilation)),
            nn.Conv2d(ch_in, ch_out, kernel_size=kernel_size, stride=1, dilation=dilation, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x
    

class bias_convolution(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size, dilation=1, direction=''):
        # default is normal convolution
        super(bias_convolution, self).__init__()
        self.direction = direction
        self.padding_size = int((kernel_size - 1) * dilation)
        # self.direction_padding = nn.ReflectionPad2d(self.padding_size)
        self.direction_padding_LU = nn.ReflectionPad2d((self.padding_size, 0, self.padding_size, 0))
        self.direction_padding_RU = nn.ReflectionPad2d((0, self.padding_size, self.padding_size, 0))
        self.direction_padding_LD = nn.ReflectionPad2d((self.padding_size, 0, 0, self.padding_size))
        self.direction_padding_RD = nn.ReflectionPad2d((0, self.padding_size, 0, self.padding_size))

        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=kernel_size, stride=1, dilation=dilation, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # print(self.padding_size)
        # x = self.direction_padding(x)
        x_LU = self.direction_padding_LU(x)
        x_RU = self.direction_padding_RU(x)
        x_LD = self.direction_padding_LD(x)
        x_RD = self.direction_padding_RD(x)

        if self.direction == 'LU':
            # padding to left up
            return self.conv(x_LU)

        elif self.direction == 'LD':
            # padding to left down
            return self.conv(x_LD)

        elif self.direction == 'RU':
            # padding to right up
            return self.conv(x_RU)

        elif self.direction == 'RD':
            # padding to right down
            return self.conv(x_RD)

        else:
            # normal padding
            return self.conv(x)
        

class offset_convolution(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(offset_convolution, self).__init__()
        self.LU_bias_convolution = bias_convolution(ch_in=ch_in, ch_out=ch_out, kernel_size=7, dilation=1,
                                                    direction='LU')
        self.LD_bias_convolution = bias_convolution(ch_in=ch_in, ch_out=ch_out, kernel_size=7, dilation=1,
                                                    direction='LD')
        self.RU_bias_convolution = bias_convolution(ch_in=ch_in, ch_out=ch_out, kernel_size=7, dilation=1,
                                                    direction='RU')
        self.RD_bias_convolution = bias_convolution(ch_in=ch_in, ch_out=ch_out, kernel_size=7, dilation=1,
                                                    direction='RD')
        self.final_conv = nn.Conv2d(ch_out * 4, ch_out, kernel_size=3, stride=1, padding=1)
        self.BN = nn.BatchNorm2d(ch_out)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        LU_BC = self.LU_bias_convolution(x)
        LD_BC = self.LD_bias_convolution(x)
        RU_BC = self.RU_bias_convolution(x)
        RD_BC = self.RD_bias_convolution(x)
        d = torch.cat((LU_BC, LD_BC, RU_BC, RD_BC), dim=1)
        d = self.final_conv(d)
        d = self.BN(d)
        d = self.activation(d)
        return d


class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x
    

class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi
