import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import torch
import torch.nn as nn

from models.modules.raftnet_modules import Dual_Channel_Block, Res_HDC
from torch.utils.checkpoint import checkpoint

'''
Model architecture from https://www.mdpi.com/2072-4292/14/18/4587
'''

# RaftNet main architecture
class RaftNet(nn.Module):
    def __init__(self, in_channel: int = 3, num_classes: int = 1) -> None:
        super(RaftNet, self).__init__()
        
        # Encoder
        self.dc_block1 = Dual_Channel_Block(in_channel, 32)
        self.dc_block2 = Dual_Channel_Block(32, 64)
        self.dc_block3 = Dual_Channel_Block(64, 128)
        self.dc_block4 = Dual_Channel_Block(128, 256)
        self.dc_block5 = Dual_Channel_Block(256, 512)
            
        # Decoder
        self.conv1 = nn.Conv2d(64+512, 512, kernel_size=3, stride=1, dilation=1, padding='same', bias=True)
        self.conv2 = nn.Conv2d(256+256, 256, kernel_size=3, stride=1, dilation=1, padding='same', bias=True)
        self.conv3 = nn.Conv2d(128+128, 128, kernel_size=3, stride=1, dilation=1, padding='same', bias=True)
        self.conv4 = nn.Conv2d(64+64, 64, kernel_size=3, stride=1, dilation=1, padding='same', bias=True)

        self.reshdc1 = Res_HDC(512, 64)
        self.reshdc2 = Res_HDC(512, 128)
        self.reshdc3 = Res_HDC(512+128, 256)
        self.reshdc4 = Res_HDC(256, 64)
        self.reshdc5 = Res_HDC(64+256, 128)
        self.reshdc6 = Res_HDC(128, 256)
        self.reshdc7 = Res_HDC(256+128, 64)
        self.reshdc8 = Res_HDC(64, 128)
        self.reshdc9 = Res_HDC(128+64, 256)
         
        self.conv_last = nn.Conv2d(256+32, num_classes, 1)

    def forward(self, x):
        
        # Encoder
        dc_1 = checkpoint(self.dc_block1, x)
        dc_2 = self.dc_block2(dc_1)
        dc_3 = self.dc_block3(dc_2)
        dc_4 = self.dc_block4(dc_3)
        dc_5 = checkpoint(self.dc_block5, dc_4)

        # Decoder
        reshdc1 = checkpoint(self.reshdc1, dc_5)
        conv1 = self.conv1(torch.cat([reshdc1, dc_5], dim=1))
        reshdc2 = self.reshdc2(conv1)
        reshdc3 = self.reshdc3(torch.cat([reshdc2, conv1], dim=1))
        conv2 = self.conv2(torch.cat([reshdc3, dc_4], dim=1))
        reshdc4 = checkpoint(self.reshdc4, conv2)
        reshdc5 = self.reshdc5(torch.cat([reshdc4, conv2], dim=1))
        conv3 = self.conv3(torch.cat([reshdc5, dc_3], dim=1))
        reshdc6 = checkpoint(self.reshdc6, conv3)
        reshdc7 = self.reshdc7(torch.cat([reshdc6, conv3], dim=1))   
        conv4 = self.conv3(torch.cat([reshdc7, dc_2], dim=1))
        reshdc8 = self.reshdc8(conv4)
        reshdc9 = self.reshdc9(torch.cat([reshdc8, conv4], dim=1))   

        out = checkpoint(self.conv_last, torch.cat([reshdc9, dc_1]), dim=1)

        return out