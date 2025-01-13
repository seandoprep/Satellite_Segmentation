import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SymmetricAttentionExtraction(nn.Module):
    def __init__(self, in_channels):
        super(SymmetricAttentionExtraction, self).__init__()
        
        # Initial convolution layers for feature extraction
        self.conv_f_m1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv_f_m2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        
        # Convolution for F_θ
        self.conv_f_theta = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1)
        
        # Expansion convolutions
        self.expand_conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.expand_conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.expand_conv3 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.expand_conv4 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        
        # Final processing
        self.conv_f_w = nn.Conv2d(2*in_channels, in_channels, kernel_size=3, padding=1)
        self.conv_f_a = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, f_m1, f_m2):
        # Initial concatenation
        concat_features = torch.cat([f_m1, f_m2], dim=1)
        f_theta = self.conv_f_theta(concat_features)
        
        # Strip pooling
        h, w = f_theta.size(2), f_theta.size(3)
        
        # Horizontal strip pooling
        h_pool = F.adaptive_avg_pool2d(f_theta, (h, 1))
        h_pool = h_pool.expand(-1, -1, -1, w)
        
        # Vertical strip pooling
        w_pool = F.adaptive_avg_pool2d(f_theta, (1, w))
        w_pool = w_pool.expand(-1, -1, h, -1)
        
        # Max pooling paths
        h_max = F.adaptive_max_pool2d(f_theta, (h, 1))
        h_max = h_max.expand(-1, -1, -1, w)
        
        w_max = F.adaptive_max_pool2d(f_theta, (1, w))
        w_max = w_max.expand(-1, -1, h, -1)
        
        # Expansion path 1 (avg + h pool)
        f_s1 = self.expand_conv1(h_pool)
        
        # Expansion path 2 (avg + w pool)
        f_s2 = self.expand_conv2(w_pool)
        
        # Expansion path 3 (max + h pool)
        f_s3 = self.expand_conv3(h_max)
        
        # Expansion path 4 (max + w pool)
        f_s4 = self.expand_conv4(w_max)
        
        # Combine expanded features
        add1 = f_s1 + f_s2
        add2 = f_s3 + f_s4
        mul = add1 * add2
        
        # Final processing
        concat = torch.cat([mul, f_theta], dim=1)
        f_w = self.conv_f_w(concat)
        f_w = self.relu(f_w)
        
        # Final attention map
        f_a = self.conv_f_a(f_w)
        f_a = self.sigmoid(f_a)
        
        # Final output
        out = f_theta * f_a
        
        return out

# Usage example:
if __name__ == "__main__":
    # Example input tensor dimensions (batch_size, channels, height, width)
    batch_size, channels, height, width = 1, 64, 32, 32
    
    # Create sample input tensors
    f_m1 = torch.randn(batch_size, channels, height, width)
    f_m2 = torch.randn(batch_size, channels, height, width)
    
    # Initialize the module
    attention_module = SymmetricAttentionExtraction(channels)
    
    # Forward pass
    output = attention_module(f_m1, f_m2)
    
    print(f"Input shape: {f_m1.shape}")
    print(f"Output shape: {output.shape}")