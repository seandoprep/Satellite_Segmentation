import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any

'''
Code from : https://www.kaggle.com/code/sungjunghwan/loss-function-of-image-segmentation
'''

SMOOTH = 1e-6

class DiceLoss(nn.Module):
    def __init__(self) -> None:
        super(DiceLoss, self).__init__()

    def forward(self, pred_mask: Any, true_mask: Any) -> torch.Tensor:
        #flatten label and prediction tensors

        pred_mask = F.sigmoid(pred_mask)

        pred_mask = pred_mask.view(-1).float()
        true_mask = true_mask.view(-1).float()

        intersection = torch.sum(pred_mask * true_mask)
        total = torch.sum(pred_mask) + torch.sum(true_mask)

        # Add a small epsilon to the denominator to avoid division by zero
        dice_loss = 1.0 - (2.0 * intersection + SMOOTH) / (total + SMOOTH)
        return dice_loss
    
    
class DiceBCELoss(nn.Module):
    def __init__(self, dice_weight=0.3, bce_weight=0.7):
        super(DiceBCELoss, self).__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.smooth = 1e-6

    def forward(self, pred_mask, true_mask):
        # BCE Loss
        bce = F.binary_cross_entropy_with_logits(pred_mask, true_mask, reduction='mean')
        
        # Dice Loss
        pred_probs = F.sigmoid(pred_mask)  
        pred_flat = pred_probs.view(pred_mask.size(0), -1)
        true_flat = true_mask.view(true_mask.size(0), -1)
        
        intersection = (pred_flat * true_flat).sum(dim=1)
        union = pred_flat.sum(dim=1) + true_flat.sum(dim=1)
       
        dice_score = torch.ones_like(intersection)
        zero_masks = true_flat.sum(dim=1) == 0
        non_zero_masks = ~zero_masks
        
        if non_zero_masks.any():
            dice_score[non_zero_masks] = (2.0 * intersection[non_zero_masks] + self.smooth) / (
                union[non_zero_masks] + self.smooth)
            
        dice_loss = 1.0 - dice_score.mean()
        
        return self.bce_weight * bce + self.dice_weight * dice_loss
    

class IoULoss(nn.Module):
    def __init__(self) -> None:
        super(IoULoss, self).__init__()

    def forward(self, pred_mask: Any, true_mask: Any) -> torch.Tensor:
        #flatten label and prediction tensors

        pred_mask = F.sigmoid(pred_mask)

        pred_mask = pred_mask.view(-1).float()
        true_mask = true_mask.view(-1).float()
        
        #intersection is equivalent to True Positive count
        #union is the mutually inclusive area of all labels & predictions 
        intersection = torch.sum(pred_mask * true_mask)
        total = torch.sum(pred_mask) + torch.sum(true_mask)
        union = total - intersection 
        
        IoU = (intersection + SMOOTH)/(union + SMOOTH)
                
        return 1 - IoU
    

class FocalLoss(nn.Module):
    def __init__(self) -> None:
        super(FocalLoss, self).__init__()

    def forward(self, pred_mask: Any, true_mask: Any, alpha=0.75, gamma=1.5):
        #flatten label and prediction tensors

        pred_mask = F.sigmoid(pred_mask)
        
        pred_mask = pred_mask.view(-1).float()
        true_mask = true_mask.view(-1).float()
        
        #first compute binary cross-entropy 
        BCE = F.binary_cross_entropy(pred_mask, true_mask, reduction='mean')
        BCE_EXP = torch.exp(-BCE)
        focal_loss = alpha * (1-BCE_EXP)**gamma * BCE
                       
        return focal_loss
    

class TverskyLoss(nn.Module):
    def __init__(self) -> None:
        super(TverskyLoss, self).__init__()

    def forward(self, pred_mask: Any, true_mask: Any, alpha=0.5, beta=0.5):
        #flatten label and prediction tensors

        pred_mask = F.sigmoid(pred_mask)
        
        pred_mask = pred_mask.view(-1).float()
        true_mask = true_mask.view(-1).float()
        
        #True Positives, False Positives & False Negatives
        TP = (pred_mask * true_mask).sum()    
        FP = ((1-true_mask) * pred_mask).sum()
        FN = (true_mask * (1-pred_mask)).sum()
       
        Tversky = (TP + SMOOTH) / (TP + alpha*FP + beta*FN + SMOOTH)  
        
        return 1 - Tversky