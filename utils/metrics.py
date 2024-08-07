import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import torch
import warnings

warnings.filterwarnings('ignore')
from typing import Any

def calculate_metrics(pred_mask: Any, true_mask: Any) -> torch.Tensor:
    '''
    Calculate Metrics

    Metrics : IOU, Pixel Accuracy, Precision, Recall, F1 score.
    '''
    pred_mask = pred_mask.view(-1).float()
    true_mask = true_mask.view(-1).float()
    eps=1e-5

    # Calculating precision, recall, and F1 score using PyTorch
    TP = ((pred_mask == 1) & (true_mask == 1)).sum()
    FP = ((pred_mask == 1) & (true_mask == 0)).sum()
    FN = ((pred_mask == 0) & (true_mask == 1)).sum()
    TN = ((pred_mask == 0) & (true_mask == 0)).sum()
    
    iou = (TP + eps) / (TP + FP + FN + eps) 
    pixel_acc = (TP + TN + eps) / (TP + TN + FP + FN + eps)
    precision = (TP + eps) / (TP + FP + eps)
    recall = (TP + eps) / (TP + FN + eps)
    f1 = 2*((precision * recall)/(precision + recall))

    return iou.item(), pixel_acc.item(), precision.item(), recall.item(), f1.item()