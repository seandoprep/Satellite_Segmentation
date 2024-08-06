import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import torch
import warnings
warnings.filterwarnings('ignore')

from typing import Any

SMOOTH = 1e-8

def calculate_metrics(pred_mask: Any, true_mask: Any) -> torch.Tensor:
    '''
    Calculate Metrics

    Metrics : IOU, Pixel Accuracy, Precision, Recall, F1 score.
    '''
    pred_mask = pred_mask.view(-1).float()
    true_mask = true_mask.view(-1).float()
    eps=1e-7

    # Overlap Metrics
    tp = torch.sum(pred_mask * true_mask)  # TP
    fp = torch.sum(pred_mask * (1 - true_mask))  # FP
    fn = torch.sum((1 - pred_mask) * true_mask)  # FN
    tn = torch.sum((1 - pred_mask) * (1 - true_mask))  # TN   

    iou = (tp) / (tp + fp + fn + eps) 
    pixel_acc = (tp + tn) / (tp + tn + fp + fn + eps)
    precision = (tp) / (tp + fp + eps)
    recall = (tp) / (tp + fn + eps)
    f1 = 2*((precision * recall)/(precision + recall + eps))

    return iou.item(), pixel_acc.item(), precision.item(), recall.item(), f1.item()