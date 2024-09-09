import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import torch
import warnings

warnings.filterwarnings('ignore')
from typing import Any, Tuple

def calculate_metrics(pred_mask: torch.Tensor, true_mask: torch.Tensor, classes: int) -> Tuple[torch.Tensor, float, float, float, float]:
    '''
    Calculate Metrics for Image Segmentation

    Metrics: IoU (Intersection over Union), Pixel Accuracy, Precision, Recall, F1 score.

    Args:
    pred_mask (torch.Tensor): Predicted mask
    true_mask (torch.Tensor): Ground truth mask
    classes (int): Number of classes (including background)

    Returns:
    Tuple[torch.Tensor, float, float, float, float]: IoU, Pixel Accuracy, Precision, Recall, F1 score
    '''
    eps = 1e-6

    if classes == 1:  # Binary segmentation
        pred_mask = pred_mask.view(-1)
        true_mask = true_mask.view(-1)

        TP = (pred_mask * true_mask).sum()
        FP = ((pred_mask == 1) & (true_mask == 0)).sum()
        FN = ((pred_mask == 0) & (true_mask == 1)).sum()
        TN = ((pred_mask == 0) & (true_mask == 0)).sum()

        pixel_accuracy = (TP + TN) / (TP + TN + FP + FN)
        iou = (TP + eps) / (TP + FP + FN + eps)
        precision = (TP + eps) / (TP + FP + eps)
        recall = (TP + eps) / (TP + FN + eps)
        f1 = 2 * (precision * recall) / (precision + recall + eps)

    else:  # Multi-class segmentation
        pred_mask = pred_mask.view(-1)
        true_mask = true_mask.argmax(dim=1).view(-1)

        pixel_accuracy_list = []
        iou_list = []
        precision_list = []
        recall_list = []
        f1_list = []

        for class_id in range(classes+1):
            TP = ((pred_mask == class_id) & (true_mask == class_id)).sum()
            FP = ((pred_mask == class_id) & (true_mask != class_id)).sum()
            FN = ((pred_mask != class_id) & (true_mask == class_id)).sum()
            TN = ((pred_mask != class_id) & (true_mask != class_id)).sum()
            
            pixel_accuracy = (TP + TN) / (TP + TN + FP + FN)
            iou = (TP + eps) / (TP + FP + FN + eps)
            precision = (TP + eps) / (TP + FP + eps)
            recall = (TP + eps) / (TP + FN + eps)
            f1 = 2 * (precision * recall) / (precision + recall + eps)

            pixel_accuracy_list.append(pixel_accuracy)
            iou_list.append(iou)
            precision_list.append(precision)
            recall_list.append(recall)
            f1_list.append(f1)

        pixel_accuracy = torch.stack(pixel_accuracy_list).mean()
        iou = torch.stack(iou_list).mean()
        precision = torch.stack(precision_list).mean()
        recall = torch.stack(recall_list).mean()
        f1 = torch.stack(f1_list).mean()

    return iou, pixel_accuracy.item(), precision.item(), recall.item(), f1.item()