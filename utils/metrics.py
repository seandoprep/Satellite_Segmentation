import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import torch
import warnings

warnings.filterwarnings('ignore')
from typing import Any

def calculate_metrics(pred_mask: Any, true_mask: Any, classes: Any) -> torch.Tensor:
    '''
    Calculate Metrics

    Metrics : IOU, Pixel Accuracy, Precision, Recall, F1 score.
    '''
    eps = 1e-7
    iou_list = []

    if classes == 1:
        TP = ((pred_mask == 1) & (true_mask == 1)).sum()
        FP = ((pred_mask == 1) & (true_mask == 0)).sum()
        FN = ((pred_mask == 0) & (true_mask == 1)).sum()

        pixel_accuracy = (pred_mask == true_mask).float().mean()
        iou = (TP + eps) / (TP + FP + FN + eps) 
        precision = (TP + eps) / (TP + FP + eps)
        recall = (TP + eps) / (TP + FN + eps)
        f1 = 2*(precision * recall)/(precision + recall)

    else:
        true_mask_argmax = torch.zeros_like(pred_mask)
        for class_id in range(classes):
            # Convert true mask into argmax type
            true_mask_argmax[true_mask[:, class_id, :, :] > 0] = class_id

            TP = ((pred_mask == class_id) & (true_mask_argmax == class_id)).sum()
            FP = ((pred_mask == class_id) & (true_mask_argmax != class_id)).sum()
            FN = ((pred_mask != class_id) & (true_mask_argmax == class_id)).sum()

            iou = (TP + eps) / (TP + FP + FN + eps) 
            iou_list.append(iou)
            
            if class_id == 0:  # Assuming background is class 0
                continue
            
            pixel_accuracy = (pred_mask == true_mask_argmax).float().mean()
            precision = (TP + eps) / (TP + FP + eps)
            recall = (TP + eps) / (TP + FN + eps)
            f1 = 2*(precision * recall)/(precision + recall)
        
        mean_iou = sum(iou_list) / len(iou_list)
        iou = mean_iou

    return iou, pixel_accuracy.item(), precision.item(), recall.item(), f1.item()