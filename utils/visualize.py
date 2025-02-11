import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn.functional as F
import warnings
import math
import cv2

warnings.filterwarnings('ignore')

from typing import Any


def visualize(
        original_img: np.ndarray,
        pred_mask: np.ndarray,
        true_mask: np.ndarray,
        img_save_path: str,
        epoch: str,
        iteration: str,
        type: str,
        num: int,
        is_binary: bool = False
) -> None:
    """
    Visualize the training/test process and save the result.

    Args:
        original_img (np.ndarray): Input image tensor with shape (C, H, W).
        pred_mask (np.ndarray): Predicted mask tensor with shape (C, H, W) for multiclass or (1, H, W) for binary.
        true_mask (np.ndarray): Ground truth mask tensor with shape (C, H, W) for multiclass or (1, H, W) for binary.
        img_save_path (str): Path to save the visualization image.
        epoch (str): Current epoch.
        iter (str): Current iteration.
        type (str): 'train' or 'test'.
        num (int): Unique identifier for the visualization.
        is_binary (bool, optional): Whether it's a binary segmentation problem. Defaults to False.
    """

    original_img = original_img.cpu()
    pred_mask = pred_mask.detach().cpu()
    true_mask = true_mask.detach().cpu()

    plt.figure(figsize=(12, 12))

    num_channels = original_img.shape[0]
    rows = 4
    cols = 4

    for i in range(num_channels):
        # Plot original image
        plt.subplot(rows, cols, i + 1)
        plt.imshow(original_img[0,i], cmap='gray')
        plt.title(f'Band {i + 1}')
        plt.axis("off")

    # Plot predicted mask
    if is_binary:
        plt.subplot(rows, cols, num_channels + 1)
        plt.imshow(pred_mask[0], cmap='gray')
        plt.title('Prediction')
        plt.axis("off")
    else:
        for i in range(num_channels+1):
            class_mask = [pred_mask == i]
            plt.subplot(rows, cols, num_channels + i + 1)
            plt.imshow(class_mask[0], cmap='gray')
            plt.title(f'Class {i}' if i > 0 else 'Background')
            plt.axis("off")

    # Plot ground truth mask
    plt.subplot(rows, cols, num_channels + 2)
    plt.axis("off")
    plt.imshow(true_mask[0] if is_binary else np.argmax(true_mask, axis=0), cmap='gray')
    plt.title('Ground Truth')

    # Save the visualization
    if type == 'train':
        filename = f'Training_result_epoch_{epoch}_iter_{iteration}.png'
    elif type == 'test':
        filename = f'Test_result_{num}.png'
    else:
        filename = f'Inference_result_{num}.png'

    plt.tight_layout()
    plt.savefig(os.path.join(img_save_path, filename))
    plt.close()


def compare_result(prediction : np.array, true_mask : np.array):
    '''
    Compare Prediction and True Mask.
    '''
    result = np.zeros((prediction.shape[0], prediction.shape[1], 3))
    result[:,:,0] = prediction
    result[:,:,2] = true_mask[0]

    return result