import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn.functional as F
import warnings
import math
warnings.filterwarnings('ignore')

from typing import Any


def visualize(
        original_img: np.ndarray,
        pred_mask: np.ndarray,
        true_mask: np.ndarray,
        img_save_path: str,
        epoch: str,
        iter: str,
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
    original_img = original_img.cpu()[0]
    pred_mask = pred_mask.detach().cpu()[0]
    true_mask = true_mask.detach().cpu()[0]

    plt.figure(figsize=(20, 12))

    # Plot original image
    num_channels = original_img.shape[0]
    rows = 2
    cols = num_channels + 2

    for i in range(num_channels):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(original_img[i], cmap='gray')
        plt.title(f'Band {i + 1}')

    # Plot predicted mask
    if is_binary:
        plt.subplot(rows, cols, num_channels + 1)
        plt.imshow(pred_mask[0], cmap='gray')
        plt.title('Prediction')
    else:
        for i in range(pred_mask.shape[0]):
            plt.subplot(rows, cols, num_channels + i + 1)
            plt.imshow(pred_mask[i], cmap='gray')
            plt.title(f'Pred Class {i}')

    # Plot ground truth mask
    plt.subplot(rows, cols, num_channels + num_channels + 1)
    plt.imshow(true_mask[0] if is_binary else np.argmax(true_mask, axis=0), cmap='gray')
    plt.title('Ground Truth')

    # Save the visualization
    if type == 'train':
        filename = f'Training_result_epoch_{epoch}_iter_{iter}.png'
    else:
        filename = f'Test_result_{num}.png'
    plt.savefig(os.path.join(img_save_path, filename))
    plt.close()


def visualize_training_log(training_logs_csv: str, img_save_path: str):
    '''
    Visualize training log and Save it.
    '''
    training_log = pd.read_csv(training_logs_csv)
    epochs = training_log['Epoch']
    loss_train = training_log['Avg Train Loss']
    loss_val = training_log['Avg Val Loss']
    IoU_train = training_log['Avg IoU Train']
    IoU_val = training_log['Avg IoU Val']
    pixacc_train = training_log['Avg Pix Acc Train']
    pixacc_val = training_log['Avg Pix Acc Val']
    precision_train = training_log['Avg Precision Train']
    precision_val = training_log['Avg Precision Val']
    recall_train = training_log['Avg Recall Train']
    recall_val = training_log['Avg Recall Val']
    f1_train = training_log['Avg F1 Train']
    f1_val = training_log['Avg F1 Val']

    plt.figure(figsize=(28,16))

    # Loss
    plt.subplot(2, 3, 1)
    plt.plot(epochs, loss_train)
    plt.plot(epochs, loss_val)
    plt.title('Train/Val Loss')
    plt.xlabel('epochs')
    plt.ylabel('Loss')
    plt.legend(('val', 'train'))

    # IoU
    plt.subplot(2, 3, 2)
    plt.plot(epochs, IoU_train)
    plt.plot(epochs, IoU_val)
    plt.title('Train/Val IoU')
    plt.xlabel('epochs')
    plt.ylabel('IoU')
    plt.legend(('val', 'train'))

    # Pixel accuracy
    plt.subplot(2, 3, 3)
    plt.plot(epochs, pixacc_train)
    plt.plot(epochs, pixacc_val)
    plt.title('Train/Val pixacc')
    plt.xlabel('epochs')
    plt.ylabel('pixacc')
    plt.legend(('val', 'train'))

    # Precision
    plt.subplot(2, 3, 4)
    plt.plot(epochs, precision_train)
    plt.plot(epochs, precision_val)
    plt.title('Train/Val precision')
    plt.xlabel('epochs')
    plt.ylabel('precision')
    plt.legend(('val', 'train'))

    # Recall
    plt.subplot(2, 3, 5)
    plt.plot(epochs, recall_train)
    plt.plot(epochs, recall_val)
    plt.title('Train/Val recall')
    plt.xlabel('epochs')
    plt.ylabel('recall')
    plt.legend(('val', 'train'))

    # f1 score
    plt.subplot(2, 3, 6)
    plt.plot(epochs, f1_train)
    plt.plot(epochs, f1_val)
    plt.title('Train/Val F1')
    plt.xlabel('epochs')
    plt.ylabel('f1')
    plt.legend(('val', 'train'))


    plt.savefig(os.path.join(img_save_path, 'Training_log.png'))
    plt.close()


def compare_result(prediction : np.array, true_mask : np.array):
    '''
    Compare Prediction and True Mask.
    '''
    result = np.zeros((prediction.shape[0], prediction.shape[1], 3))
    result[:,:,0] = prediction
    result[:,:,2] = true_mask[0]

    return result