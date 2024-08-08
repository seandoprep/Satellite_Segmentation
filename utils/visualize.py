import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn.functional as F
import warnings
warnings.filterwarnings('ignore')

from typing import Any


def visualize_train( 
        original_img: Any,
        pred_mask: Any,
        true_mask: Any,
        img_save_path : str,
        epoch : str,
        iter : str,
        ) -> None:
    '''
    Visualize training process per epoch and Save it(For 3 band data).
    '''
    original_img_cpu = original_img[0].cpu().numpy()
    pred_mask_binary = F.sigmoid(pred_mask[0, 0]) > 0.5
    pred = pred_mask_binary.cpu().detach().numpy()
    pred = np.expand_dims(pred, axis=2)
    true = true_mask[0].cpu().detach().numpy()

    # Visuzlize
    band_number = original_img_cpu.shape[0]

    plt.figure(figsize=(28,16))
    band_names = ['NDCI', 'NDWI', 'NIR', 'RED']
    for i in range(band_number):
        band = original_img_cpu[i,:,:]
        plt.subplot(3, 3, i+1)
        plt.imshow(band, cmap='gray')
        plt.title('{}'.format(band_names[i]))


    # Prediction
    plt.subplot(3, 3, band_number+1)
    plt.imshow(pred, cmap='gray')
    plt.title('Prediction')

    # True Mask
    plt.subplot(3, 3, band_number+2)
    plt.imshow(true, cmap='gray')
    plt.title('True Mask')

    plt.savefig(os.path.join(img_save_path, 'Training_result_epoch_{}_iter_{}.png'.format(epoch, iter)))
    plt.close()


def visualize_test( 
        original_img: Any,
        pred_mask: Any,
        true_mask: Any,
        img_save_path : str,
        num : int
        ) -> None:
    '''
    Visualize test process per image and Save it(For 3 band data).
    '''

    # Get data
    original_img_cpu = original_img[0].cpu().numpy()
    pred_mask_binary = F.sigmoid(pred_mask[0, 0]) > 0.5
    pred = pred_mask_binary.cpu().detach().numpy()
    pred = np.expand_dims(pred, axis=2)
    true = true_mask[0].cpu().detach().numpy()

    # Visuzlize
    band_number = original_img_cpu.shape[0]

    plt.figure(figsize=(28,16))
    band_names = ['NDCI', 'NDWI', 'NIR', 'RED']

    for i in range(band_number):
        band = original_img_cpu[i,:,:]
        plt.subplot(3, 3, i+1)
        plt.imshow(band, cmap='gray')
        plt.title('{}'.format(band_names[i]))

    # Prediction
    plt.subplot(3, 3, band_number+1)
    plt.imshow(pred, cmap='gray')
    plt.title('Prediction')

    # True Mask
    plt.subplot(3, 3, band_number+2)
    plt.imshow(true, cmap='gray')
    plt.title('True Mask')

    plt.savefig(os.path.join(img_save_path, 'Test_result_{}.png'.format(num)))
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
    lr = training_log['Learning Rate']

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