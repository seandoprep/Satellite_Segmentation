import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import torch
from torch.nn import init
import numpy as np
import random
import spectral.io.envi as envi
import cv2
import warnings
warnings.filterwarnings('ignore')

from glob import glob


def pad_crop(original_array : np.ndarray, split_size : int):
    '''
    Pad and Crop Large Satellite Images for deep learning training.
    '''
    _, original_height, original_width = original_array.shape

    # Padding 
    X_num = original_width // split_size + 1
    Y_num = original_height // split_size + 1

    pad_x = (split_size * (X_num)) - original_width
    pad_y = (split_size * (Y_num)) - original_height

    padded_array = np.pad(original_array, ((0,0),(0,pad_y),(0,pad_x)), 'constant', constant_values=0)

    # Cropping
    cropped_images = []
    for i in range(Y_num):
        for j in range(X_num):
            start_y = i * split_size
            start_x = j * split_size
            end_y = start_y + split_size
            end_x = start_x + split_size

            cropped_image = padded_array[:, start_y:end_y, start_x:end_x]
            cropped_images.append(cropped_image)

    return np.array(cropped_images)


def unpad(padded_array : np.ndarray, pad_length : int):
    '''
    Unpad 2d padded image
    '''
    # Unpadding
    height, width = padded_array.shape
    unpadded_array = padded_array[pad_length:height-pad_length, pad_length:width-pad_length]

    return unpadded_array


def restore_img(image_list : list, original_height : int, original_width : int, split_size : int):
    '''
    Restore Image by stitching cropped images
    '''
    # Calculate image number per row and column
    X_num = original_width // split_size + 1
    Y_num = original_height // split_size + 1

    pad_x = (split_size * (X_num)) - original_width
    pad_y = (split_size * (Y_num)) - original_height

    zero_array = np.zeros((original_height+pad_y, original_width+pad_x))

    for i in range(Y_num):
        for j in range(X_num):
            start_y = i * split_size
            start_x = j * split_size
            end_y = start_y + split_size
            end_x = start_x + split_size
            zero_array[start_y:end_y, start_x:end_x] = image_list[i * X_num + j] 

    restored_img = zero_array[:-pad_y, :-pad_x]

    return restored_img


def band_norm(band : np.array, norm_type : str, value_check : bool):
    '''
    Band Normalization for Satellite Image(Sentinel-1/2). 

    Input : Raw band data
    Return : Normalized band data

    Tips : 
    1) Negative values are changed to Positive values for deep learning training
    2) norm_type should be one of linear_norm, or dynamic_world_norm
    3) Modify boundary values as necessary
    4) This code is suited for Input Image which is already Land/Sea Masked(Land value : 0)
     
    Need to add z-score, std normalization
    Reference : https://medium.com/sentinel-hub/how-to-normalize-satellite-images-for-deep-learning-d5b668c885af
    '''
    SMOOTH = 1e-5

    if np.any(band < 0):
        band_abs = band - np.min(band)
        band_mean = np.mean(band_abs[band_abs != -np.min(band)])
        band_abs[band_abs == -np.min(band)] = band_mean
    else:
        band_abs = band

    if norm_type == 'linear_norm':
        input_band_lower_bound, input_band_upper_bound = np.percentile(band_abs[band_abs != -np.min(band)], 1), np.percentile(band_abs[band_abs != -np.min(band)], 99)
        input_band_range = input_band_upper_bound - input_band_lower_bound

        band_norm = (band_abs - input_band_lower_bound) / input_band_range  # Percentile Normalization
        band_norm = np.clip((band_norm) / np.max(band_norm), 0, 1)  # Let Value Range : [0, 1]

    elif norm_type == 'dynamic_world_norm':
    
        def sigmoid(x):
            return 1 / (1 + np.exp(-(x+SMOOTH)))

        band_log = np.log1p(band_abs)
        band_log_for_percentile = np.log1p(band_abs[band_abs != -np.min(band)])

        input_band_lower_bound, input_band_upper_bound = np.percentile(band_log_for_percentile, 30), np.percentile(band_log_for_percentile, 70)  # Percentile Normalization
        input_band_range = input_band_upper_bound - input_band_lower_bound

        band_norm = sigmoid((band_log - input_band_lower_bound) / (input_band_range))  # Let Value Range : [0, 1] by Sigmoid Operation
    
    elif norm_type == 'mask_norm':
        band_norm = band_abs / 255.0

    else:
        raise Exception("norm_type should be one of 'linear_norm', or 'dynamic_world_norm'.")

    if value_check:
        print("Band Value :\n", band)
        print("Band Min Max :", np.min(band), np.max(band))
        print("Band abs Value :\n", band_abs)
        print("Band abs Min Max :", np.min(band_abs), np.max(band_abs))
        print("Input Lower Bound :", input_band_lower_bound)
        print("Input Upper Bound :", input_band_upper_bound)
        print("Band Norm Value :", band_norm)
        print("Band Norm Min Max :", np.min(band_norm), np.max(band_norm))
        print('--------------------------------------------------')

    return band_norm


def read_envi_file(img_path, norm = True, norm_type = 'linear_norm'):
    '''
    Read ENVI file Format and return it as numpy array type.

    Input : Directory where ENVI hdr, img file exist.
    Return : Numpy array of stacked ENVI .img data.
    '''
    hdr_files = sorted(glob(os.path.join(img_path, "*.hdr")))
    img_files = sorted(glob(os.path.join(img_path, "*.img")))
    band_nums = len(hdr_files)

    envi_data = []
    for i in range(band_nums):
        envi_hdr_path = hdr_files[i]
        envi_img_path = img_files[i]

        data = envi.open(envi_hdr_path, envi_img_path)
        if norm:
            img = np.array(data.load())[:,:,0]
            img = band_norm(img, norm_type, False)
        else:
            img = np.array(data.load())[:,:,0]
            
        envi_data.append(img)

    return np.array(envi_data)


def remove_noise(binary_image, opening_kernel_size=(3, 3), closing_kernel_size=(3, 3), opening_iterations=1, closing_iterations=1):
    '''
    Find Target Object.

    Input : Binarized array
    Return : Binarized array with Morpological operations(Closing, Opening)
    '''

    # Define the structuring elements for morphological operations
    opening_kernel = np.ones(opening_kernel_size, np.uint8)
    closing_kernel = np.ones(closing_kernel_size, np.uint8)
    
    # Apply morphological closing operation
    closing = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, closing_kernel, iterations=closing_iterations)
    
    # Apply morphological opening operation
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, opening_kernel, iterations=opening_iterations)
    
    return opening


def find_arrays_with_object(arrays_list):
    '''
    Find Target Object.

    Input : Binarized numpy array(0 : no object, 1 : object exists)
    Return : Indices of pixel which has value 1
    '''
    indices_with_one = [index for index, array in enumerate(arrays_list) if np.any(array > 0)]

    return indices_with_one


def set_seed(random_seed : int):
    '''
    Control Randomness

    Input : Random seed number
    '''
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
    np.random.seed(random_seed)
    random.seed(random_seed)

    # Can fully control randomness, but speed will be slow
    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = False


def gpu_test():
    '''
    Check GPU availability
    '''
    if torch.version.cuda is None:
        print('Pytorch with CUDA is not ready')
    else :
        print('Pytorch Version : {}'.format(torch.version.cuda))

    if torch.cuda.is_available():
        print('CUDA is currently available')
    else: 
        print('CUDA is currently unavailable')


def count_parameters(model):
    '''
    Check Model Parameters

    Input : DL model
    Return : Paremeter number of input DL model
    '''
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# def init_weights(net, init_type='normal', gain=0.02):
#     def init_func(m):
#         classname = m.__class__.__name__
#         if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
#             if init_type == 'normal':
#                 init.normal_(m.weight.data, 0.0, gain)
#             elif init_type == 'xavier':
#                 init.xavier_normal_(m.weight.data, gain=gain)
#             elif init_type == 'kaiming':
#                 init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
#             elif init_type == 'orthogonal':
#                 init.orthogonal_(m.weight.data, gain=gain)
#             else:
#                 raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
#             if hasattr(m, 'bias') and m.bias is not None:
#                 init.constant_(m.bias.data, 0.0)
#         elif classname.find('BatchNorm2d') != -1:
#             init.normal_(m.weight.data, 1.0, gain)
#             init.constant_(m.bias.data, 0.0)

#     print('initialize network with %s' % init_type)
#     net.apply(init_func)