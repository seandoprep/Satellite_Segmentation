import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import torch
from torch.nn import init
import numpy as np
import random
import spectral.io.envi as envi
import cv2
import warnings
import netCDF4 as nc
import rasterio
import torch.nn.functional as F
from typing import Tuple
warnings.filterwarnings('ignore')

from glob import glob
from PIL import Image
from utils.metrics import calculate_metrics

def pad_crop(original_array : np.ndarray, split_size : int, sampling : bool, indices : list):
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

    sampling_indices = 0
    
    if sampling:
        # Sampling for aquaculture data
        sampling_indices = find_arrays_with_object(cropped_images)

        # Add Non-aquaculture data for generalization performance
        #cnt = 0
        #for num in range(len(cropped_images)):   
        #    cnt += 1 
        #    if num in sampling_indices:
        #        pass
        #    else:
        #        sampling_indices.append(num)    
        #    if cnt == 100:
        #        break
        
        cropped_images = [cropped_images[i] for i in sampling_indices]

    if indices:
        cropped_images = [cropped_images[i] for i in indices]
        sampling_indices = indices

    return np.array(cropped_images), sampling_indices


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


def replace_mean_pixel_value(band):
    '''
    Replace land pixel values into mean sea water pixel values
    '''
    # All values become positive
    band_abs = band - np.min(band)
    
    # Replace land pixel values into sea water mean pixel values
    band_mean = np.mean(band_abs[band_abs != -np.min(band)])   
    band_abs[band_abs == -np.min(band)] = band_mean

    return band_abs


def band_norm(band : np.array, norm_type : str, value_check : bool):
    '''
    Band Normalization for Satellite Image(Sentinel-1/2). 

    Input : Raw band data
    Return : Normalized band data

    Tips : 
    1) Negative values are changed to Positive values for deep learning training
    2) norm_type should be one of linear_norm, dynamic_world_norm, robust_norm, hist_stretch or z_score_norm
    3) Modify boundary values as necessary
    4) This code is suited for Input Image which is already Land/Sea Masked(Land value : 0)

    Reference : https://medium.com/sentinel-hub/how-to-normalize-satellite-images-for-deep-learning-d5b668c885af
    '''
    SMOOTH = 1e-5
    
    # NULL value processing for S1
    band[band == -9999] = 0
    band[np.isnan(band)] = 0
    band_abs = band.copy()

    if norm_type == 'linear_norm':

        valid_mask = band_abs != 0
        valid_data = band_abs[valid_mask]
        
        if np.any(valid_data < 0):
            valid_data = valid_data - np.mean(valid_data)
            
        p10, p90 = np.percentile(valid_data, [10, 90])
        band_range = p90 - p10
        
        band_norm = np.zeros_like(band, dtype=float)
        
        band_norm[valid_mask] = (valid_data - p10) / band_range
        band_norm = np.clip(band_norm, 0, 1)

    elif norm_type == 'dynamic_world_norm':

        valid_mask = band_abs != 0
        
        if not np.any(valid_mask):
            if value_check:
                print("Warning: No valid data found (all zeros)")
            return np.zeros_like(band, dtype=float)
            
        valid_data = band_abs[valid_mask]
        
        valid_data = np.clip(valid_data, SMOOTH, None)
        valid_data = np.log1p(valid_data)
        
        p30, p70 = np.percentile(valid_data, [30, 70])
        
        if np.abs(p70 - p30) < SMOOTH:
            p70 = p30 + SMOOTH
            
        band_norm = np.zeros_like(band, dtype=float)
        
        normalized = -(valid_data - p30) / (p70 - p30 + SMOOTH)
        band_norm[valid_mask] = 1 / (1 + np.exp(normalized))
        band_norm = np.clip(band_norm, 0, 1)
            
    elif norm_type == 'robust_norm':
            
        median = np.median(band_abs[band_abs != 0])
        mad = np.median(np.abs(band_abs[band_abs != 0] - median)) + SMOOTH
                
        band_norm = (band_abs - median) / mad
        
        p1, p99 = np.percentile(band_norm[band_abs != 0], [1, 99])
        band_norm = np.clip(band_norm, p1, p99)
        band_norm = (band_norm - p1) / (p99 - p1)
        
        band_norm[band_abs == 0] = 0

    elif norm_type == 'z_score_norm':

        # Z-score Normalization
        band_mean = np.mean(band[band != 0])   
        band_std = np.std(band[band != 0])   
        band_z_scores = (band - band_mean) / band_std
        
        # Scaling into [0,1]
        min_z = np.min(band_z_scores)
        max_z = np.max(band_z_scores)
        band_norm = (band_z_scores - min_z) / (max_z - min_z)

    elif norm_type == 'mask_norm':
        band_norm = (band_abs > 0).astype(np.float32)

    else:
        raise Exception("norm_type should be one of 'linear_norm', 'dynamic_world_norm', 'robust_norm', 'z_score_norm'.")

    if value_check:
        print("Band Value :\n", band)
        print("Band Min Max :", np.min(band), np.max(band))
        print("Band Norm Value :", band_norm)
        print("Band Norm Min Max :", np.min(band_norm), np.max(band_norm))
        print('--------------------------------------------------')

    return band_norm


def get_files(file_path):
    exts = []
    file_list= os.listdir(file_path)
    for file in file_list:
        ext = os.path.splitext(file)[-1].lower()
        exts.append(ext)
    ext = list(set(exts))
    return ext


def get_data_info(file_path):
    files_path = []
    ext = get_files(file_path)
    if any(e in ['.hdr', '.tif', '.tiff', '.nc'] for e in ext):
        files_path = sorted(glob(os.path.join(file_path, "*.hdr"))  +
                            glob(os.path.join(file_path, "*.tif"))  + 
                            glob(os.path.join(file_path, "*.tiff")) +
                            glob(os.path.join(file_path, "*.nc"))) 
    result = len(files_path)
    return result


def read_file(file_path, norm=True, norm_type='linear_norm'):
    '''
    Read ENVI, TIFF, TIF, NC file Format and return it as numpy array type.

    Input : Directory where satellite data exists.
    Return : Numpy array of stacked satellite data.
    '''
    data_array = []
    ext = get_files(file_path)

    # ENVI type 
    if any(e in ['.hdr', '.img'] for e in ext):
        hdr_files_path = sorted(glob(os.path.join(file_path, "*.hdr")))
        img_files_path = sorted(glob(os.path.join(file_path, "*.img")))
        band_nums = len(hdr_files_path)

        for i in range(band_nums):
            envi_hdr_path = hdr_files_path[i]
            envi_img_path = img_files_path[i]
            data = envi.open(envi_hdr_path, envi_img_path)
            img = np.array(data.load())[:,:,0]

            #print(envi_hdr_path)
            #if '2020_S1_VV.hdr' in envi_hdr_path :
            #    img = np.clip(img, -20, -10)

            #elif '2020_S1_VH.hdr' in envi_hdr_path :
            #    img = np.clip(img, -30, -10)

            #elif 'NDCI.hdr' in envi_hdr_path :
            #    img = np.clip(img, -0.22, 0)

            #elif 'NDWI.hdr' in envi_hdr_path :
            #    img = np.clip(img, 0, 1)

            #elif 'scaled_NIR.hdr' in envi_hdr_path :
            #    img = np.clip(img, 0, 0.3)

            #elif 'scaled_Red.hdr' in envi_hdr_path :
            #    img = np.clip(img, 0, 0.3)
            
            #else:
            #    print("No Available data")

            if norm:
                img = band_norm(img, norm_type, False)
            data_array.append(img)

    # TIFF, TIF type 
    elif any(e in ['.tif', '.tiff'] for e in ext):
        tiffs_file_path = sorted(glob(os.path.join(file_path, "*.tif")) + 
                                  glob(os.path.join(file_path, "*.tiff")))
        with rasterio.open(tiffs_file_path[0]) as src:
            band_count = src.count  
            for band in range(1, band_count + 1):
                img = src.read(band)
                if norm:
                    img = band_norm(img, norm_type, False)
                data_array.append(img)
        
    # NetCDF type 
    elif '.nc' in ext:
        nc_file_path = sorted(glob(os.path.join(file_path, "*.nc")))
        ds = nc.Dataset(nc_file_path[0])
        band_names = list(ds.variables.keys())[1:-3]
        for i in range(len(band_names)):
            band_name = str(band_names[i])
            img = ds[band_name][:]
            if norm:
                img = band_norm(img, norm_type, False)
            data_array.append(img)
        ds.close()

    else:
        raise ValueError(f"Unsupported file format: {ext}")

    return np.array(data_array)


def remove_noise(
    binary_image: np.ndarray,
    opening_kernel_size: Tuple[int, int] = (3, 3),
    closing_kernel_size: Tuple[int, int] = (3, 3),
    opening_iterations: int = 1,
    closing_iterations: int = 1
) -> np.ndarray:
    """
    Remove noise from binary image using morphological operations.
    Returns cleaned binary image.
    """
    # Validate input
    if binary_image.dtype != np.uint8:
        binary_image = binary_image.astype(np.uint8)
    
    # Create kernels
    opening_kernel = np.ones(opening_kernel_size, np.uint8)
    closing_kernel = np.ones(closing_kernel_size, np.uint8)
    
    # Apply morphological operations
    closing = cv2.morphologyEx(
        binary_image,
        cv2.MORPH_CLOSE,
        closing_kernel,
        iterations=closing_iterations
    )
    
    opening = cv2.morphologyEx(
        closing,
        cv2.MORPH_OPEN,
        opening_kernel,
        iterations=opening_iterations
    )
    
    return opening


def find_arrays_with_object(arrays_list):
    '''
    Find Target Object.

    Input : Binarized numpy array(0 : no object, 1 : object exists)
    Return : Indices of pixel which has value 1
    '''
    indices_with_object = [index for index, array in enumerate(arrays_list) if np.any(array > 0)]

    return indices_with_object


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


class LossDebugger:
    '''
    Debug model
    '''
    def __init__(self):
        self.debug_info = {}
    
    def check_tensor(self, tensor, name=""):
        if torch.is_tensor(tensor):
            info = {
                'isnan': torch.isnan(tensor).any().item(),
                'isinf': torch.isinf(tensor).any().item(),
                'min': tensor.min().item(),
                'max': tensor.max().item(),
                'mean': tensor.mean().item(),
                'std': tensor.std().item() if tensor.numel() > 1 else 0
            }
            self.debug_info[name] = info
            return info
        return None

    def check_grad(self, model, name=""):
        grad_info = {}
        for n, p in model.named_parameters():
            if p.grad is not None:
                grad_info[n] = {
                    'grad_nan': torch.isnan(p.grad).any().item(),
                    'grad_inf': torch.isinf(p.grad).any().item(),
                    'grad_min': p.grad.min().item(),
                    'grad_max': p.grad.max().item(),
                    'grad_mean': p.grad.mean().item(),
                    'weight_min': p.min().item(),
                    'weight_max': p.max().item(),
                }
        self.debug_info[f"{name}_grad"] = grad_info
        return grad_info