import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import torch
import numpy as np
import random
import spectral.io.envi as envi
import cv2
import warnings
import netCDF4 as nc
import rasterio
import tifffile

from typing import Optional, Tuple, Union, List, Generator
warnings.filterwarnings('ignore')
from glob import glob
from pathlib import Path


def crop_img(
    data_array: np.ndarray,
    split_size: int,
    Y_num: int,
    X_num: int,
    hist_equal: bool,
) -> List[np.ndarray]:
    """
    Generate Cropped Images.
    """
    result = []

    for i in range(Y_num):
        for j in range(X_num):
            start_y = i * split_size
            start_x = j * split_size
            end_y = start_y + split_size
            end_x = start_x + split_size
            
            cropped_image = data_array[:, start_y:end_y, start_x:end_x]
            
            if hist_equal:
                cropped_image = histogram_equalization(cropped_image)

            result.append(cropped_image)

    return np.array(result, dtype=np.float32)


def process_satellite_data(
    data_dir: Union[str, Path],
    split_size: Optional[int] = 224,
    sampling: bool = False,
    indices: Optional[List] = None,
    negative_sampling_num: int = 0,
    hist_equal: bool = True,
    norm: bool = True,
    norm_type: str = 'linear_norm',
    value_check: bool = False,
    save_cropped_data: bool = False,
) -> List[np.ndarray]:
    """
    Process satellite data by reading files and optionally performing padding and cropping.
    
    Args:
        data_dir: Directory containing satellite data files
        split_size: Size for splitting large images. If None, assumes pre-cropped data
        sampling: Whether to perform sampling on cropped images
        indices: Indices for sampling.
        negative_sampling_num: Number of hard negative samples. If None, skip hard negative sampling
        hist_equal: Whether to perform histogram equalization
        norm: Whether to normalize the data
        norm_type: Type of normalization ('linear_norm', 'dynamic_world_norm', etc.)
        value_check: Whether to print value checking information
        save_cropped_data: Whether to save processed data

    Returns:
        Numpy array of (processed image array, sampling indices if sampling=True else None)
    """
    # Read the file data
    data_array = read_file(data_dir, norm=norm, norm_type=norm_type)
    print(f"Data array got shape {data_array.shape}")

    # If split_size is not provided, return data array directly
    if split_size is None:
        if norm and value_check:
            print(f"Data shape: {data_array.shape}")
            print(f"Data range: [{np.min(data_array)}, {np.max(data_array)}]")
        return data_array, None
    
    # Calculate padding dimensions for original img
    _, original_height, original_width = data_array.shape
    X_num = (original_width + split_size - 1) // split_size
    Y_num = (original_height + split_size - 1) // split_size
    
    pad_x = (X_num * split_size) - original_width
    pad_y = (Y_num * split_size) - original_height
    
    # Pad the array
    padded_array = np.pad(data_array, ((0,0), (0,pad_y), (0,pad_x)), 
                         'constant', constant_values=0)
    
    print(f"Padded imgs got shape {padded_array.shape}")

    # Set save directory
    dir_name = 'img' if padded_array.shape[0] != 1 else 'msk'
    save_dir = os.path.join(data_dir, "processed_images/{}".format(dir_name)) if save_cropped_data else None
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    # Crop into smaller images
    processed_images = None
    processed_images = crop_img(padded_array, split_size, Y_num, X_num, hist_equal)
    print(f"Cropped imgs got shape {processed_images.shape}")

    # Get only-object samples if required
    sampling_indices = None
    if sampling:
        if indices:
            sampling_indices = indices
            processed_images = np.array([processed_images[i] for i in sampling_indices])
            print(f"Sampled imgs got shape {processed_images.shape}")

            if save_cropped_data and save_dir and os.listdir(save_dir) == []:
                for index, img in enumerate(processed_images):
                    tifffile.imwrite(os.path.join(save_dir, f'{dir_name}_{index+1}.tiff'), img)

            return processed_images, sampling_indices
        else:
            sampling_indices = find_arrays_with_object(processed_images)
        
        # Hard Negative Sampling for improving generalization performance if required
        if negative_sampling_num:
            sampling_count = 0
            all_indices = list(range(len(processed_images)))
            np.random.shuffle(all_indices)
            
            for idx in all_indices:
                if sampling_count >= negative_sampling_num:
                    break
                if idx not in sampling_indices:
                    sampling_indices.append(idx)
                    sampling_count += 1
            
            sampling_indices.sort()
            
            if value_check:
                print(f"Added {sampling_count} hard negative samples")
                print(f"Total samples: {len(sampling_indices)}")
        
        processed_images = np.array([processed_images[i] for i in sampling_indices])
        print(f"Sampled imgs got shape {processed_images.shape}")

        if save_cropped_data and save_dir and os.listdir(save_dir) == []:
            for index, img in enumerate(processed_images):
                tifffile.imwrite(os.path.join(save_dir, f'{dir_name}_{index+1}.tiff'), img)

        return processed_images, sampling_indices
    
    return processed_images, sampling_indices


def unpad(padded_array : np.ndarray, pad_length : int):
    '''
    Unpad 2d padded image
    '''
    # Unpadding
    _, _, height, width = padded_array.shape
    unpadded_array = padded_array[0, 0, pad_length:height-pad_length, pad_length:width-pad_length]

    return unpadded_array


def restore_img(image_list : list, original_height : int, original_width : int, split_size : int):
    '''
    Restore Image by stitching cropped images
    '''
    # Calculate image number per row and column
    X_num = (original_width + split_size - 1) // split_size
    Y_num = (original_height + split_size - 1) // split_size
    
    pad_x = (X_num * split_size) - original_width
    pad_y = (Y_num * split_size) - original_height

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


def read_envi(file_path):
    '''
    Read ENVI type format data
    '''
    result = []
    # ENVI type 
    hdr_files_path = sorted(glob(os.path.join(file_path, "*.hdr")))
    img_files_path = sorted(glob(os.path.join(file_path, "*.img")))
    band_nums = len(hdr_files_path)
    if not hdr_files_path:
        raise ValueError(f"No ENVI files found in {file_path}")
    try:
        for i in range(band_nums):
            envi_hdr_path = hdr_files_path[i]
            envi_img_path = img_files_path[i]
            data = envi.open(envi_hdr_path, envi_img_path)
            band_data = np.array(data.load())
            if len(band_data.shape) == 3:
                band_data = band_data[:,:,0]
            result.append(band_data)

    except Exception as e:
        raise Exception(f"Error processing ENVI file: {str(e)}")
    
    return result


def read_tif(file_path):
    '''
    Read tif/tiff type format data
    '''
    # TIFF, TIF type 
    tiffs_file_path = sorted(glob(os.path.join(file_path, "*.tif")) + 
                                glob(os.path.join(file_path, "*.tiff")))
    if not tiffs_file_path:
        raise ValueError(f"No TIFF files found in {file_path}")

    file_num = len(tiffs_file_path)
    result = []

    try:
        for path in tiffs_file_path:
            with rasterio.open(path) as src:
                band_count = src.count  
                for band in range(1, band_count + 1):
                    img = src.read(band)
                    if file_num == 1:
                        result = img
                        return result
                    else:
                        result.append(img)
    except Exception as e:
        raise Exception(f"Error processing TIFF file: {str(e)}")
    
    return file_num, result


def read_nc(file_path):
    '''
    Read NC type format data
    '''
    # NetCDF type
    nc_file_path = sorted(glob(os.path.join(file_path, "*.nc")))
    if not nc_file_path:
        raise ValueError(f"No NC files found in {file_path}")
    
    file_num = len(nc_file_path)
    result = []

    try:
        for path in nc_file_path:
            img = []
            ds = nc.Dataset(path)
            band_names = list(ds.variables.keys())[1:-3]
            for i in range(len(band_names)):
                band_name = str(band_names[i])
                band = ds[band_name][:]
                img.append(band)
            ds.close()
            if file_num == 1:
                result = img
                return result
            else:
                result.append(img)

    except Exception as e:
        raise Exception(f"Error processing NC file: {str(e)}")
    
    return result

def read_file(file_path, norm=True, norm_type='linear_norm'):
    '''
    Read ENVI, TIFF, TIF, NC file Format and return it as numpy array type.

    Input : Directory where satellite data exists.
    Return : Numpy array type satellite data.
    '''
    data_array = []
    ext = get_files(file_path)

    # ENVI type 
    if any(e in ['.hdr', '.img'] for e in ext):
        data_array = read_envi(file_path)

    # TIFF, TIF type 
    elif any(e in ['.tif', '.tiff'] for e in ext):
        file_num, img = read_tif(file_path)
        if norm:
            if isinstance(img, list):
                img = [band_norm(x, norm_type, False) for x in img]
            else:
                img = band_norm(img, norm_type, False)

        if file_num == 1:
            data_array.append(img)
        else:
            data_array = img

    # NetCDF type 
    elif '.nc' in ext:
        file_num, img = read_nc(file_path)
        if norm:
            if isinstance(img, list):
                img = [band_norm(x, norm_type, False) for x in img]
            else:
                img = band_norm(img, norm_type, False)

        if file_num == 1:
            data_array.append(img)
        else:
            data_array = img

    else:
        raise ValueError(f"Unsupported file format: {ext}")

    return np.array(data_array)



def histogram_equalization(img: np.ndarray):
    """
    Histogram equalize code for numpy array
    Returns histogram equalized image
    """    
    hist_eqaulized_img = []

    for channel in img :

        zero_mask = (channel == 0)
        
        if np.all(zero_mask):
            hist_eqaulized_img.append(channel)
            continue
        
        flat_img = channel.flatten()
        hist, bins = np.histogram(flat_img[flat_img != 0], bins=256, density=True)      
        cdf = hist.cumsum()
        cdf = cdf / cdf[-1]

        result = np.zeros_like(flat_img)
        mask = (flat_img != 0)
        result[mask] = np.interp(flat_img[mask], bins[:-1], cdf)

        result = result.reshape(channel.shape)
        hist_eqaulized_img.append(result)
    
    return np.array(hist_eqaulized_img)


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