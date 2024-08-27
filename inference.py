import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import sys
import torch
import click
import traceback
import albumentations as A
import numpy as np
import torch.nn.functional as F
import netCDF4 as nc 

from tqdm import tqdm
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
from models.models.deeplabv3plus import DeepLabV3Plus
from models.models.unet import UNet
from models.models.resunetplusplus import ResUnetPlusPlus
from models.models.mdoaunet import MDOAU_net
from models.models.u2net import U2NET
from models.models.attent_unet import AttU_Net

from dataset import InferenceDataset
from utils.util import set_seed, gpu_test, unpad, restore_img, remove_noise, read_file
from utils.save_data import save_nc, label_binary_image, mask_to_hexagon, mask_to_shp
from utils.visualize import compare_result
from datetime import datetime
from PIL import Image

Image.MAX_IMAGE_PIXELS = None


INPUT_CHANNEL_NUM = 4
INPUT = (256, 256)
CLASSES = 1  # For Binary Segmentatoin


@click.command()
@click.option("-D", "--data-dir", type=str, default='data\\Train', required=True, help="Path for Data Directory")
@click.option(
    "-M",
    "--model-name",
    type=str,
    default='unet',
    help="Choose models for Binary Segmentation. unet, deeplabv3plus, resunetplusplus, mdoaunet, u2net, attentunet are now available",
)
@click.option(
    "-P",
    "--model-path",
    type=str,
    default='./weights/train_result/best_model.pth',
    help="Path for pretrained model weight file",
)
@click.option(
    "-B",
    "--batch-size",
    type=int,
    default=1,
    help="Batch size of data for Inference. Default - 8",
)
def main(
    data_dir: str,
    model_name : str,
    model_path : str,
    batch_size : int) -> None:
    """
    Inference Script for Binary Segmentation.
    Please make sure your data is structured according to the folder structure specified in the Github Repository.
    Reference : https://github.com/mukund-ks/DeepLabV3Plus-PyTorch

    Refer to the Options below for usage.
    """
    click.secho(message="🔎 Inference...", fg="blue")

    set_seed(99)
    custom_transform = A.Compose([
        ToTensorV2(),
    ])

    try:
        inference_dataset = InferenceDataset(data_dir=data_dir, transform=custom_transform)
        inference_dataloader = DataLoader(inference_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
        click.echo(message=f"\n{click.style('Inference Size: ', fg='blue')}{inference_dataset.__len__()}\n")
        inference_dataloader = tqdm(inference_dataloader, desc="Inference", unit="image")
    except Exception as _:
        click.secho(message="\n❗ Error\n", fg="red")
        click.secho(message=traceback.format_exc(), fg="yellow")
        sys.exit("Non-Existent Data Dir")

    # Defining Model
    if model_name == 'unet':
        model = UNet(in_channel=INPUT_CHANNEL_NUM, num_classes=CLASSES)
        print("Model : U-Net")
    elif model_name == 'deeplabv3plus':
        model = DeepLabV3Plus(in_channel=INPUT_CHANNEL_NUM, num_classes=CLASSES)
        print("Model : DeepLabV3+")
    elif model_name == 'resunetplusplus':
        model = ResUnetPlusPlus(in_channel=INPUT_CHANNEL_NUM, num_classes=CLASSES)
        print("Model : ResUNet++")
    elif model_name == 'mdoaunet':
        model = MDOAU_net(in_channel=INPUT_CHANNEL_NUM, num_classes=CLASSES)
        print("Model : MDOAUNet")
    elif model_name == 'u2net':
        model = U2NET(in_channel=INPUT_CHANNEL_NUM, num_classes=CLASSES)
        print("Model : U2-Net")
    elif model_name == 'attentunet':
        model = AttU_Net(in_channel=INPUT_CHANNEL_NUM, num_classes=CLASSES)
        print("Model : Attention U-Net")    

    # Load Trained Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_test()
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()

    # Save test result
    inference_base_dir = 'outputs/inference_output'
    now = datetime.now()
    folder_name = now.strftime("%Y_%m_%d_%H_%M_%S") + model_name
    inference_output_dir = os.path.join(inference_base_dir, folder_name)

    try:
        os.makedirs(inference_output_dir, exist_ok=True)
        click.secho(message="Inference output folder was successfully created\n", fg="blue")
    except OSError as e:
        click.secho(message="\n❗ Error\n", fg="red")
        sys.exit("OSError while creating output data dir")

    # Main loop
    image_list = []
    pad_length = 16

    img_path = os.path.join(data_dir, 'Image')
    _, image_height, image_width = read_file(img_path, True, 'dynamic_world_norm').shape

    with torch.no_grad():
        for i, (images, _) in enumerate(inference_dataloader):
            images = images.to(device)

            outputs = model(images)
            
            pred_mask_binary = F.sigmoid(outputs[0].squeeze()) > 0.5
            pred_mask_np = pred_mask_binary.cpu().detach().numpy()
            pred_mask_np = unpad(pred_mask_np, pad_length)
            image_list.append(pred_mask_np)

    # Restore Images  
    click.secho(message="🔎 Restoring data...", fg="green")

    restored = restore_img(image_list, image_height, image_width, 224)
    restored_img = np.array(restored, np.uint8) 
    img = Image.fromarray((restored_img*255))

    # Save into JPG images
    click.secho(message="🔎 Save into .jpg format data...", fg="green")

    jpg_path = os.path.join(inference_output_dir, 'Inference_output.jpg')
    img.save((jpg_path), 'JPEG')

    # Save into NC images
    click.secho(message="🔎 Save into .nc format data...", fg="green")
    original_array_path = 'data\Train\original_nc\Final_Images_msk.nc'  # Need Original Data for adding coordinate information
    nc_path = os.path.join(inference_output_dir, 'Inference_output.nc')

    fdata = nc.Dataset(original_array_path)
    lat_grid = np.array(fdata['lat'][:])
    lon_grid = np.array(fdata['lon'][:])
    save_nc(nc_path, restored_img, lat_grid, lon_grid)

    # Save into shapefile data(linestring & polygon)
    click.secho(message="🔎 Save .shp format data...", fg="green")
    morphed_img = remove_noise(restored_img)
    labeled_image = label_binary_image(morphed_img)
    mask_to_shp(labeled_image, inference_output_dir, lon_grid, lat_grid)

    # Save Hexagon data into Shapefile(Polygon)
    click.secho(message="🔎 Save Hexagon data...", fg="green")
    hexbin_path = os.path.join(inference_output_dir, 'Hexbin.shp')

    mask_to_hexagon(inference_output = restored_img, 
                    output_path = hexbin_path, 
                    grid_size = (134, 110), 
                    bins = [250, 500, 1000, 2500, 5000, 10000, 50000], 
                    mincnt = 250,
                    alpha = 0.5)

    # Compare with Manual data
    click.secho(message="🔎 Comparing Images...", fg="green")

    prediction_path = os.path.join(inference_output_dir, 'Inference_output.jpg')
    prediction = Image.open(prediction_path)
    prediction_np = np.array(prediction)
    
    true_mask_path = 'data\Train\Mask'
    true_mask_np = read_file(true_mask_path, None, None)

    result = compare_result(prediction_np, true_mask_np)
    img_np = np.array(result, np.uint8)
    img = Image.fromarray(img_np)
    img.save((os.path.join(inference_output_dir, 'Compare_output.jpg')), 'JPEG')

    click.secho(message="🎉 Inference Done!", fg="blue")
    return

if __name__ == "__main__":
    main()