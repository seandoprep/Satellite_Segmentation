import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import sys
import torch
import click
import traceback
import albumentations as A

from tqdm import tqdm
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
from models.models.deeplabv3plus import DeepLabV3Plus
from models.models.unet import UNet
from models.models.resunetplusplus import ResUnetPlusPlus
from models.models.mdoaunet import MDOAU_net
from models.models.u2net import U2NET

from dataset import SatelliteDataset
from utils.util import set_seed, gpu_test
from utils.metrics import calculate_metrics
from utils.visualize import visualize_test
from datetime import datetime

INPUT_CHANNEL_NUM = 4
INPUT = (256, 256)
CLASSES = 1  # For Binary Segmentatoin


@click.command()
@click.option("-D", "--data-dir", type=str, default='data\\Train\\ENVI', required=True, help="Path for Data Directory")
@click.option(
    "-M",
    "--model-name",
    type=str,
    default='resunetplusplus',
    help="Choose models for Binary Segmentation. unet, deeplabv3plus, resunetplusplus, and transunet are now available.",
)
@click.option(
    "-S",
    "--model-path",
    type=str,
    default='./weights/train_result/best_model.pth',
    help="Path for pretrained model weight file",
)
def main(
    data_dir: str,
    model_name : str,
    model_path : str) -> None:
    """
    Test Script for Binary Segmentation.
    Please make sure your data is structured according to the folder structure specified in tshe Github Repository.
    Reference : https://github.com/mukund-ks/DeepLabV3Plus-PyTorch

    Refer to the Options below for usage.
    """
    click.secho(message="🔎 Evaluation...", fg="blue")

    set_seed(99)
    custom_transform = A.Compose([
        ToTensorV2(),
    ])

    try:
        test_dataset = SatelliteDataset(data_dir=data_dir, split="test", transform=custom_transform)
        test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, drop_last=True)
        click.echo(message=f"\n{click.style('Test Size: ', fg='blue')}{test_dataset.__len__()}\n")
        test_dataloader = tqdm(test_dataloader, desc="Test", unit="image")
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
    elif model_name == 'msunet':
        model = MS_UNet(in_channel=INPUT_CHANNEL_NUM, num_classes=CLASSES)
        print("Model : Multispectral U-Net")
    elif model_name == 'u2net':
        model = U2NET(in_channel=INPUT_CHANNEL_NUM, num_classes=CLASSES)
        print("Model : U2-Net")


    # Load Trained Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_test()
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()

    # Save test result
    test_base_dir = 'outputs/test_output'
    now = datetime.now()
    folder_name = now.strftime("%Y_%m_%d_%H_%M_%S") + model_name
    test_output_dir = os.path.join(test_base_dir, folder_name)

    try:
        os.makedirs(test_output_dir, exist_ok=True)
        click.secho(message="Test output folder was successfully created\n", fg="blue")
    except OSError as e:
        click.secho(message="\n❗ Error\n", fg="red")
        sys.exit("OSError while creating output data dir")

    # Main loop
    total_iou_test = 0.0
    total_pixel_accuracy_test = 0.0
    total_precision_test = 0.0
    total_recall_test = 0.0
    total_f1_test = 0.0

    with torch.no_grad():
        for i, (image, mask) in enumerate(test_dataloader):
            image = image.to(device)
            mask = mask.to(device)

            output = model(image)
            pred_mask = output > 0.5

            visualize_test(image, output, mask, 
                        img_save_path= test_output_dir, num = i)

            iou_test, pixel_accuracy_test, precision_test, recall_test, f1_test = calculate_metrics(
                pred_mask, mask
            )

            total_iou_test += iou_test
            total_pixel_accuracy_test += pixel_accuracy_test
            total_precision_test += precision_test
            total_recall_test += recall_test
            total_f1_test += f1_test
        
            # Displaying metrics in the progress bar description
            test_dataloader.set_postfix(
                test_iou=iou_test,
                test_pix_acc=pixel_accuracy_test,
                test_precision=precision_test,
                test_recall=recall_test,
                test_f1=f1_test,
            )

    avg_iou_test = total_iou_test / len(test_dataloader)
    avg_pixel_accuracy_test = total_pixel_accuracy_test / len(test_dataloader)
    avg_precision_test = total_precision_test / len(test_dataloader)
    avg_recall_test = total_recall_test / len(test_dataloader)
    avg_f1_test = total_f1_test / len(test_dataloader)                

    print(
        f"{'-'*50}"
        f"Avg IoU Test: {avg_iou_test:.4f}\n"
        f"Avg Pix Acc Test: {avg_pixel_accuracy_test:.4f}\n"
        f"Avg Precision Test: {avg_precision_test:.4f}\n"
        f"Avg Recall Test: {avg_recall_test:.4f}\n"
        f"Avg F1 Test: {avg_f1_test:.4f}\n"
        f"{'-'*50}"
        )

    click.secho(message="🎉 Test Done!", fg="blue")

    return

if __name__ == "__main__":
    main()