import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.environ['NO_ALBUMENTATIONS_UPDATE']='True'

import sys
import torch
import click
import traceback
import wandb
import albumentations as A
import torch.nn.functional as F

from tqdm import tqdm
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
from models.models.deeplabv3plus import DeepLabV3Plus
from models.models.unet import UNet
from models.models.resunetplusplus import ResUnetPlusPlus
from models.models.mdoaunet import MDOAU_net
from models.models.u2net import U2NET
from models.models.attent_unet import AttU_Net

from dataset import SatelliteDataset
from utils.util import set_seed, gpu_test, get_data_info, unpad
from utils.metrics import calculate_metrics
from utils.visualize import visualize
from datetime import datetime

# Data Info
INPUT_CHANNEL_NUM = get_data_info("C:\workspace\KOREA_AQUACULTURE_DETECTION\dl_train_data\Train\Image")
CLASSES = get_data_info("C:\workspace\KOREA_AQUACULTURE_DETECTION\dl_train_data\Train\Mask")
if CLASSES == 1:
    is_binary = True
else:
    is_binary = False

@click.command()
@click.option("-D", "--data-dir", type=str, default='C:\workspace\KOREA_AQUACULTURE_DETECTION\dl_train_data\Train', required=True, help="Path for Data Directory")
@click.option(
    "-M",
    "--model-name",
    type=str,
    default='resunetplusplus',
    help="Choose models for Binary Segmentation. unet, deeplabv3plus, resunetplusplus, mdoaunet, u2net, attentunet are now available",
)
@click.option(
    "-P",
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


    """
    Set Random Seed
    Build Dataset & DataLoader
    """
    set_seed(39)
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


    """
    Check GPU Availability & Set Model
    """
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_test()
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()


    """
    Logging(wandb) 
    """
    wandb.init(
    project="aquaculture_segmentation",
    resume=True,
    config={
        "test_model_path": model_path,
        "test_data_dir": data_dir,
    }
)
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


    """
    Test Loop 
    """
    total_iou_test = 0.0
    total_pixel_accuracy_test = 0.0
    total_precision_test = 0.0
    total_recall_test = 0.0
    total_f1_test = 0.0

    with torch.no_grad():
        for i, (image, true_mask) in enumerate(test_dataloader):
            true_mask = true_mask.permute(0,3,1,2)
            image, true_mask = image.to(device), true_mask.to(device)
            image, true_mask = image.float(), true_mask.float()

            pred_probs = model(image) 
            if CLASSES == 1:  # Binary Segmentation
                pred_probs = F.sigmoid(pred_probs)
                pred_mask = pred_probs > 0.5  
            elif CLASSES != 1:  # Multiclass Segmentation
                pred_probs = F.softmax(pred_probs, dim=1)
                pred_mask = torch.argmax(pred_probs, dim=1)
            
            image = image[:, :, 16:-16, 16:-16]  # Unpad
            pred_probs = pred_probs[:, :, 16:-16, 16:-16]  # Unpad
            pred_mask = pred_mask[:, :, 16:-16, 16:-16]  # Unpad
            true_mask = true_mask[:, :, 16:-16, 16:-16]  # Unpad
         
            iou_test, pixel_accuracy_test, precision_test, recall_test, f1_test = calculate_metrics(
                pred_mask, true_mask, CLASSES
            )

            visualize(image, pred_mask, true_mask,
                     img_save_path= test_output_dir, 
                      epoch='none', iteration='none', type='test', 
                      num = i, is_binary = is_binary)
            
            total_iou_test += iou_test
            total_pixel_accuracy_test += pixel_accuracy_test
            total_precision_test += precision_test
            total_recall_test += recall_test
            total_f1_test += f1_test
        
            # Displaying metrics in the progress bar description
            test_dataloader.set_postfix(
                test_iou=iou_test.item(),
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


    '''
    Wandb logging
    '''
    wandb.log({
        "test_iou": avg_iou_test.item(),
        "test_pixel_accuracy": avg_pixel_accuracy_test,
        "test_precision": avg_precision_test,
        "test_recall": avg_recall_test,
        "test_f1": avg_f1_test
    })    

    click.secho(message="🎉 Test Done!", fg="blue")

    return

if __name__ == "__main__":
    main()