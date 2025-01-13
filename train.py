import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.environ['NO_ALBUMENTATIONS_UPDATE']='True'

import csv
import sys
import click
import traceback
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import albumentations as A
import gc
import wandb
import matplotlib.pyplot as plt

from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
from torch.utils.data import DataLoader
from dataset import SatelliteDataset
from models.models.deeplabv3plus import DeepLabV3Plus
from models.models.unet import UNet
from models.models.resunetplusplus import ResUnetPlusPlus
from models.models.mdoaunet import MDOAU_net
from models.models.u2net import U2NET
from models.models.attent_unet import AttU_Net

from loss import DiceLoss, DiceBCELoss, IoULoss, FocalLoss, TverskyLoss
from utils.util import gpu_test, set_seed, count_parameters, get_data_info
from utils.metrics import calculate_metrics
from utils.visualize import visualize_training_log, visualize
from datetime import datetime
from scheduler import CosineAnnealingWarmUpRestarts

# Data Info
INPUT_CHANNEL_NUM = get_data_info("C:\workspace\KOREA_AQUACULTURE_DETECTION\dl_train_data\Train\Image")
CLASSES = get_data_info("C:\workspace\KOREA_AQUACULTURE_DETECTION\dl_train_data\Train\Mask")
if CLASSES == 1:
    is_binary = True
else:
    is_binary = False


@click.command()
@click.option("-D", "--data-dir", type=str, default='C:\workspace\KOREA_AQUACULTURE_DETECTION\dl_train_data\Train', help="Path for Data Directory")
@click.option(
    "-M",
    "--model-name",
    type=str,
    default='resunetplusplus',
    help="Choose models for Binary Segmentation. unet, deeplabv3plus, resunetplusplus, mdoaunet, u2net, attentunet are now available",
)
@click.option(
    "-E",
    "--num-epochs",
    type=int,
    default=300,
    help="Number of epochs to train the model for. Default - 100",
)
@click.option(
    "-L",
    "--learning-rate",
    type=float,
    default=0.0001,
    help="Learning Rate for model. Default - 1e-3",
)
@click.option(
    "-B",
    "--batch-size",
    type=int,
    default=8,
    help="Batch size of data for training. Default - 8",
)
@click.option(
    "-S",
    "--early-stop",
    type=bool,
    default=True,
    help="Stop training if val_loss hasn't improved for a certain no. of epochs. Default - True",
)
def main(
    data_dir: str,
    model_name : str,
    num_epochs: int,
    learning_rate: float,
    batch_size: int,
    early_stop: bool,
) -> None:
    """
    Training Script for Binary Segmentation.
    Please make sure your data is structured according to the folder structure specified in the Github Repository.
    Reference : https://github.com/mukund-ks/DeepLabV3Plus-PyTorch

    Refer to the Options below for usage.
    """
    click.secho(message="🚀 Training...", fg="blue", nl=True)


    """
    Set Random Seed
    Build Dataset & DataLoader
    Check Data Statistics
    """
    set_seed(39)
    custom_transform = A.Compose([
        A.Rotate(limit=(-10, 10), p=0.7),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        ToTensorV2()
    ])
  
    try:
        train_dataset = SatelliteDataset(
            data_dir=data_dir, split="train", transform=custom_transform
        )
        val_dataset = SatelliteDataset(
            data_dir=data_dir, split="val", transform=custom_transform
        )
    except Exception as _:
        click.secho(message="\n❗ Error \n", fg="red")
        click.secho(message=traceback.format_exc(), fg="yellow")
        sys.exit("Non-Existent Data Dir")

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=False, drop_last=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=False, drop_last=True)


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
    model.to(device)
    num_params = count_parameters(model)
    print("Total Parameters(M):", num_params/1000000)


    """
    Hyperparameter Setting
    """
    if CLASSES == 1:
        #criterion = DiceLoss()
        criterion = DiceBCELoss()
        #criterion = IoULoss()
        #criterion = FocalLoss()
        #criterion = TverskyLoss()
    else:
        criterion = nn.CrossEntropyLoss()
        #criterion = nn.BCEWithLogitsLoss()

    #optimizer = optim.AdamW(model.parameters(), lr = learning_rate)
    optimizer = optim.Adam(model.parameters(), lr = learning_rate)
    #optimizer = optim.SGD(model.parameters(),lr=learning_rate, momentum=0.9, weight_decay=0.0005)
    
    #scheduler = CosineAnnealingWarmUpRestarts(optimizer, T_0=150, T_mult=1, eta_max=0.1,  T_up=10, gamma=0.5)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lambda epoch: 0.95 ** epoch)
    #scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

    # For Early-Stopping
    patience_epochs = 50
    no_improvement_epochs = 0


    """
    Logging(wandb) 
    """
    # api key : be75ffb42a4ed5d3dd3ea5681c5d77961e02546b
    wandb.init(
        project="aquaculture_segmentation",
        config={
            "architecture": model_name,
            "dataset": "aquaculture",
            "epochs": num_epochs,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "optimizer": "AdamW",
            "loss_function": "DiceLoss",
            "scheduler": "ExponentialLR",
            "input_channels": INPUT_CHANNEL_NUM,
            "classes": CLASSES,
        }
    )

    best_val_loss = float("inf")
    click.echo(
        f"\n{click.style(text=f'Train Size: ', fg='blue')}{train_dataset.__len__()}\t{click.style(text=f'val Size: ', fg='blue')}{val_dataset.__len__()}\n"
    )

    # Save train result
    train_base_dir = 'outputs/train_output'
    now = datetime.now()
    folder_name = now.strftime("%Y_%m_%d_%H_%M_%S") + model_name
    train_output_dir = os.path.join(train_base_dir, folder_name)

    try:
        os.makedirs(train_output_dir, exist_ok=True)
        click.secho(message="Train output folder was successfully created\n", fg="blue")
    except OSError as e:
        click.secho(message="\n❗ Error\n", fg="red")
        sys.exit("OSError while creating output data dir")


    """
    Train Loop 
    """
    for epoch in range(num_epochs):

        # Clean Memory
        gc.collect()
        torch.cuda.empty_cache()

        # TRAINING
        model.train()
        train_loss = 0.0
        total_iou_train = 0.0
        total_pixel_accuracy_train = 0.0
        total_precision_train = 0.0
        total_recall_train = 0.0
        total_f1_train = 0.0

        train_dataloader = tqdm(
            train_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch"
        )

        current_lr = optimizer.param_groups[0]["lr"]

        iteration = 0
        for images, true_masks in train_dataloader:
            true_masks = true_masks.permute(0,3,1,2)
            images, true_masks = images.to(device), true_masks.to(device)
            images, true_masks = images.float(), true_masks.float()
            
            optimizer.zero_grad()   

            pred_probs = model(images)
            pred_probs_sig = F.sigmoid(pred_probs)

            t_loss = criterion(pred_probs_sig, true_masks)
            t_loss.backward()
            
            optimizer.step()

            train_loss += t_loss.item()
            iteration += 1
                                
            # Calculating metrics for training
            with torch.no_grad():
                if CLASSES == 1:  # Binary Segmentation
                    pred_masks = pred_probs_sig > 0.5
                else:  # Multi-class Segmentation
                    pred_masks = torch.argmax(torch.softmax(pred_probs, dim=1), dim=1)
                    
                # Visualize train process
                if epoch % 20 == 0:
                    visualize(images, pred_masks, true_masks,
                            img_save_path= train_output_dir,
                            epoch = str(epoch), iteration = str(iteration),
                            type='train', num = None, is_binary = is_binary)

                iou_train, pixel_accuracy_train, precision_train, recall_train, f1_train = calculate_metrics(
                    pred_masks, true_masks, CLASSES
                )

                total_iou_train += iou_train
                total_pixel_accuracy_train += pixel_accuracy_train
                total_precision_train += precision_train
                total_recall_train += recall_train
                total_f1_train += f1_train

            # Displaying metrics in the progress bar description
            train_dataloader.set_postfix(
                loss=t_loss.item(),
                train_iou=iou_train.item(),
                train_pix_acc=pixel_accuracy_train,
                train_precision=precision_train,
                train_recall=recall_train,
                train_f1=f1_train,
                lr=current_lr,
            )

        train_loss /= len(train_dataloader)
        avg_iou_train = total_iou_train / len(train_dataloader)
        avg_pixel_accuracy_train = total_pixel_accuracy_train / len(train_dataloader)
        avg_precision_train = total_precision_train / len(train_dataloader)
        avg_recall_train = total_recall_train / len(train_dataloader)
        avg_f1_train = total_f1_train / len(train_dataloader)

        scheduler.step(train_loss)


        """
        Validation Loop 
        """
        # VALIDATION
        model.eval()
        val_loss = 0.0
        total_iou_val = 0.0
        total_pixel_accuracy_val = 0.0
        total_precision_val = 0.0
        total_recall_val = 0.0
        total_f1_val = 0.0

        val_dataloader = tqdm(val_dataloader, desc=f"Validation", unit="batch")

        with torch.cuda.amp.autocast():
            with torch.no_grad():
                for images, true_masks in val_dataloader:
                    images, true_masks = images.to(device), true_masks.to(device)
                    true_masks = true_masks.permute(0,3,1,2)
                    images, true_masks = images.float(), true_masks.float()

                    pred_probs = model(images)
                    pred_probs_sig = F.sigmoid(pred_probs)

                    v_loss = criterion(pred_probs_sig, true_masks)
                    val_loss += v_loss.item()

                    # Calculating metrics for Validation
                    if CLASSES == 1:  # Binary Segmentation
                        pred_masks = pred_probs_sig > 0.5
                    else:  # Multi-class Segmentation
                        pred_masks = torch.argmax(torch.softmax(pred_probs, dim=1), dim=1)

                    iou_val, pixel_accuracy_val, precision_val, recall_val, f1_val = calculate_metrics(
                        pred_masks, true_masks, CLASSES
                    )

                    total_iou_val += iou_val
                    total_pixel_accuracy_val += pixel_accuracy_val
                    total_precision_val += precision_val
                    total_recall_val += recall_val
                    total_f1_val += f1_val

                    # Displaying metrics in progress bar description
                    val_dataloader.set_postfix(
                        val_loss=v_loss.item(),
                        val_iou=iou_val.item(),
                        val_pix_acc=pixel_accuracy_val,
                        val_precision=precision_val,
                        val_recall=recall_val,
                        val_f1=f1_val,
                        lr=current_lr,
                    )

        val_loss /= len(val_dataloader)
        avg_iou_val = total_iou_val / len(val_dataloader)
        avg_pixel_accuracy_val = total_pixel_accuracy_val / len(val_dataloader)
        avg_precision_val = total_precision_val / len(val_dataloader)
        avg_recall_val = total_recall_val / len(val_dataloader)
        avg_f1_val = total_f1_val / len(val_dataloader)

        print(
            f"{'-'*50}"
            f"\nEpoch {epoch + 1}/{num_epochs}\n"
            f"Avg Train Loss: {train_loss:.4f}\n"
            f"Avg Validation Loss: {val_loss:.4f}\n"
            f"Avg IoU Train: {avg_iou_train:.4f}\n"
            f"Avg IoU Val: {avg_iou_val:.4f}\n"
            f"Avg Pix Acc Train: {avg_pixel_accuracy_train:.4f}\n"
            f"Avg Pix Acc Val: {avg_pixel_accuracy_val:.4f}\n"
            f"Avg Precision Train: {avg_precision_train:.4f}\n"
            f"Avg Precision Val: {avg_precision_val:.4f}\n"
            f"Avg Recall Train: {avg_recall_train:.4f}\n"
            f"Avg Recall Val: {avg_recall_val:.4f}\n"
            f"Avg F1 Train: {avg_f1_train:.4f}\n"
            f"Avg F1 Val: {avg_f1_val:.4f}\n"
            f"Current LR: {current_lr}\n"
            f"{'-'*50}"
        )


        '''
        Save Best Model
        '''
        if val_loss < best_val_loss:

            checkpoint_path = "weights/train_result/best_model.pth"
            no_improvement_epochs = 0

            # Pytorch 
            click.secho(
                message=f"\n👀 val_loss improved from {best_val_loss:.4f} to {val_loss:.4f}\n",
                fg="green",
            )
            best_val_loss = val_loss
            torch.save(model.state_dict(), checkpoint_path) 
            
            # Wandb 
            artifact = wandb.Artifact(
                name=f"{model_name}_model", 
                type="model",
                description=f"Best model checkpoint with val_loss: {val_loss:.4f}"
            )
            artifact.add_file(checkpoint_path)
            wandb.log_artifact(artifact)

            click.secho(message="Saved Best Model! 🙌\n", fg="green")
            print(f"{'-'*50}")
        else:
            no_improvement_epochs += 1
            click.secho(
                message=f"\nval_loss did not improve from {best_val_loss:.4f}\n", fg="yellow"
            )
            print(f"{'-'*50}")


        '''
        Wandb logging
        '''
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_iou": avg_iou_train.item(),
            "val_iou": avg_iou_val.item(),
            "train_pixel_accuracy": avg_pixel_accuracy_train,
            "val_pixel_accuracy": avg_pixel_accuracy_val,
            "train_precision": avg_precision_train,
            "val_precision": avg_precision_val,
            "train_recall": avg_recall_train,
            "val_recall": avg_recall_val,
            "train_f1": avg_f1_train,
            "val_f1": avg_f1_val,
            "learning_rate": current_lr,
        })

    if epoch % 10 == 0:
        fig, axes = plt.subplots(2, images.shape[0] + 1, figsize=(15, 6))
        
        for band_num in range(images.shape[0]):
            axes[0, band_num].imshow(images[band_num])
            axes[0, band_num].set_title(f'Band {band_num+1}')
            axes[0, band_num].axis('off')
        
        axes[0, -1].imshow(pred_masks.float()[0].cpu().numpy())
        axes[0, -1].set_title('Prediction')
        axes[0, -1].axis('off')
        
        axes[1, images.shape[0]//2].imshow(true_masks.float()[0].cpu().numpy())
        axes[1, images.shape[0]//2].set_title('Ground Truth')
        axes[1, images.shape[0]//2].axis('off')
        
        for i in range(images.shape[0] + 1):
            if i != images.shape[0]//2:
                axes[1, i].axis('off')
        
        plt.tight_layout()
        
        wandb.log({
            "training_progress": wandb.Image(plt),
            "epoch": epoch
        })
        
        plt.close()


        '''
        Early Stopping
        '''         
        # Early-Stopping
        if early_stop:
            if no_improvement_epochs >= patience_epochs:
                click.secho(
                    message=f"\nEarly Stopping: val_loss did not improve for {patience_epochs} epochs.\n",
                    fg="red",
                )

    click.secho(message="🎉 Training Done!", fg="blue", nl=True)

    return

if __name__ == "__main__":
    main()
