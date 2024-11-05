import os
import re
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset as BaseDataset, Subset
import cv2
import numpy as np
from model import Model
from model import ModelWithScheduler
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import wandb
from sklearn.model_selection import KFold
from datetime import datetime
import albumentations as A
from pytorch_lightning.loggers import WandbLogger

import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

def pad_im(image, target_shape=(768, 768)):
        if image.ndim == 3 and (image.shape[2] == 3 or image.shape[2] == 1):  # If image is in HWC format
            h, w, c = image.shape
            image = np.transpose(image, (2, 0, 1))  # Convert to CHW for processing
        elif image.ndim == 3 and (image.shape[0] == 3 or image.shape[0] == 1):  # If image is in CHW format
            c, h, w = image.shape
            # image = np.transpose(image, (1, 2, 0))  # Convert to HWC for processing
        elif image.ndim == 2:
            h, w = image.shape
        else:
            raise ValueError("Unexpected image format, image is of shape:", image.shape)
        

        # Calculate scaling factor to fit within target shape while maintaining aspect ratio
        scale = min(target_shape[0] / h, target_shape[1] / w)
        new_h, new_w = int(h * scale), int(w * scale)

        # Ensure dimensions are divisible by 32
        new_h = ((new_h + 31) // 32) * 32
        new_w = ((new_w + 31) // 32) * 32

        # Resize image while maintaining aspect ratio
        resized_image = np.array([cv2.resize(img, (new_w, new_h)) for img in image])
        resized_mask = np.array([cv2.resize(img, (new_w, new_h)) for img in image])

        pad_h = max(target_shape[0] - new_h, 0)
        pad_w = max(target_shape[1] - new_w, 0)
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        # print(f"Padding values - top: {pad_top}, bottom: {pad_bottom}, left: {pad_left}, right: {pad_right}")

        padded_im = np.pad(resized_image, ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right)), mode='constant')
        # print(f"Padded image shape: {padded_im.shape}")

        return padded_im

def convert_mask(mask):
    norm_mask = mask / 255.0
    mask_expanded = np.expand_dims(norm_mask, axis=0)
    return mask_expanded

class Dataset(BaseDataset):
    def __init__(self, images_dir, masks_dir, augmentation=None, preprocessing=None):
        self.image_ids = [file for file in os.listdir(images_dir) if file.endswith(('.jpeg'))]
        self.image_ids.sort()
        self.mask_ids = [file for file in os.listdir(masks_dir) if file.endswith(('.png'))]
        self.mask_ids.sort()

        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.image_ids]
        self.masks_fps = [os.path.join(masks_dir, mask_id) for mask_id in self.mask_ids]
        
        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):
        image = cv2.imread(self.images_fps[i])
        if image is None:
            raise FileNotFoundError(f"Unable to load image at {self.images_fps[i]}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks_fps[i], 0)
        
        # Debug: Check initial shapes
        # print(f"Initial shapes - Image: {image.shape}, Mask: {mask.shape}")
        
        # Apply augmentations if specified
        if self.augmentation:
            augmented = self.augmentation(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        # Apply brightness/contrast adjustment only to the image
        image = get_brightness_contrast_aug()(image=image)['image']
            
        image = np.transpose(image, (2, 0, 1))
        image = pad_im(image)
        
        mask = convert_mask(mask)
        mask = pad_im(mask)

        return {"image": image, "mask": mask}

    def __len__(self):
        return len(self.image_ids)

# Define augmentations using albumentations
def get_geom_augmentations():
    return A.Compose([
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=1.4, rotate_limit=30, p=0.5),  # Translation only
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),  # Brightness adjustment
    ])

# Define brightness/contrast augmentation that will apply only to the image
def get_brightness_contrast_aug():
    return A.Compose([
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),  # Brightness adjustment only
    ], is_check_shapes=False)
    
def train_model(k_folds=5):
    # x_train_dir = '/notebooks/triage/hemofpn/data/train/images' 
    # y_train_dir = '/notebooks/triage/hemofpn/data/train/masks'
    # x_valid_dir = '/notebooks/triage/hemofpn/data/val/images'
    # y_valid_dir = '/notebooks/triage/hemofpn/data/val/masks'
    # x_test_dir = '/notebooks/triage/hemofpn/data/test/images'
    # y_test_dir = '/notebooks/triage/hemofpn/data/test/masks'
    
    # x_train_dir = '/notebooks/triage/hemofpn/data_f8/train/images' 
    # y_train_dir = '/notebooks/triage/hemofpn/data_f8/train/masks'
    # x_valid_dir = '/notebooks/triage/hemofpn/data_f8/val/images'
    # y_valid_dir = '/notebooks/triage/hemofpn/data_f8/val/masks'
    # x_test_dir = '/notebooks/triage/hemofpn/data_f8/test/images'
    # y_test_dir = '/notebooks/triage/hemofpn/data_f8/test/masks'
    x_dir = '/notebooks/triage/hemofpn/data_f8/train/images'
    y_dir = '/notebooks/triage/hemofpn/data_f8/train/masks'
    
#     Main
#     x_dir = '/notebooks/triage/hemofpn/data_v2/all/images'
#     y_dir = '/notebooks/triage/hemofpn/data_v2/all/masks'

    dataset = Dataset(x_dir, y_dir, augmentation=get_geom_augmentations())

    # K-Fold Cross-Validation
    kfold = KFold(n_splits=k_folds, shuffle=True)
    fold_results = []
    # Generate a unique timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for fold, (train_ids, valid_ids) in enumerate(kfold.split(dataset)):
        print(f"Training fold {fold + 1}/{k_folds}...")

        # Subset datasets for this fold
        train_subsampler = Subset(dataset, train_ids)
        valid_subsampler = Subset(dataset, valid_ids)

        train_loader = DataLoader(train_subsampler, batch_size=4, shuffle=True, num_workers=4)
        valid_loader = DataLoader(valid_subsampler, batch_size=4, shuffle=False, num_workers=4)

        # Model setup
        # model = Model("FPN", "resnet34", in_channels=3, out_classes=1)
        model = ModelWithScheduler("Unet", "resnet34", in_channels=3, out_classes=1)
        # model = ModelWithScheduler()

        # Checkpointing and WandB Logger
        checkpoint_dir = f'/notebooks/triage/hemofpn/checkpoints/unet/{timestamp}/fold_{fold + 1}'
        # Callback to save the top 2 best checkpoints based on validation loss (or any metric you choose)
        best_checkpoint_callback = ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename=f'unet-model-dataf8-augs-lrsch-{timestamp}-fold{fold + 1}-best',
            save_top_k=2,          # Save the top 2 checkpoints
            monitor="val_loss",     # Change to the metric you want to track, e.g., "valid_dataset_iou"
            mode="min",             # Set to "min" for the lowest validation loss, or "max" for IoU
        )

        # Callback to save only the last checkpoint
        last_checkpoint_callback = ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename=f'unet-model-dataf8-augs-lrsch-{timestamp}-fold{fold + 1}-last',
            save_last=True,         # Save the last checkpoint
        )
        
        wandb_logger = WandbLogger(
            project="Hemobox",
            name=f"Fold_{fold + 1}_{timestamp}",
            log_model=True
        )
        wandb_logger.experiment.config.update({
            "learning_rate": 1e-3,
            "architecture": "Unet",
            "encoder": "resnet34",
            "batch_size": 4,
            "scheduler": "ReduceLROnPlateau",
            "k_folds": k_folds
        })

        # Ensure the checkpoint directory exists
        os.makedirs(best_checkpoint_callback.dirpath, exist_ok=True)
        os.makedirs(last_checkpoint_callback.dirpath, exist_ok=True)

        trainer = pl.Trainer(
            gpus=1,
            max_epochs=100,
            callbacks=[best_checkpoint_callback, last_checkpoint_callback],
            logger=wandb_logger
        )
        
         # Train and validate
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=valid_loader)
        # Validate and collect metrics
        valid_metrics = trainer.validate(model, dataloaders=valid_loader, verbose=False)
        print(f"Validation metrics for fold {fold + 1}: {valid_metrics}")  # Debug print
        
        # Append metrics if they are valid
        if valid_metrics and isinstance(valid_metrics, list) and isinstance(valid_metrics[0], dict):
            fold_results.append(valid_metrics[0])
        else:
            print(f"Warning: No valid metrics for fold {fold + 1}, adding default values.")
            fold_results.append({'valid_dataset_iou': 0, 'valid_per_image_iou': 0})  # Default values in case of error

        # Close the current WandB run for the fold
        wandb.finish()

    # Calculate average performance across folds
    avg_valid_metrics = {k: np.mean([fold[k] for fold in fold_results]) for k in fold_results[0]}
    print(f"Average validation metrics across folds: {avg_valid_metrics}")
        

if __name__ == "__main__":
    train_model(k_folds=5)
