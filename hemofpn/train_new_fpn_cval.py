import os
import re
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset as BaseDataset, Subset
import cv2
import numpy as np
from model import Model
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import wandb
from sklearn.model_selection import KFold

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
        def extract_frame_number(filename):
            match = re.search(r'frame_(\d+)', filename)
            return int(match.group(1)) if match else 0

        #self.ids = [file for file in os.listdir(images_dir) if file.endswith(('.jpg', '.png', '.jpeg')) and not file.startswith('.')]
        #self.ids.sort(key=extract_frame_number)
        self.image_ids = [file for file in os.listdir(images_dir) if file.endswith(('.jpg', '.png', '.jpeg')) and not file.startswith('.')]
        self.image_ids.sort(key=extract_frame_number)
        self.mask_ids = [file for file in os.listdir(masks_dir) if file.endswith(('.jpg', '.png', '.jpeg')) and not file.startswith('.')]
        self.mask_ids.sort(key=extract_frame_number)
        
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.image_ids]
        self.masks_fps = [os.path.join(masks_dir, mask_id) for mask_id in self.mask_ids]
        
        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):
        image = cv2.imread(self.images_fps[i])
        if image is None:
            raise FileNotFoundError(f"Unable to load image at {self.images_fps[i]}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = np.transpose(image, (2, 0, 1))
        image = pad_im(image)

        mask = cv2.imread(self.masks_fps[i], 0)
        mask = convert_mask(mask)
        mask = pad_im(mask)

        return {"image": image, "mask": mask}

    def __len__(self):
        return len(self.image_ids)

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
    
    x_dir = '/notebooks/triage/hemofpn/data_v2/all/images'
    y_dir = '/notebooks/triage/hemofpn/data_v2/all/masks'

    dataset = Dataset(x_dir, y_dir)

    # K-Fold Cross-Validation
    kfold = KFold(n_splits=k_folds, shuffle=True)
    fold_results = []

    for fold, (train_ids, valid_ids) in enumerate(kfold.split(dataset)):
        print(f"Training fold {fold + 1}/{k_folds}...")

        # Subset datasets for this fold
        train_subsampler = Subset(dataset, train_ids)
        valid_subsampler = Subset(dataset, valid_ids)

        train_loader = DataLoader(train_subsampler, batch_size=1, shuffle=True, num_workers=4)
        valid_loader = DataLoader(valid_subsampler, batch_size=1, shuffle=False, num_workers=4)

        # Model setup
        model = Model("FPN", "resnet34", in_channels=3, out_classes=1)

        # Checkpointing and WandB Logger
        checkpoint_callback = ModelCheckpoint(
            dirpath=f'/notebooks/triage/hemofpn/checkpoints/fold_{fold + 1}',
            filename=f'fpn-model-datav2-fold{fold + 1}',
            save_last=True,
        )
        
        wandb_logger = WandbLogger(
            project="Hemobox",
            name=f"Fold_{fold + 1}",
            log_model=True
        )
        
        # Ensure the checkpoint directory exists
        os.makedirs(checkpoint_callback.dirpath, exist_ok=True)

        trainer = pl.Trainer(
            gpus=1,
            max_epochs=100,
            callbacks=[checkpoint_callback],
            logger=wandb_logger
        )
        
         # Train and validate
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=valid_loader)

    # Calculate average performance across folds
    avg_valid_metrics = {k: np.mean([fold[k] for fold in fold_results]) for k in fold_results[0]}
    print(f"Average validation metrics across folds: {avg_valid_metrics}")
        

if __name__ == "__main__":
    train_model(k_folds=5)
