import os
import re
import torch
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from segmentation_models_pytorch.losses import DiceLoss

def get_model():
    model = smp.FPN(
        encoder_name="resnet34",        # Choose encoder, e.g., mobilenet_v2 or efficientnet-b7
        encoder_weights="imagenet",     # Use `imagenet` pre-trained weights for encoder initialization
        classes=1,                      # Model output channels (number of classes in your dataset)
        activation=None                 # Activation function
    )
    return model


class ModelWithScheduler(pl.LightningModule):
    def __init__(self):
        super().__init__()
        # Define your model architecture here
        self.model = Model("FPN", "resnet34", in_channels=3, out_classes=1)
        self.loss_fn = DiceLoss(mode="binary", from_logits=True)  # Same loss as in Model

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch["image"], batch["mask"]
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch["image"], batch["mask"]
        y_hat = self(x)
        val_loss = self.loss_fn(y_hat, y)
        self.log("val_loss", val_loss, on_step=True, on_epoch=True, prog_bar=True)
        # Log IoU or other metrics here as needed
        return {"val_loss": val_loss}

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)  # Initial learning rate
        scheduler = {
            'scheduler': ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True),
            'monitor': 'val_loss',  # Adjust based on the validation metric you're interested in
            'interval': 'epoch',
            'frequency': 1
        }
        return [optimizer], [scheduler]

class Model(pl.LightningModule):

    def __init__(self, arch, encoder_name, in_channels, out_classes, **kwargs):
        super().__init__()
        self.model = smp.create_model(
            arch, encoder_name=encoder_name, in_channels=in_channels, classes=out_classes, **kwargs
        )

        # Preprocessing parameters for image
        params = smp.encoders.get_preprocessing_params(encoder_name)
        self.register_buffer("std", torch.tensor(params["std"]).view(1, 3, 1, 1))
        self.register_buffer("mean", torch.tensor(params["mean"]).view(1, 3, 1, 1))

        # For image segmentation dice loss could be the best first choice
        self.loss_fn = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)

    def forward(self, image):
        # Normalize image here
        image = (image - self.mean) / self.std
        mask = self.model(image)
        return mask

    def shared_step(self, batch, stage):
        image = batch["image"]
        mask = batch["mask"]

        logits_mask = self.forward(image)
        loss = self.loss_fn(logits_mask, mask)

        prob_mask = logits_mask.sigmoid()
        pred_mask = (prob_mask > 0.5).float()

        tp, fp, fn, tn = smp.metrics.get_stats(pred_mask.long(), mask.long(), mode="binary")

        return {
            "loss": loss,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }

    def shared_epoch_end(self, outputs, stage):
        tp = torch.cat([x["tp"] for x in outputs])
        fp = torch.cat([x["fp"] for x in outputs])
        fn = torch.cat([x["fn"] for x in outputs])
        tn = torch.cat([x["tn"] for x in outputs])

        per_image_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro-imagewise")
        dataset_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")

        metrics = {
            f"{stage}_per_image_iou": per_image_iou,
            f"{stage}_dataset_iou": dataset_iou,
        }

        self.log_dict(metrics, prog_bar=True)

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")            

    def training_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "train")

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, "valid")

    def validation_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "valid")

    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, "test")  

    def test_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "test")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.0001)
