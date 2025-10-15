import torch
import numpy as np
import torchvision.transforms as T
import torchvision.transforms.functional as F
import lightning as L
import wandb

from lightning.pytorch.loggers import WandbLogger
from PIL import Image
from models.METU import lt_MeTU
from datasets import CamVid
from torch.utils.data import DataLoader

image_transform = T.Compose(
    [
        T.Resize((224, 224)),  # 이미지 resize
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),  # 색상 변환
        T.ToTensor(),  # [0,1] float tensor
        T.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]  # ImageNet 기준
        ),
    ]
)

mask_transform = T.Compose(
    [
        T.Resize((224, 224), interpolation=Image.NEAREST),
        T.PILToTensor(),  #
    ]
)


if __name__ == "__main__":

    wandb_logger = WandbLogger(
        project="CamVid Segmentation",
        name="MeTU-xxs",
        log_model=True,
        save_dir="./wandb_logs/MeTU",
    )

    model = lt_MeTU(
        learning_rate=1e-3, model_size="xxs", encoder_pretrained=True, classes=11
    )

    ds_train = CamVid(
        mode="train", transform=image_transform, target_transform=mask_transform
    )
    ds_valid = CamVid(
        mode="val", transform=image_transform, target_transform=mask_transform
    )
    ds_test = CamVid(
        mode="test", transform=image_transform, target_transform=mask_transform
    )

    train_loader = DataLoader(ds_train, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(ds_valid, batch_size=32, shuffle=False, num_workers=4)

    trainer = L.Trainer(
        max_epochs=50,
        accelerator="auto",
        devices=1,
        precision=16,
        logger=wandb_logger,  # this connects everything to wandb
        log_every_n_steps=50,
        callbacks=[
            # Add model checkpoint callback that also logs to wandb
            L.pytorch.callbacks.ModelCheckpoint(
                dirpath="checkpoints/CamVid/MeTU",
                filename="MeTU-xxs-{epoch:02d}-{val/mIoU:.3f}",
                monitor="val/mIoU",
                mode="max",
                save_top_k=3,
            ),
            # Early stopping
            L.pytorch.callbacks.EarlyStopping(
                monitor="val/loss", patience=10, mode="min"
            ),
        ],
    )

    # Train the model - metrics will be logged to wandb automatically
    trainer.fit(model, train_loader, val_loader)

    # Test the model
    trainer.test(model, val_loader)

    # Finish the wandb run
    wandb.finish()
