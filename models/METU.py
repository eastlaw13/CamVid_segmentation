import timm
import torch
import torch.nn as nn
import lightning as L
import torch.nn.functional as F
from torchinfo import summary
from util import iou_component, iou_calculation


class MobileViTEncoder(nn.Module):
    """
    Pretrained MobileViT v1 with Imagnet-1k based Encoder from timm API.

    Args:
        model_size (str): 'xxs', 'xs', 's'
        pretrained (bool): If True, load ImageNet pretrained weights
    """

    def __init__(self, model_size="xxs", pretrained=True):
        super().__init__()
        self.model_size = model_size
        self.pretrained = pretrained
        self.full_model = timm.create_model(
            f"mobilevit_{self.model_size}", pretrained=self.pretrained
        )
        self.stem = self.full_model.stem  # Output: [1 16, 112, 112]
        self.stage1 = self.full_model.stages[0]  # Output: [1 16, 112, 112]
        self.stage2 = self.full_model.stages[1]  # output: [1, 24, 56, 56]
        self.stage3 = self.full_model.stages[2]  # output: [1, 48, 28, 28]
        self.stage4 = self.full_model.stages[3]  # output: [1, 64, 14, 14]
        self.stage5 = self.full_model.stages[4]  # output: [1, 80, 7, 7]
        self.final_conv = self.full_model.final_conv  # output: [1, 320, 7, 7]

    def forward(self, x):
        skips_features = (
            []
        )  # layer name(B,C,W,H): [stage1(1 16, 112, 112), stage2(1, 24, 56, 56]), stage3(1, 48, 28, 28), stage4(1, 64, 14, 14), stage5(1, 80, 7, 7)]
        out = self.stem(x)
        out = self.stage1(out)
        skips_features.append(out)

        out = self.stage2(out)
        skips_features.append(out)

        out = self.stage3(out)
        skips_features.append(out)

        out = self.stage4(out)
        skips_features.append(out)

        out = self.stage5(out)
        skips_features.append(out)

        out = self.final_conv(out)

        return out, skips_features


class DecoderBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.skip_proj = (
            nn.Conv2d(skip_ch, out_ch, kernel_size=1) if skip_ch is not None else None
        )
        self.conv1 = nn.Conv2d(
            in_ch + (out_ch if skip_ch is not None else 0),
            out_ch,
            kernel_size=3,
            padding=1,
        )
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch)

    def forward(self, x, skip=None):
        if skip is not None and self.skip_proj is not None:
            skip = self.skip_proj(skip)
            x = F.interpolate(
                x, size=skip.shape[2:], mode="bilinear", align_corners=False
            )
            x = torch.cat([x, skip], dim=1)
        else:
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return x


class MeTU(nn.Module):
    def __init__(self, model_size="xxs", encoder_pretrained=True, classes=1):
        super().__init__()
        self.encoder = MobileViTEncoder(model_size, encoder_pretrained)
        self.classes = classes

        dummy_input = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            _, skips = self.encoder(dummy_input)
        skip_channels = [f.shape[1] for f in skips]
        bottleneck_ch = self.encoder.final_conv.out_channels

        decoder_config = []
        prev_ch = bottleneck_ch
        for skip_ch in reversed(skip_channels):
            out_ch = max(skip_ch, prev_ch // 2)
            decoder_config.append((prev_ch, skip_ch, out_ch))
            prev_ch = out_ch

        self.decoder_blocks = nn.ModuleList(
            [
                DecoderBlock(in_ch=in_ch, skip_ch=skip_ch, out_ch=out_ch)
                for in_ch, skip_ch, out_ch in decoder_config
            ]
        )

        self.out_conv = nn.Conv2d(decoder_config[-1][2], self.classes, kernel_size=1)

        # Optional: freeze encoder
        for param in self.encoder.parameters():
            param.requires_grad = False

    def forward(self, x):
        input_h, input_w = x.shape[2], x.shape[3]
        bottleneck, skips = self.encoder(x)
        x = bottleneck
        for i, dec_block in enumerate(self.decoder_blocks):
            skip_feat = skips[-(i + 1)]
            x = dec_block(x, skip_feat)

        x = F.interpolate(
            x, size=(input_h, input_w), mode="bilinear", align_corners=False
        )
        out = self.out_conv(x)
        return out


class lt_MeTU(L.LightningModule):
    def __init__(
        self, learning_rate, model_size="xxs", encoder_pretrained=True, classes=1
    ):
        super().__init__()
        self.model_size = model_size
        self.encoder_pretrained = encoder_pretrained
        self.classes = classes
        self.learning_rate = learning_rate
        self.save_hyperparameters()
        self._init_iou_components()
        self.validation_step_outputs = []
        self.test_step_outputs = []

        self.model = MeTU(
            model_size=self.model_size,
            encoder_pretrained=self.encoder_pretrained,
            classes=self.classes,
        )

    def _init_iou_components(self):
        num_classes = self.classes

        self.register_buffer(
            "train_intersections", torch.zeros(num_classes, dtype=torch.long)
        )
        self.register_buffer("train_unions", torch.zeros(num_classes, dtype=torch.long))

        self.register_buffer(
            "val_intersections", torch.zeros(num_classes, dtype=torch.long)
        )
        self.register_buffer("val_unions", torch.zeros(num_classes, dtype=torch.long))

        self.register_buffer(
            "test_intersections", torch.zeros(num_classes, dtype=torch.long)
        )
        self.register_buffer("test_unions", torch.zeros(num_classes, dtype=torch.long))

    def _reset_iou_components(self, stage):
        if stage == "train":
            self.train_intersections.zero_()
            self.train_unions.zero_()
        elif stage == "val":
            self.val_intersections.zero_()
            self.val_unions.zero_()
        elif stage == "test":
            self.test_intersections.zero_()
            self.test_unions.zero_()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        imgs, msks = batch
        logits = self(imgs)
        loss = F.cross_entropy(logits, msks, ignore_index=-1)

        preds = logits.argmax(dim=1)
        inter, union = iou_component(preds, msks, self.classes, ignore_idx=-1)

        self.train_unions += union
        self.train_intersections += inter

        self.log("train/loss", loss, on_step=False, on_epoch=True)
        self.log("train/lr", self.optimizers().param_groups[0]["lr"])

        return loss

    def on_train_epoch_end(self):
        m_iou = iou_calculation(self.train_intersections, self.train_unions)
        self.log("train/mIoU", m_iou, on_epoch=True)
        self._reset_iou_components("train")

    def validation_step(self, batch, batch_idx):
        imgs, msks = batch
        logits = self(imgs)
        loss = F.cross_entropy(logits, msks, ignore_index=-1)

        preds = torch.argmax(logits, dim=1)
        inter, union = iou_component(preds, msks, self.classes, ignore_idx=-1)
        self.val_intersections += inter.to(self.device)
        self.val_unions += union.to(self.device)

        self.validation_step_outputs.append(
            {
                "loss": loss.detach(),
            }
        )

    def on_validation_epoch_end(self):
        if not self.validation_step_outputs:
            return

        avg_loss = torch.stack([x["loss"] for x in self.validation_step_outputs]).mean()
        self.log("val/loss", avg_loss, on_epoch=True)

        m_iou = iou_calculation(self.train_intersections, self.train_unions)

        self.log("val/mIoU", m_iou, on_epoch=True)
        self._reset_iou_components("val")

    def test_step(self, batch, batch_idx):
        imgs, msks = batch
        logits = self(imgs)
        loss = F.cross_entropy(logits, msks)
        preds = logits.argmax(dim=1)

        self.test_step_outputs.append({"loss": loss.detach()})
        inter, union = iou_component(preds, msks, self.classes, ignore_idx=-1)

        self.test_intersections += inter
        self.test_unions += union
        return loss

    def on_test_epoch_end(self):
        if not self.test_step_outputs:
            return
        m_iou = iou_calculation(self.train_intersections, self.train_unions)
        avg_loss = torch.stack([x["loss"] for x in self.test_step_outputs]).mean()

        # Log test metrics to wandb
        self.log("test/loss", avg_loss, on_epoch=True)
        self._reset_iou_components("test")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.learning_rate, weight_decay=1e-4
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=100, eta_min=1e-06
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val/loss"},
        }


if __name__ == "__main__":
    from torchinfo import summary

    model = lt_MeTU(
        model_size="xxs", encoder_pretrained=True, learning_rate=1e-3, classes=11
    )
    summary(model, (1, 3, 224, 224))
