import timm
import torch
import torch.nn as nn
import lightning as L
import torch.nn.functional as F
from torchmetrics.classification import MulticlassJaccardIndex
from torchinfo import summary
from einops import rearrange


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


import torch


def compute_iou_components(preds, targets, num_classes, ignore_index=-1):
    """
    Segmentation 예측값과 정답 마스크로부터 IoU 계산에 필요한 픽셀 수를 누적합니다.

    Args:
        preds (torch.Tensor): 모델의 예측값 (B, H, W). 각 픽셀은 [0, num_classes-1] 범위의 클래스 인덱스.
        targets (torch.Tensor): 정답 마스크 (B, H, W). 각 픽셀은 [0, num_classes-1] 또는 ignore_index.
        num_classes (int): 유효 클래스 개수 (배경 제외 11).
        ignore_index (int): 무시할 픽셀 값 (-1).

    Returns:
        tuple: (intersection_sum, union_sum)
    """

    # ignore_index에 해당하는 픽셀을 마스크합니다.
    print(f"preds: {preds.size()} || targets: {targets.size()}")
    valid_mask = targets != ignore_index

    # 유효 픽셀만 사용합니다.
    preds = preds[valid_mask]
    targets = targets[valid_mask]

    # 텐서를 1차원으로 펼칩니다.
    preds = preds.flatten()
    targets = targets.flatten()

    # IoU 구성 요소를 누적할 텐서를 초기화합니다.
    intersection_sum = torch.zeros(num_classes, dtype=torch.long, device=preds.device)
    union_sum = torch.zeros(num_classes, dtype=torch.long, device=preds.device)

    for cls_idx in range(num_classes):
        # Intersection: True Positive (TP)
        # (예측 == 클래스) & (정답 == 클래스)
        is_pred = preds == cls_idx
        is_target = targets == cls_idx

        intersection = (is_pred & is_target).sum()

        # Union: TP + FP + FN
        # (예측 == 클래스) | (정답 == 클래스)
        union = (is_pred | is_target).sum()

        intersection_sum[cls_idx] = intersection.item()
        union_sum[cls_idx] = union.item()

    return intersection_sum, union_sum


def calculate_mIoU(intersection_sum, union_sum):
    """
    누적된 Intersection 및 Union 값을 사용하여 mIoU를 계산합니다.
    """
    # 분모(union_sum)가 0인 경우를 처리합니다. (0/0 = NaN 방지)
    # 0/0은 0으로 간주하거나, 계산에서 제외할 수 있습니다. 여기서는 0으로 처리합니다.

    # 텐서를 float로 변환하여 나누기 연산을 준비합니다.
    intersection_f = intersection_sum.float()
    union_f = union_sum.float()

    # 0으로 나누는 것을 방지하기 위해 where 조건을 사용합니다.
    iou_per_class = torch.where(
        union_f > 0, intersection_f / union_f, torch.tensor(0.0, device=union_f.device)
    )

    # 평균 IoU (mIoU)를 계산합니다.
    # 클래스가 0인 경우가 있었다면, mIoU 계산에서 제외할 수 있습니다.
    valid_classes = (union_f > 0).sum().item()

    if valid_classes == 0:
        return torch.tensor(
            0.0, device=union_f.device
        )  # 전체 배치가 ignore_index만 포함했을 경우

    # 유효 클래스의 IoU 평균
    m_iou = iou_per_class.sum() / valid_classes

    return m_iou


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
        # self.train_iou = MulticlassJaccardIndex(num_classes=11, ignore_index=-1)
        # self.val_iou = MulticlassJaccardIndex(num_classes=11, ignore_index=-1)
        # self.test_iou = MulticlassJaccardIndex(num_classes=11, ignore_index=-1)
        self.model = MeTU(
            model_size=self.model_size,
            encoder_pretrained=self.encoder_pretrained,
            classes=self.classes,
        )

    def _init_iou_components(self):
        # 유효 클래스 수 (classes=11로 가정)
        num_classes = self.classes

        # 훈련 단계
        # nn.Module의 register_buffer를 사용하여 변수를 GPU로 자동 전송하게 합니다.
        self.register_buffer(
            "train_intersections", torch.zeros(num_classes, dtype=torch.long)
        )
        self.register_buffer("train_unions", torch.zeros(num_classes, dtype=torch.long))

        # 검증 단계
        self.register_buffer(
            "val_intersections", torch.zeros(num_classes, dtype=torch.long)
        )
        self.register_buffer("val_unions", torch.zeros(num_classes, dtype=torch.long))

        # 테스트 단계
        self.register_buffer(
            "test_intersections", torch.zeros(num_classes, dtype=torch.long)
        )
        self.register_buffer("test_unions", torch.zeros(num_classes, dtype=torch.long))

    def _reset_iou_components(self, stage):
        """IoU 누적 변수 초기화"""
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
        inter, union = compute_iou_components(
            preds, msks, self.classes, ignore_index=-1
        )

        self.train_unions += union
        self.train_intersections += inter

        self.log("train/loss", loss, on_step=False, on_epoch=True)
        self.log("train/lr", self.optimizers().param_groups[0]["lr"])

        return loss

    def on_train_epoch_end(self):
        m_iou = calculate_mIoU(self.train_intersections, self.train_unions)
        self.log("train/mIoU", m_iou, on_epoch=True)
        self._reset_iou_components("train")

    def validation_step(self, batch, batch_idx):
        imgs, msks = batch
        logits = self(imgs)
        loss = F.cross_entropy(logits, msks, ignore_index=-1)

        preds = torch.argmax(logits, dim=1)
        inter, union = compute_iou_components(
            preds, msks, self.classes, ignore_index=-1
        )

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

        m_iou = calculate_mIoU(self.val_intersections, self.val_unions)
        self.log("val/mIoU", m_iou, on_epoch=True)
        self._reset_iou_components("val")

    def test_step(self, batch, batch_idx):
        imgs, msks = batch
        logits = self(imgs)
        loss = F.cross_entropy(logits, msks)
        preds = logits.argmax(dim=1)

        self.test_step_outputs.append({"loss": loss.detach()})
        inter, union = compute_iou_components(
            preds, msks, self.classes, ignore_index=-1
        )

        self.test_intersections += inter
        self.test_unions += union
        return loss

    def on_test_epoch_end(self):
        if not self.test_step_outputs:
            return
        m_iou = calculate_mIoU(self.test_intersections, self.test_unions)
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
