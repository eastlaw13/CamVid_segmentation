import os
import cv2
import torch
import numpy as np
from models.METU import lt_MeTU
import torchvision.transforms as T
from PIL import Image

CKPT_DIR = "./checkpoints/CamVid/MeTU/MeTU-xxs-epoch=48-val/mIoU=0.480.ckpt"

colormap = {
    0: (128, 128, 128),
    1: (192, 0, 128),
    2: (0, 0, 64),
    3: (64, 192, 128),
    4: (0, 0, 192),
    5: (192, 192, 0),
    6: (128, 128, 64),
    7: (64, 64, 128),
    8: (64, 0, 192),
    9: (64, 128, 64),
    10: (192, 0, 192),
    -1: (0, 0, 0),
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = lt_MeTU.load_from_checkpoint(CKPT_DIR)
model.eval()
model.freeze()

num_classes = 11
np.random.seed(42)

transform_img = T.Compose(
    [
        T.Resize((224, 224)),
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


transform_mask = T.Compose(
    [
        T.Resize((224, 224), interpolation=Image.NEAREST),
        T.PILToTensor(),
    ]
)


def decode_mask(mask_tensor):
    mask_np = mask_tensor.numpy()  # (H,W)
    h, w = mask_np.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for cls in range(num_classes):
        color_mask[mask_np == cls] = colormap[cls]
    return Image.fromarray(color_mask)


img_path = "../../data/CamVid/images/test/image_0.png"
gt_path = "../../data/CamVid/masks/test/image_0.png"
output_dir = "./results"

if not os.path.exists(output_dir):
    os.makedirs(output_dir, exist_ok=True)

fname = "MeTU-xxs.png"


img = Image.open(img_path).convert("RGB")
img_tensor = transform_img(img).unsqueeze(0).to(device)  # (1,3,H,W)


with torch.no_grad():
    logits = model(img_tensor)
    pred_mask = logits.argmax(dim=1).squeeze(0).cpu()  # (H,W)


gt_mask = Image.open(gt_path).convert("I")
gt_mask = transform_mask(gt_mask).squeeze(0).long()  # (H,W)


pred_color = decode_mask(pred_mask)
gt_color = decode_mask(gt_mask)


combined = Image.new("RGB", (img.width * 3, img.height))
img_resized = img.resize((img.width, img.height))
pred_color_resized = pred_color.resize((img.width, img.height))
gt_color_resized = gt_color.resize((img.width, img.height))
combined.paste(img_resized, (0, 0))
combined.paste(gt_color_resized, (img.width, 0))
combined.paste(pred_color_resized, (img.width * 2, 0))

combined.save(os.path.join(output_dir, fname))
print(f"Saved comparison: {fname}")
