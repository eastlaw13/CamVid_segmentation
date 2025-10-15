import os
import re
import numpy as np
import torchvision.transforms as transform
from PIL import Image
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

DATA_PATH = "../../data/CamVid"


def sorter(text):
    num = re.findall(r"\d+", text)
    return int(num[0]) if num else 0


class CamVid(Dataset):
    """
    Camvid dataset class.

    Args:
        mode (str): 'train', 'val', 'test'
        transform: Image transform
        target transform: Mask image transform
    """

    def __init__(self, mode="train", transform=None, target_transform=None):
        self.mode = mode
        self.img_path = DATA_PATH + f"/images/{self.mode}"
        self.msk_path = DATA_PATH + f"/masks/{self.mode}"
        self.img_list = sorted(os.listdir(self.img_path), key=sorter)
        self.msk_list = sorted(os.listdir(self.msk_path), key=sorter)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.img_path, self.img_list[idx])).convert("RGB")
        msk = Image.open(os.path.join(self.msk_path, self.msk_list[idx])).convert("I")
        if self.transform:
            img = self.transform(img)

        if self.target_transform:
            msk = self.target_transform(msk).squeeze(0).long()

        return img, msk


if __name__ == "__main__":
    ds_train = CamVid(mode="train")
    img, msk = ds_train[0]
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(msk)
    plt.axis("off")

    plt.show()
