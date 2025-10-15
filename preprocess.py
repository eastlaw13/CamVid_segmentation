import csv
import os
import shutil
import numpy as np
from PIL import Image
from tqdm import tqdm

DATA_PATH = "../../data/CamVid"
new_dict = {
    0: ["Sky"],
    1: ["Building", "Wall", "Bridge", "Tunnel", "Archway"],
    2: ["Column_Pole", "TrafficLight", "TrafficCone"],
    3: ["Road", "LaneMkgsDriv", "LaneMkgsNonDriv", "RoadShoulder", "ParkingBlock"],
    4: ["Sidewalk"],
    5: ["Tree", "VegetationMisc"],
    6: ["SignSymbol", "Misc_Text"],
    7: ["Fence"],
    8: [
        "Car",
        "SUVPickupTruck",
        "Truck_Bus",
        "Train",
        "OtherMoving",
        "CartLuggagePram",
    ],
    9: ["Pedestrian", "Child", "Animal"],
    10: ["Bicyclist", "MotorcycleScooter"],
    -1: ["Void"],  # ignore_index
}

old_dict = {}
with open(DATA_PATH + "/class_dict.csv", newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        cls = row["name"]
        r, g, b = int(row["r"]), int(row["g"]), int(row["b"])
        old_dict[cls] = (r, g, b)


color2name_dict = {values: key for key, values in old_dict.items()}
name2class_dict = {item: key for key, values in new_dict.items() for item in values}

mode = "train"
old_image_path = DATA_PATH + f"/original/{mode}"
old_mask_path = DATA_PATH + f"/original/{mode}_labels"
new_image_path = DATA_PATH + f"/images/{mode}"
new_mask_path = DATA_PATH + f"/masks/{mode}"


def convert_cls(old_mask_path, old_image_path, new_mask_path, new_image_path):
    image_list = os.listdir(old_image_path)
    mask_list = os.listdir(old_mask_path)
    os.makedirs(new_image_path, exist_ok=True)
    os.makedirs(new_mask_path, exist_ok=True)
    i = 0
    for name in mask_list:
        new_name = f"image_{i}.png"
        mask = np.array(Image.open(os.path.join(old_mask_path, name)).convert("RGB"))
        h, w, _ = mask.shape

        new_mask = np.full((h, w), fill_value=-1, dtype=np.int32)

        for color, obj in color2name_dict.items():
            temp = np.all(mask == color, axis=-1)
            new_mask[temp] = name2class_dict[obj]

        Image.fromarray(new_mask, mode="I").save(os.path.join(new_mask_path, new_name))
        base_name = f"{name[:-6]}.png"

        if base_name in image_list:

            shutil.copy2(
                os.path.join(old_image_path, base_name),
                os.path.join(new_image_path, new_name),
            )
        i += 1


for mode in tqdm(["train", "val", "test"]):
    old_image_path = DATA_PATH + f"/original/{mode}"
    old_mask_path = DATA_PATH + f"/original/{mode}_labels"
    new_image_path = DATA_PATH + f"/images/{mode}"
    new_mask_path = DATA_PATH + f"/masks/{mode}"

    convert_cls(old_mask_path, old_image_path, new_mask_path, new_image_path)
