"""
import os
import json
import glob
import numpy as np

root = "/media/storage1/rpatro/repos/EndoGaussian/data/scared/dataset_1/keyframe_3/data"

frame_data_dir = os.path.join(root, "frame_data")
image_dir = os.path.join(root, "rgb_data")

output = {
    "camera_angle_x": None,
    "frames": []
}

# sort to keep consistent ordering
json_files = sorted(glob.glob(os.path.join(frame_data_dir, "*.json")))

for i, jf in enumerate(json_files):
    with open(jf, 'r') as f:
        data = json.load(f)

    # --- intrinsics ---
    KL = np.array(data["camera-calibration"]["KL"])
    fx = KL[0, 0]
    cx = KL[0, 2]

    # compute camera_angle_x once
    if output["camera_angle_x"] is None:
        width = 1280  # adjust if needed
        output["camera_angle_x"] = 2 * np.arctan(width / (2 * fx))

    # --- pose ---
    c2w = np.array(data["camera-pose"])

    # --- image path ---
    img_name = f"{i:06d}.png"
    img_path = os.path.join("images", img_name)

    output["frames"].append({
        "file_path": img_path,
        "transform_matrix": c2w.tolist()
    })

# save
with open(os.path.join(root, "transforms.json"), "w") as f:
    json.dump(output, f, indent=2)
"""

"""

import torch
import cv2
import os

model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
model.eval()

transform = torch.hub.load("intel-isl/MiDaS", "transforms").small_transform

input_dir = "data/scared/dataset_1/keyframe_2/data/images"
output_dir = "data/scared/dataset_1/keyframe_2/data/left_monodam"
os.makedirs(output_dir, exist_ok=True)

for img_name in sorted(os.listdir(input_dir)):
    img_path = os.path.join(input_dir, img_name)
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    input_batch = transform(img)
    print(input_batch.shape)

    with torch.no_grad():
        prediction = model(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    depth = prediction.cpu().numpy()

    # normalize to 0–255
    depth = (depth - depth.min()) / (depth.max() - depth.min())
    depth = (depth * 255).astype("uint8")

    out_path = os.path.join(output_dir, img_name)
    cv2.imwrite(out_path, depth)

"""


import os
import cv2
from tqdm import tqdm

# Paths - Update these to your actual directories
folders = [
    'data/scared/dataset_1/keyframe_2/data/images',
    'data/scared/dataset_1/keyframe_3/data/images',
]

for folder_path in folders:
    print(f"Processing folder: {folder_path}")
    # Create a backup folder just in case
    output_path = folder_path + "_cropped"
    os.makedirs(output_path, exist_ok=True)
    
    for filename in tqdm(os.listdir(folder_path)):
        if filename.endswith(".png"):
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            
            if img is None:
                continue
            
            # Get dimensions
            h, w = img.shape[:2]
            
            # Crop the top half (0 to h/2)
            # This turns (2048, 1280) into (1024, 1280)
            cropped_img = img[0:h//2, :]
            
            cv2.imwrite(os.path.join(output_path, filename), cropped_img)

print("Done! Replace your original folders with the '_cropped' versions.")
