import os
import glob
import cv2

# --- SET YOUR PATH ---
DEPTH_DIR = "/media/storage1/rpatro/datasets/HeadAndNeck/P101/run3/depth"

depth_files = sorted(glob.glob(os.path.join(DEPTH_DIR, "*.png")))

print(f"Converting {len(depth_files)} depth maps to single-channel grayscale...")

for filepath in depth_files:
    # IMREAD_GRAYSCALE strips the 3 channels out immediately
    depth_img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    cv2.imwrite(filepath, depth_img)

print("🎉 Channels stripped successfully.")