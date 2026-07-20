import cv2
import numpy as np

#load mask
mask = cv2.imread("/media/storage1/rpatro/datasets/HeadAndNeck/P101/run3/masks/00200.png", cv2.IMREAD_GRAYSCALE)

#check what percentage of image is 255
pct_white = (mask == 255).sum() / mask.size
print(f"Percentage of white pixels: {pct_white * 100:.2f}%")

#verify depth map scaling

d1 = cv2.imread("/media/storage1/rpatro/datasets/HeadAndNeck/P101/run3/depth/00100_depth.png", cv2.IMREAD_UNCHANGED)
d2 = cv2.imread("/media/storage1/rpatro/datasets/HeadAndNeck/P101/run3/depth/00200_depth.png", cv2.IMREAD_UNCHANGED)

print(f"Depth 1 range: min={d1.min()}, max={d1.max()}, mean={d1.mean():.2f}")
print(f"Depth 2 range: min={d2.min()}, max={d2.max()}, mean={d2.mean():.2f}")