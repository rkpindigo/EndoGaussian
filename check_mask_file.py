import cv2

mask = cv2.imread("/media/storage1/rpatro/repos/MedSAM3/headAndNeck/P101/run3/00001_mask.png", cv2.IMREAD_GRAYSCALE)
print("Mask min value:", mask.min())
print("Mask max value:", mask.max())
print("Unique values:", set(mask.flatten()))