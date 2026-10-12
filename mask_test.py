import cv2, numpy as np
m = cv2.imread(r"C:/Users/sjann/Downloads/digantara/ssa_annotation/output/masks/input1_mask.png", cv2.IMREAD_UNCHANGED)
print("pixels per class:", np.bincount(m.ravel(), minlength=3))   # background, blob, streak
vis = np.zeros((*m.shape, 3), np.uint8)
vis[m == 1] = (255, 90, 90)    # blobs: purple/blue (BGR)
vis[m == 2] = (60, 60, 255)    # streaks: red
vis = cv2.dilate(vis, np.ones((5, 5), np.uint8))   # thicken so tiny stars are visible
cv2.imwrite("mask_visible.png", vis)