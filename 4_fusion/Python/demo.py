import cv2
import numpy as np
from fusion import compute_weight

img_fn = ["../images/A.jpg", "../images/B.jpg", "../images/C.jpg", "../images/D.jpg"]
img_list = [cv2.imread(fn) for fn in img_fn]

weights = compute_weight(img_list)

res = np.zeros(img_list[0].shape, dtype=np.float32)

for k in range(len(img_list)):
    for c in range(3):
        res[:, :, c] += cv2.multiply(img_list[k][:, :, c], weights[k])

# res_8bit = np.clip(res * 255, 0, 255).astype(np.uint8)
# cv2.imwrite("./fusion.jpg", res)

for i in range(len(weights)):
    cv2.imwrite("./weight_%d.jpg" % i, weights[i])