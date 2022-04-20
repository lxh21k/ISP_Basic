import cv2
import numpy as np
from fusion import compute_weight

# fusion without using pyramid

if __name__ == "__main__":
    img_fn = ["../images/A.jpg", "../images/B.jpg", "../images/C.jpg", "../images/D.jpg"]
    img_list = [cv2.imread(fn) for fn in img_fn]

    weights = compute_weight(img_list)

    # img_list_fl32 = np.stack([img.astype(np.float32) / 255 for img in img_list], axis=0)
    weights_fl32 = np.stack([weight.astype(np.float32) / 255 for weight in weights], axis=0)
    res = np.zeros(img_list[0].shape, dtype=np.float32)

    for k in range(len(img_list)):
        for c in range(3): # color channels
            res[:, :, c] += np.multiply(img_list[k][:, :, c], weights_fl32[k])

    res_8bit = np.clip(res, 0, 255).astype(np.uint8)
    cv2.imwrite("./fusion.jpg", res)

    for i in range(len(weights)):
        cv2.imwrite("./weight_%d.jpg" % i, weights[i])