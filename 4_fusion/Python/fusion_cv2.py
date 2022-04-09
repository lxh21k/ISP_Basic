import cv2
import numpy as np

img_fn = ["../images/A.jpg", "../images/B.jpg", "../images/C.jpg", "../images/D.jpg"]
img_list = [cv2.imread(fn) for fn in img_fn]

merge_mertens = cv2.createMergeMertens()
res_mertens = merge_mertens.process(img_list)

res_mertens_8bit = np.clip(res_mertens * 255, 0, 255).astype(np.uint8)
cv2.imwrite("../images/mertens.jpg", res_mertens_8bit)