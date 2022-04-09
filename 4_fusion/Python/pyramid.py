import cv2
from fusion import gaussian_pyramid, laplacian_pyramid

img = cv2.imread("/Users/leo/Documents/Codes/ISP_Basic/4_fusion/images/A.jpg")

lap_py = laplacian_pyramid(img, depth=5)

for layer in lap_py:
    cv2.imwrite("./lap_pyramid_%d.jpg" % lap_py.index(layer), layer)