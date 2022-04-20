import cv2
import numpy as np
from fusion import gaussian_pyramid, laplacian_pyramid, compute_weight, pyramid_reconstruct

if __name__ == "__main__":
    img_fn = ["../images/A.jpg", "../images/B.jpg", "../images/C.jpg", "../images/D.jpg"]
    img_list = [cv2.imread(fn) for fn in img_fn]

    depth = 3

    weights = compute_weight(img_list)

    # img_list = np.stack([img.astype(np.float32) / 255 for img in img_list], axis=0)
    # weights = np.stack([weight.astype(np.float32) / 255 for weight in weights], axis=0)

    img_lap_py = []
    weight_gau_py = []
    for (img, weight) in zip(img_list, weights):
        img_lap_py.append(laplacian_pyramid(img, depth))
        weight_gau_py.append(gaussian_pyramid(weight, depth))

    for layer in range(depth+1):
        cv2.imwrite("./lap_pyramid_%d.jpg" % layer, img_lap_py[0][layer])
        cv2.imwrite("./weight_%d.jpg" % layer, weight_gau_py[0][layer])

    fusion_py = []
    for layer in range(depth+1):
        fusion_layer = np.zeros(img_lap_py[0][layer].shape, dtype=np.float32)
        for n in range(len(img_list)):
            # lap_layer_fl = np.float32(img_lap_py[n][layer])/255
            # lap_layer = lap_layer_fl
            lap_layer = img_lap_py[n][layer]
            weight_layer_fl = np.float32(weight_gau_py[n][depth - layer])/255
            weight_layer = np.dstack((weight_layer_fl, weight_layer_fl, weight_layer_fl))
            fusion_layer += cv2.multiply(lap_layer, weight_layer, dtype=cv2.CV_8UC3)
        fusion_py.append(fusion_layer)

    fusion_py_8bit = []
    for layer in fusion_py:
        # print(layer.shape)
        fusion_py_8bit.append(np.clip(layer, 0, 255).astype(np.uint8))
        # cv2.imwrite("./fusion_py_%d.jpg" % fusion_py.index(layer), fusion_py_8bit[-1])

    # fusion_py_8bit = np.clip(fusion_py[0] * 255, 0, 255).astype(np.uint8)
    fusion_res = pyramid_reconstruct(fusion_py_8bit)

    cv2.imwrite("../images/fusion_res.jpg", fusion_res)