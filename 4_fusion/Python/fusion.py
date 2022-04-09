import cv2
import numpy as np

def up_sample(img, dst_shape=None):
    kernel = cv2.getGaussianKernel(ksize=5, sigma=0.5)

    if dst_shape is None:
        res = cv2.resize(img, None, fx=2, fy=2)
    else:
        res = cv2.resize(img, dst_shape)

    res = cv2.filter2D(res, cv2.CV_8U, kernel)

    return res

def down_sample(img):
    kernel = cv2.getGaussianKernel(ksize=5, sigma=0.5)

    res = cv2.filter2D(img, cv2.CV_8U, kernel)
    res = cv2.resize(res, None, fx=0.5, fy=0.5)

    return res

def gaussian_pyramid(img, depth):
    layer = img.copy()
    gp = [layer]
    for i in range(depth):
        gp.append(down_sample(gp[-1]))
    return gp

def laplacian_pyramid(img, depth):
    gp = gaussian_pyramid(img, depth)
    lp = [gp[-1]]
    for i in range(depth, 0, -1):
        next_layer_shape = [gp[i-1].shape[1], gp[i-1].shape[0]]
        layer = cv2.subtract(gp[i-1], up_sample(gp[i], dst_shape=next_layer_shape))
        lp.append(layer)
    return lp
    
def compute_weight(imgs):

    # weights of three measures: contrast, saturation, well-exposedness
    (w_c, w_s, w_e) = (1, 1, 1)
    weights = []
    weights_sum = np.zeros(imgs[0].shape[:2], dtype=np.float32)

    for img in imgs:
        img = np.float32(img) / 255
        W = np.ones(img.shape[:2], dtype=np.float32) 

        # contrast 
        img_gary = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        laplacian_img = cv2.Laplacian(img_gary, cv2.CV_32F)   
        W_contrast = np.abs(laplacian_img) ** w_c
        W = np.multiply(W, W_contrast)

        # saturation
        W_saturation = img.std(axis=2, dtype=np.float32) ** w_s
        W = np.multiply(W, W_saturation)

        # well-exposedness
        sigma = 0.2
        W_well_exposedness = np.prod(np.exp(- ((img-0.5) ** 2) / (2 * (sigma ** 2))), axis=2, dtype=np.float32) ** w_e
        W = np.multiply(W, W_well_exposedness)

        weights_sum += W
        weights.append(W)

    nonzero_pix = weights_sum > 0
    for i in range(len(weights)):
        weights[i][nonzero_pix] /= weights_sum[nonzero_pix]
        weights[i] = np.uint8(weights[i] * 255)

    return weights