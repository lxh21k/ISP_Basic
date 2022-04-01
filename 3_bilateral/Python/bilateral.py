import os
import cv2

def gaussian_cv(ori_image, down_times=5):
    temp_gau = ori_image.copy()
    gaussian_pyramid = [temp_gau]
    for i in range(down_times):
        temp_gau = cv2.pyrDown(temp_gau)
        gaussian_pyramid.append(temp_gau)
    return gaussian_pyramid

def gaussian(ori_image, down_times=5):
    temp_gau = ori_image.copy()
    gaussian_pyramid = [temp_gau]
    for i in range(down_times):
        temp_gau = cv2.resize(temp_gau, (0,0), fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST)
        gaussian_pyramid.append(temp_gau)
    return gaussian_pyramid

def laplacian_cv(gaussian_pyramid, up_times=5):
    laplacian_pyramid = [gaussian_pyramid[-1]]

    for i in range(up_times, 0, -1):
        temp_pyrUp = cv2.pyrUp(gaussian_pyramid[i])
        temp_lap = cv2.subtract(gaussian_pyramid[i-1], temp_pyrUp)
        laplacian_pyramid.append(temp_lap)
    return laplacian_pyramid



if __name__ == '__main__':
    img_path = "./data/butterfly_noisy.PNG"
    noisy_img = cv2.imread(img_path)

    # cv2.imshow("nosiy image", noisy_img)
    # cv2.waitKey()

    gau_pyramid = gaussian_cv(noisy_img)
    # lap_pyramid = laplacian(gauss_pyramid)

    for layer in gau_pyramid:
        cv2.imshow("layer", layer)
        cv2.waitKey()