import os
from tempfile import tempdir
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
        temp_gau = cv2.GaussianBlur(temp_gau, (3, 3), 0)
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

def bilateral_cv(ori_image, down_times=5):
    temp_bil = ori_image.copy()
    bilateral_pyramid = [temp_bil]
    for i in range(down_times):
        temp_bil = cv2.bilateralFilter(temp_bil, 40, 75, 75)
        temp_bil = cv2.resize(temp_bil, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST)
        bilateral_pyramid.append(temp_bil)
    return bilateral_pyramid

def direct_down(ori_image, down_times=5):
    temp_dir = ori_image.copy()
    direct_pyramid = [temp_dir]
    for i in range(down_times):
        temp_dir = cv2.resize(temp_dir, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST)
        direct_pyramid.append(temp_dir)
    return direct_pyramid


if __name__ == '__main__':
    img_path = "./data/butterfly_noisy.PNG"
    noisy_img = cv2.imread(img_path)

    # cv2.imshow("nosiy image", noisy_img)
    # cv2.waitKey()

    gau_pyramid = gaussian(noisy_img)
    bil_pyramid = bilateral_cv(noisy_img)
    dir_pyramid = direct_down(noisy_img)
    lap_pyramid = laplacian_cv(gau_pyramid)
    lap_pyramid_bil = laplacian_cv(bil_pyramid)

    lap_pyramid_denoise = lap_pyramid.copy()
    for i in range(len(lap_pyramid) - 1):
        lap_pyramid_denoise[i+1] = cv2.bilateralFilter(lap_pyramid[i+1], 2, 75, 75)
        

    # reconstructr
    result_gau = lap_pyramid[0]
    for i in range(len(lap_pyramid) - 1):
        result_gau = cv2.pyrUp(result_gau)
        result_gau = cv2.add(result_gau, lap_pyramid[i+1])
        # result_gau = cv2.bilateralFilter(result_gau, 9, 75, 75)

    # result_bil = lap_pyramid_bil[0]
    # for i in range(len(lap_pyramid_bil) - 1):
    #     result_bil = cv2.pyrUp(result_bil)
    #     result_bil = cv2.add(result_bil, lap_pyramid_bil[i+1])

    # cv2.imwrite("result_bil.png", result_bil)
    cv2.imwrite("result_gau.png", result_gau)


    for layer in gau_pyramid:
        cv2.imwrite("./data/gau_custom_" + str(gau_pyramid.index(layer)) + ".PNG", layer)

    for layer in lap_pyramid_denoise:
        cv2.imwrite("./data/gau_denoise_" + str(lap_pyramid_denoise.index(layer)) + ".PNG", layer)

    # for layer in dir_pyramid:
    #     cv2.imwrite("./data/dir_" + str(dir_pyramid.index(layer)) + ".PNG", layer)

    for layer in bil_pyramid:
        # cv2.imshow("layer", layer)
        # cv2.waitKey()
        cv2.imwrite("./data/bil_" + str(bil_pyramid.index(layer)) + ".PNG", layer)

    # for layer in lap_pyramid:
    #     cv2.imwrite("./data/lap_" + str(lap_pyramid.index(layer)) + ".PNG", layer)

    for layer in lap_pyramid_bil:
        cv2.imwrite("./data/lap_bil_" + str(lap_pyramid_bil.index(layer)) + ".PNG", layer)