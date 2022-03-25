from dis import dis
import os
import numpy as np
import cv2
from scipy import signal

def display_image(image, title=""):
    """显示图像

    :param image: 图像
    :param title: 图像标题
    """
    image_s = cv2.resize(image, (624, 832))
    cv2.imshow(title, image_s)
    cv2.waitKey()

def linear(cfa, black, white):
    """把RGB三通道的像素值映射到0-1，并进行黑电平校正

    :param cfa: RGB三通道的像素值
    :param black: black level
    :param white: 白点值
    :return: 线性化后的cfa
    """
    cfa = np.clip(cfa, black, white)
    cfa = cfa.astype(np.float32)
    cfa = (cfa - black) / (white - black)
    return cfa

def mask_WB(shape, rwb, bwb, pattern="RGGB"):
    """根据pattern生成mask

    :param shape: mask的shape
    :param rwb: 红色的white balance
    :param bwb: 蓝色的white balance
    :param pattern: pattern
    :return: mask
    """
    mask = np.zeros(shape)
    wb = {'R': rwb, 'G': 1, 'B': bwb}
    for channel, (y, x) in zip(pattern, [(0, 0), (0, 1), (1, 0), (1, 1)]):
        mask[y::2, x::2] = wb[channel]
        
    return mask

def mask_Bayer(shape, pattern="RGGB"):
    """根据pattern生成mask

    :param shape: mask的shape
    :param pattern: pattern
    :return: mask
    """
    mask = dict((channel, np.zeros(shape)) for channel in 'RGB')
    
    for channel, (y, x) in zip(pattern, [(0, 0), (0, 1), (1, 0), (1, 1)]):
        mask[channel][y::2, x::2] = 1
        
    return tuple(mask[c].astype(bool) for c in 'RGB')

if __name__ == '__main__':

    width = 4160
    height = 3120
    
    black = 63
    white = 1023   # 2^10 - 1

    rwb = 2.070468
    bwb = 1.742253
    pattern = 'RGGB'

    path = 'input_raw_dump_4160x3120_input_0_ev0_processTime20210601_142818.raw'
    print(os.path.getsize(path))

    with open(path, "rb") as f:
        raw = f.read()
        cfa = np.frombuffer(raw, dtype=np.uint16)
        cfa = cfa.reshape((height, width))  # color filter array
        # 用16bit来保存每个通道的数据, 但可用bit数只有10~14, 智能手机使用的sensor bit数通常只有10bit, 也就是0~1023
        print(np.amax(cfa))
        print(np.amin(cfa))

        # 线性化/黑电平校正
        cfa_linear = linear(cfa, black, white)
        display_image(cfa_linear, "cfa_linear")
        # cv2.imwrite('cfa_linear.jpg', cfa_s)

        # 白平衡
        wbm = mask_WB(cfa.shape, rwb, bwb, pattern)
        cfa_wb = cfa_linear * wbm
        cfa_wb = np.clip(cfa_wb, 0.0001, 1)
        display_image(cfa_wb, "cfa_wb")

        # LSC, Lens Shading Correction

        # Demosaic
        Rm, Gm, Bm = mask_Bayer(cfa.shape, pattern)

        # bilinear convolution kernel, RB共用
        H_G = np.asarray([[0,1,0], [1,4,1], [0,1,0]]) / 4
        H_RB = np.asarray([[1,2,1], [2,4,2], [1,2,1]]) / 4

        R = signal.convolve2d(cfa_wb*Rm, H_RB, mode='same')
        G = signal.convolve2d(cfa_wb*Gm, H_G, mode='same')
        B = signal.convolve2d(cfa_wb*Bm, H_RB, mode='same')

        R = np.clip(R, 0.0001, 1)
        G = np.clip(G, 0.0001, 1)
        B = np.clip(B, 0.0001, 1)

        demosaic_rgb = np.dstack((B, G, R))
        display_image(demosaic_rgb, "demosaic_rgb")

        # color space conversion, 使用一个3*3的颜色变换矩阵来进行颜色校正
        # srgb转xyz标准色彩空间的矩阵, 这是标准规定的
        srgb2xyz = np.mat([ [0.4124564, 0.3575761, 0.1804375],
                            [0.2126729, 0.7151522, 0.0721750],
                            [0.0193339, 0.1191920, 0.9503041]])
        # 相机空间转xyz色彩空间的矩阵, dng里面有提供
        cam2xyz = np.mat([  [0.7188, 0.1641, 0.0781],
                            [0.2656, 0.8984, -0.1562],
                            [0.0625, -0.4062, 1.1719]])
        # 求逆得到xyz色彩空间转srgb色彩空间的矩阵
        xyz2srgb = srgb2xyz.I
        cam2srgb = cam2xyz * xyz2srgb
        # 保证矩阵每一行元素之和为1
        cam2srgb_norm = cam2srgb / np.repeat(np.sum(cam2srgb, 1), 3).reshape(3, 3)

        r = cam2srgb_norm[0, 0] * R + cam2srgb_norm[0, 1] * G + cam2srgb_norm[0, 2] * B
        g = cam2srgb_norm[1, 0] * R + cam2srgb_norm[1, 1] * G + cam2srgb_norm[1, 2] * B
        b = cam2srgb_norm[2, 0] * R + cam2srgb_norm[2, 1] * G + cam2srgb_norm[2, 2] * B

        csc_rgb = np.dstack((b, g, r))
        display_image(csc_rgb, "csc_rgb")

        # gamma校正
        gamma = 2.2
        gamma_rgb = np.power(csc_rgb, 1/gamma)
        display_image(gamma_rgb, "gamma_rgb")
        
