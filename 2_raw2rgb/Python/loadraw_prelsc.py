from dis import dis
import os
import numpy as np
import cv2
from scipy import signal

def display_Image(image, title=""):
    """显示图像

    :param image: 图像
    :param title: 图像标题
    """
    image_s = cv2.resize(image, (416, 312)) # 分辨率过大的时候笔记本会卡死
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

def lsc_Channel(data, channel):
    """对meta data中的矩阵进行数值压缩和插值放大，生成对应通道的LSC矩阵

    :param data: 从txt中读取的lsc相关数据
    :param channel: 通道 "r", "gr", "gb", "b"
    :return: 对应通道的lsc矩阵 (3120, 4160)
    """
    channel_flag = int(np.where(data == (channel+':'))[0])
    channel_mat = data[channel_flag+1: channel_flag+13+1]

    channel_lsc = []

    for line in channel_mat:
        line = line.rstrip()
        line = list(map(int, line.split(' ')))
        channel_lsc.append(line)

    channel_lsc = np.array(channel_lsc)
    channel_lsc = channel_lsc / 1024
    channel_lsc = cv2.resize(channel_lsc, (4160, 3120))

    return channel_lsc

def mask_LSC(shape, r_lsc_mask, gr_lsc_mask, gb_lsc_mask, b_lsc_mask):
    """合并四个通道的LSC矩阵，得到一张和Bayer Raw对应的Mask

    :param shape: 返回的mask的shape
    :param r_lsc_mask: 
    :param gr_lsc_mask: 
    :param gb_lsc_mask: 
    :param b_lsc_mask: 
    :return: mask
    """
    full_lsc_mask = np.zeros(shape)
    channels = ["r", "gr", "b", "gb"]
    lsc_masks = {"r": r_lsc_mask, "gr": gr_lsc_mask, "b": b_lsc_mask, "gb": gb_lsc_mask}
    for channel, (y, x) in zip(channels, [(0, 0), (0, 1), (1, 0), (1, 1)]):
        mask = np.zeros(shape)
        mask[y::2, x::2] = 1
        mask = np.multiply(mask, lsc_masks[channel])
        full_lsc_mask = np.add(full_lsc_mask, mask)

    return full_lsc_mask


def read_Meta(meta_path):
    """ 读取meta文件中的LSC矩阵，并拼接四个通道的LSC矩阵为一个全尺寸的mask

    :param meta_path: meta文件的路径
    :return: LSC矩阵
    """
    with open(meta_path, 'r') as f:
        data = f.read().split('\n')
        data = np.array(data)

        r_lsc_mask = lsc_Channel(data, 'r')
        gr_lsc_mask = lsc_Channel(data, 'gr')
        b_lsc_mask = lsc_Channel(data, 'b')
        gb_lsc_mask = lsc_Channel(data, 'gb')
        full_lsc_mask = mask_LSC((3120, 4160), r_lsc_mask, gr_lsc_mask, gb_lsc_mask, b_lsc_mask)
    
    return full_lsc_mask

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
    meta_path = "./lsc.txt"
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
        # display_Image(cfa_linear, "cfa_linear")

        # LSC, Lens Shading Correction
        full_lsc_mask = read_Meta(meta_path)

        cfa_lsc = cfa_linear * full_lsc_mask

        # cfa_lsc = (cfa_lsc - np.amin(cfa_lsc)) / (np.amax(cfa_lsc) - np.amin(cfa_lsc))
        # for (y, x) in [(0,0), (0,1), (1,0), (1,1)]:
        #     cfa_lsc[y::2, x::2] = (cfa_lsc[y::2, x::2] - np.amin(cfa_lsc[y::2, x::2])) / (np.amax(cfa_lsc[y::2, x::2]) - np.amin(cfa_lsc[y::2, x::2]))

        # print(np.amax(cfa_lsc))
        # cfa_lsc = np.clip(cfa_lsc, 0.0001, 1)

        display_Image(cfa_lsc, "cfa_lsc")

        # 白平衡
        wbm = mask_WB(cfa.shape, rwb, bwb, pattern)
        cfa_wb = cfa_lsc * wbm
        for (y, x) in [(0,0), (0,1), (1,0), (1,1)]:
            cfa_wb[y::2, x::2] = (cfa_wb[y::2, x::2] - np.amin(cfa_wb[y::2, x::2])) / (np.amax(cfa_wb[y::2, x::2]) - np.amin(cfa_wb[y::2, x::2]))
        cfa_wb = np.clip(cfa_wb, 0.0001, 1)
        display_Image(cfa_wb, "cfa_wb")

        # Demosaic
        Rm, Gm, Bm = mask_Bayer(cfa.shape, pattern)
        R = cfa_wb*Rm

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
        display_Image(demosaic_rgb, "demosaic_rgb")

        # color space conversion, 使用一个3*3的颜色变换矩阵来进行颜色校正
        # srgb转xyz标准色彩空间的矩阵, 这是标准规定的
        srgb2xyz = np.mat([ [0.4124564, 0.3575761, 0.1804375],
                            [0.2126729, 0.7151522, 0.0721750],
                            [0.0193339, 0.1191920, 0.9503041]])
        # 相机空间转xyz色彩空间的矩阵, dng里面有提供
        # cam2xyz = np.mat([  [0.7188, 0.1641, 0.0781],
        #                     [0.2656, 0.8984, -0.1562],
        #                     [0.0625, -0.4062, 1.1719]])
        cam2xyz = np.mat([[1.679688, -0.742188, 0.054688],
                          [-0.164062, 1.351562, -0.187500],
                          [0.078125, -0.687500, 1.609375]])
        # 求逆得到xyz色彩空间转srgb色彩空间的矩阵
        xyz2srgb = srgb2xyz.I
        cam2srgb = cam2xyz * xyz2srgb
        # 保证矩阵每一行元素之和为1
        cam2srgb_norm = cam2xyz / np.repeat(np.sum(cam2xyz, 1), 3).reshape(3, 3)

        r = cam2srgb_norm[0, 0] * R + cam2srgb_norm[0, 1] * G + cam2srgb_norm[0, 2] * B
        g = cam2srgb_norm[1, 0] * R + cam2srgb_norm[1, 1] * G + cam2srgb_norm[1, 2] * B
        b = cam2srgb_norm[2, 0] * R + cam2srgb_norm[2, 1] * G + cam2srgb_norm[2, 2] * B

        r = np.clip(r, 0.0001, 1)
        g = np.clip(g, 0.0001, 1)
        b = np.clip(b, 0.0001, 1)

        csc_rgb = np.dstack((b, g, r))
        # display_Image(csc_rgb, "csc_rgb")

        # gamma校正
        gamma = 2.2
        gamma_rgb = np.power(csc_rgb, 1/gamma)
        display_Image(gamma_rgb, "gamma_rgb")
        gamma_rgb = gamma_rgb * 255
        cv2.imwrite('gamma_rgb.jpg', gamma_rgb)
        
