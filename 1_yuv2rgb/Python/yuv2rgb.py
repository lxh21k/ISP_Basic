import os
import numpy as np
import cv2

IMG_HEIGHT = 3072 
IMG_WIDTH = 4080 

def nv21_to_YUV(yuv_bytes):
    """分离出NV21格式的YUV图像通道，返回三个通道对应的矩阵

    :param yuv_bytes: NV21格式图像
    :return: Y,U,V三个通道的矩阵
    """
    uv_h = int(IMG_HEIGHT / 2)
    uv_w = int(IMG_WIDTH / 2)
    
    Y = np.zeros((IMG_HEIGHT, IMG_WIDTH), dtype=np.uint8)
    U = np.zeros((uv_h, uv_w), dtype=np.uint8)
    V = np.zeros((uv_h, uv_w), dtype=np.uint8)

    y_length = IMG_HEIGHT * IMG_WIDTH

    Y[:, :] = yuv_bytes[: y_length].reshape((IMG_HEIGHT, IMG_WIDTH))
    VU = yuv_bytes[y_length :]
    V[:, :] = VU[::2].reshape(uv_h, uv_w)   # VU区域中取奇数索引位置上的值,即V通道的数值
    U[:, :] = VU[1::2].reshape(uv_h, uv_w)  # VU区域中取偶数索引位置上的值,即U通道的数值

    return Y, U, V

def yuv2rgb(Y, U, V):
    """根据YUV三个通道的矩阵转换成RGB通道，并合成RGB图像

    :param Y: Y channel
    :param U: U channel
    :param V: V channel
    :return: RGB image
    """
    bgr_bytes = np.zeros((IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.uint8)
    V = np.repeat(V, 2, 0)
    V = np.repeat(V, 2, 1)
    U = np.repeat(U, 2, 0)
    U = np.repeat(U, 2, 1)

    # 采用bt601色域标准进行转换
    r = Y + 1.14*(V - np.array([128]))
    g = Y - 0.39*(U - np.array([128])) - 0.58*(V - np.array([128]))
    b = Y + 2.03*(U - np.array([128]))

    # 越界的值按照边界处理
    r = np.where(r < 0, 0, r)
    r = np.where(r > 255,255,r)
    g = np.where(g < 0, 0, g)
    g = np.where(g > 255,255,g)
    b = np.where(b < 0, 0, b)
    b = np.where(b > 255,255,b)

    bgr_bytes[:, :, 2] = r
    bgr_bytes[:, :, 1] = g
    bgr_bytes[:, :, 0] = b

    return bgr_bytes


if __name__ == '__main__':

    yuv = "./data/yuvdata/DUMP_2021_0608_082629655_size_4080x3072_4080x3072_r270_input_1_iso_0_ct_0_yuvhdr.nv21"
    print(os.path.getsize(yuv)) 

    with open(yuv, "rb") as yuv_image:
        yuv_bytes = yuv_image.read()
        yuv_bytes = np.frombuffer(yuv_bytes, np.uint8)

        Y, U, V = nv21_to_YUV(yuv_bytes)
        bgr_bytes = yuv2rgb(Y,U,V)

        # cv2.imwrite("y_channel.jpg", Y)
        cv2.imwrite("result_bgr.jpg", bgr_bytes)
