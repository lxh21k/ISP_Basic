import os
import numpy as np
import cv2

IMG_HEIGHT = 3072 
IMG_WIDTH = 4080 

def nv21_to_YUV(yuv_bytes):
    uv_h = int(IMG_HEIGHT / 2)
    uv_w = int(IMG_WIDTH / 2)
    
    Y = np.zeros((IMG_HEIGHT, IMG_WIDTH), dtype=np.uint8)
    U = np.zeros((uv_h, uv_w), dtype=np.uint8)
    V = np.zeros((uv_h, uv_w), dtype=np.uint8)

    y_length = IMG_HEIGHT * IMG_WIDTH
    uv_length = uv_h * uv_w * 2

    Y[:, :] = yuv_bytes[: y_length].reshape((IMG_HEIGHT, IMG_WIDTH))
    VU = yuv_bytes[y_length :]
    V[:, :] = VU[::2].reshape(uv_h, uv_w)   # VU区域中取奇数索引位置上的值,即V通道的数值
    U[:, :] = VU[1::2].reshape(uv_h, uv_w)  # VU区域中取偶数索引位置上的值,即U通道的数值

    return Y, U, V

def yuv2rgb(Y, U, V):
    bgr_data = np.zeros((IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.uint8)
    V = np.repeat(V, 2, 0)
    V = np.repeat(V, 2, 1)
    U = np.repeat(U, 2, 0)
    U = np.repeat(U, 2, 1)

    c = (Y - np.array([16])) * 298
    d = U - np.array([128])
    e = V - np.array([128])

    r = (c + 409 * e + 128) // 256
    g = (c - 100 * d - 208 * e + 128) // 256
    b = (c + 516 * d + 128) // 256

    r = np.where(r < 0, 0, r)
    r = np.where(r > 255,255,r)

    g = np.where(g < 0, 0, g)
    g = np.where(g > 255,255,g)

    b = np.where(b < 0, 0, b)
    b = np.where(b > 255,255,b)

    bgr_data[:, :, 2] = r
    bgr_data[:, :, 1] = g
    bgr_data[:, :, 0] = b

    return bgr_data

if __name__ == '__main__':
    yuv = "./data/yuvdata/DUMP_2021_0608_082629655_size_4080x3072_4080x3072_r270_input_1_iso_0_ct_0_yuvhdr.nv21"
    print(os.path.getsize(yuv))

    with open(yuv, "rb") as yuv_image:
        yuv_bytes = yuv_image.read()
        yuv_bytes = np.frombuffer(yuv_bytes, np.uint8)
        Y, U, V = nv21_to_YUV(yuv_bytes)
        bgr_data = yuv2rgb(Y,U,V)
        print(bgr_data.shape)
        # print(np.amax(Y))     # 8-bit 数值范围为0-255
        # cv2.imwrite("y_channel.jpg", Y)
        
        cv2.imwrite("result_bgr.jpg", bgr_data)
