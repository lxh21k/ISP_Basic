from scipy import signal

import numpy as np
import cv2

input = cv2.imread("in.png")

kernel = np.asarray([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]]) / 9

output = np.zeros(input.shape)

print(output[:, :, 0].shape)

for i in range(input.shape[-1]):
    output[:, :, i] = signal.convolve2d(input[:, :, i], kernel, mode='same')

cv2.imwrite("out1.png", output)