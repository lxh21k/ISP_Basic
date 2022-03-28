from dis import dis
import os
import numpy as np
import cv2

def mask_LSC(shape, r_lsc_mask, gr_lsc_mask, gb_lsc_mask, b_lsc_mask):

    full_mask = np.zeros(shape)
    channels = ["r", "gr", "b", "gb"]
    lsc_mask = {"r": r_lsc_mask, "gr": gr_lsc_mask, "b": b_lsc_mask, "gb": gb_lsc_mask}
    for channel, (y, x) in zip(channels, [(0, 0), (0, 1), (1, 0), (1, 1)]):
        mask = np.zeros(shape)
        mask[y::2, x::2] = 1
        mask = np.multiply(mask, lsc_mask[channel])
        full_mask = np.add(full_mask, mask)

    return full_mask

def txt2array(data, channel):
    channel_flag = int(np.where(data == (channel+':'))[0])
    channel_mat = data[channel_flag+1: channel_flag+13+1]

    channel_lsc = []

    for line in channel_mat:
        line = line.rstrip()
        line = list(map(int, line.split(' ')))
        channel_lsc.append(line)

    channel_lsc = np.array(channel_lsc)
    channel_lsc = channel_lsc / 1000
    channel_lsc_full = cv2.resize(channel_lsc, (4160, 3120), interpolation=cv2.INTER_LINEAR)

    return channel_lsc_full


if __name__ == '__main__':
    meta_path = "./lsc.txt"
    with open(meta_path, 'r') as f:
        data = f.read().split('\n')
        data = np.array(data)

        r_lsc_mask = txt2array(data, 'r')
        gr_lsc_mask = txt2array(data, 'gr')
        b_lsc_mask = txt2array(data, 'b')
        gb_lsc_mask = txt2array(data, 'gb')
        full_mask = mask_LSC((3120, 4160), r_lsc_mask, gr_lsc_mask, gb_lsc_mask, b_lsc_mask)
        print(full_mask)