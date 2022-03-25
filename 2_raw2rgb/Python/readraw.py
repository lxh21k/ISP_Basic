import numpy as np
fd = open('input_raw_dump_4160x3120_input_0_ev0_processTime20210601_142818.raw','rb')
rows = 4160 
cols = 3120
f = np.fromfile(fd, dtype=np.uint8, count=rows*cols)
im = f.reshape((cols, rows))
fd.close

import cv2
im_s = cv2.resize(im, (416, 312))
cv2.imshow('', im_s)
cv2.waitKey()
cv2.destoryAllWindows()
