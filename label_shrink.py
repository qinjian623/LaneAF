import cv2
import numpy as np
from mmseg.datasets.hm.utils.transforms import GroupRandomRotation


def shrink(im, size):
    s_w, s_h = size
    h, w, _ = im.shape
    ret = np.zeros((s_h, s_w))
    stride_w = w // s_w
    stride_h = h // s_h
    for i in range(0, h, stride_h):
        for j in range(0, w, stride_w):
            patch = im[i:i+stride_h, j:j+stride_w, :]
            # print(patch.max())
            if i//stride_h >= s_h or j//stride_w >= s_w:
                continue
            ret[i//stride_h, j//stride_w] = patch.max() * 60
    return ret


t = GroupRandomRotation(degree=(100, 180), padding=(122, 255))
im = cv2.imread("/home/jian/02450.png")
ret = t((im, im))

ret = shrink(ret[1], (64, 32))

# print(ret.max())

cv2.imwrite("/home/jian/02450_seg_not.png", ret.astype(np.uint8))
cv2.imshow("K", ret)
cv2.waitKey(0)
# cv2.imwrite("/home/jian/02450_seg.png", ret[1].astype(np.uint8))
