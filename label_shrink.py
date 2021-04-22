import cv2
import numpy as np
import timm
from torchvision.models.detection.backbone_utils import BackboneWithFPN

from datasets.transforms import GroupRandomRotation
from models.raw_resnet import FPNFusion

model_names = timm.list_models(pretrained=True)
for n in model_names:
    print(n)
m = timm.create_model("dla34", pretrained=True)
print(m)
fpn = BackboneWithFPN(m, return_layers={'level5': 's32', 'level4': 's16'}, in_channels_list=[256, 512],
                      out_channels=128)
import torch
input = torch.rand((1, 3, 1024, 512))
r = fpn(input)
for k, v in r.items():
    print(k, v.shape)
ff = FPNFusion(['s32', 's16'], 128, mode='concat')
print(ff(r).shape)
# print(fpn)
exit()


def shrink(im, size):
    s_w, s_h = size
    h, w, _ = im.shape
    ret = np.zeros((s_h, s_w))
    stride_w = w // s_w
    stride_h = h // s_h
    for i in range(0, h, stride_h):
        for j in range(0, w, stride_w):
            patch = im[i:i + stride_h, j:j + stride_w, :]
            # print(patch.max())
            if i // stride_h >= s_h or j // stride_w >= s_w:
                continue
            ret[i // stride_h, j // stride_w] = patch.max()
    return ret


t = GroupRandomRotation(degree=(-1, 1), padding=(122, 0))
im = cv2.imread("/home/jian/02450.png")
ret = t((im, im))
kk = cv2.resize(ret[1], (32, 16), cv2.INTER_MAX)
print(kk[:, :, 0].max(), kk.min())
ret = shrink(ret[1], (32, 16))

print(np.abs(kk[:, :, 2] - ret).sum())
exit()

# print(ret.max())

cv2.imwrite("/home/jian/02450_seg_not.png", ret.astype(np.uint8) * 60)
# cv2.imshow("K", ret)
# cv2.waitKey(10)
# cv2.imwrite("/home/jian/02450_seg.png", ret[1].astype(np.uint8))


ret = ret.astype(np.uint8)
from utils.affinity_fields import generateAFs

a, b = generateAFs(ret, viz=True)
