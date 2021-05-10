import torch
import torch.nn as nn
import torchvision as tv
import cv2
import matplotlib.pyplot as plt
import numpy as np
m = tv.models.resnet18(True)
m.eval()
inn = nn.InstanceNorm2d(3)
im = cv2.imread('/home/jian/data/00690.jpg')
im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB).astype(np.float)
im = torch.from_numpy(im).permute(2, 0, 1).contiguous().unsqueeze(0)
print(im.shape)
# cv2.imshow("F", im)
# im = cv2.imread('/home/jian/data/02940.jpg')
im = inn(im)
plt.hist(im.flatten())
plt.show()

im = cv2.imread('/home/jian/data/02340.jpg')
im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB).astype(np.float)# cv2.waitKey(0)
im = torch.from_numpy(im).permute(2, 0, 1).contiguous().unsqueeze(0)
im = inn(im)
plt.hist(im.flatten())
plt.show()
