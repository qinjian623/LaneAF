import time

import torch
import torch.nn as nn


class SmallNet(nn.Module):
    def __init__(self):
        super(SmallNet, self).__init__()
        self._n = nn.Sequential(
            nn.Conv2d(19, 64, 3, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, 16, 3, 1, 1, bias=False),
        )

    def forward(self, x):
        return self._n(x)


def main():
    data = torch.randn((1, 3 + 16, 576, 1024)).cuda()
    sn = SmallNet().cuda()
    torch.cuda.synchronize()
    s = time.time()
    for i in range(10):
        sn(data)
    torch.cuda.synchronize()
    e = time.time()
    print((e - s)/10)
    print(sn(data).shape)


if __name__ == '__main__':
    main()