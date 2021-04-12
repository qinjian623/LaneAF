import torch.nn as nn


class AFHead(nn.Module):
    def __init__(self, feat_num, heads):
        super().__init__()
        self._heads = heads
        for head, nc in self._heads.items():
            fc = nn.Conv2d(feat_num, nc, kernel_size=1, stride=1, bias=True)
            self.__setattr__(head, fc)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        z = {}
        for head in self._heads:
            z[head] = self.__getattr__(head)(x)
        return [z]


if __name__ == '__main__':
    h = AFHead(256, {"hm": 1, "haf": 1, "vaf": 2})
    import torch
    x = torch.rand(1, 256, 224, 224)
    r = h(x)
    for n, t in r[0].items():
        print(n, t.shape)
