import timm
import torch
import torch.nn as nn
import torchvision as tv
from torchvision.models.detection.backbone_utils import BackboneWithFPN


class Dummy(nn.Module):
    def __init__(self):
        super(Dummy, self).__init__()

    def forward(self, x):
        return x


class AFHeadRes(nn.Module):
    def __init__(self, feat_num, heads):
        super().__init__()
        self._heads = heads
        self._fn = feat_num

        self._dapter = nn.Sequential(
            nn.Conv2d(feat_num, 256, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        for head, nc in self._heads.items():
            fc = nn.Conv2d(64, nc, kernel_size=1, stride=1, bias=True)
            self.__setattr__(head, fc)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self._dapter(x)
        z = {}
        for head in self._heads:
            z[head] = self.__getattr__(head)(x)
        return [z]


class FPNFusion(nn.Module):
    def __init__(self, names, channels, mode='add'):
        super(FPNFusion, self).__init__()
        assert (len(names) == 2)
        assert mode in ["add", "concat"]
        self._names = names

        self._up = nn.Upsample(scale_factor=2)
        self._mode = mode
        if self._mode == "concat":
            self._cv = nn.Sequential(
                nn.Conv2d(channels * 2, channels, kernel_size=5, stride=1, padding=2),
                nn.BatchNorm2d(channels),
                nn.ReLU(),
                nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(channels),
                nn.ReLU(),
            )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, xs):
        x0 = self._up(xs[self._names[0]])
        x1 = xs[self._names[1]]
        if self._mode == "concat":
            out = self._cv(torch.cat([x0, x1], dim=1))
        else:
            out = x0 + x1
        return out


class ResNetAF(nn.Module):
    def __init__(self, heads, pretrained=True):
        super(ResNetAF, self).__init__()
        bb = tv.models.resnet18(pretrained)
        self.bb = nn.Sequential(
            bb.conv1,
            bb.bn1,
            bb.relu,
            bb.maxpool,
            bb.layer1,
            bb.layer2,
            bb.layer3,
            bb.layer4
        )
        self.head = AFHeadRes(512, heads=heads)

    def forward(self, x):
        feat = self.bb(x)
        return self.head(feat)


class FPNAF(nn.Module):
    def __init__(self, heads):
        super(FPNAF, self).__init__()
        self.fpn_neck = FPNFusion(['s32', 's16'], 128, mode='concat')
        self.head = AFHeadRes(128, heads=heads)
        self.fpn = self.__init_fpn__()

    def __init_fpn__(self):
        raise NotImplementedError

    def forward(self, x):
        fpnr = self.fpn(x)
        feat = self.fpn_neck(fpnr)
        return self.head(feat)


class DLAFPNAF(FPNAF):
    def __init__(self, heads):
        super(DLAFPNAF, self).__init__(heads)

    def __init_fpn__(self):
        bb = timm.create_model('dla34', pretrained=True)
        fpn = BackboneWithFPN(bb, return_layers={'level5': 's32', 'level4': 's16'}, in_channels_list=[256, 512],
                              out_channels=128)
        return fpn


class ResFPNAF(FPNAF):
    def __init__(self, heads):
        super(ResFPNAF, self).__init__(heads)

    def __init_fpn__(self):
        bb = tv.models.resnet18(True)
        fpn = BackboneWithFPN(bb, return_layers={'layer4': 's32', 'layer3': 's16'}, in_channels_list=[256, 512],
                              out_channels=128)
        return fpn

class DLAAF(ResNetAF):
    def __init__(self, heads, pretrained=True):
        super(DLAAF, self).__init__(heads, pretrained)
        bb = timm.create_model('dla34', pretrained=True)
        self.bb = nn.Sequential(
            bb.base_layer,
            bb.level0,
            bb.level1,
            bb.level2,
            bb.level3,
            bb.level4,
            bb.level5,
        )


if __name__ == '__main__':
    # m = DLAAF({"hm": 1, "vaf": 2, "haf": 1})
    m = ResFPNAF({"hm": 1, "vaf": 2, "haf": 1})
    d = torch.rand((1, 3, 512, 512))
    for k,v in m(d)[0].items():
        print(k, v.shape)
    # print(m)
