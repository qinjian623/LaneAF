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
    def __init__(self, names, channels, mode='concat'):
        super(FPNFusion, self).__init__()
        assert (len(names) == 2 or len(names) == 3)
        assert mode in ["add", "concat"]
        self._names = names

        self._up = nn.Upsample(scale_factor=2)
        self._mode = mode
        if self._mode == "concat":
            if len(self._names) == 2:
                self._cv = nn.Sequential(
                    nn.Conv2d(channels * 2, channels, kernel_size=5, stride=1, padding=2),
                    nn.BatchNorm2d(channels),
                    nn.ReLU(),
                    nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(channels),
                    nn.ReLU(),
                )
            if len(self._names) == 3:
                self._cv0 = nn.Sequential(
                    nn.Conv2d(channels * 2, channels, kernel_size=5, stride=1, padding=2),
                    nn.BatchNorm2d(channels),
                    nn.ReLU(),
                    nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(channels),
                    nn.ReLU(),
                )
                self._cv1 = nn.Sequential(
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
        if len(self._names) == 2:
            x0 = self._up(xs[self._names[0]])
            x1 = xs[self._names[1]]
            if self._mode == "concat":
                out = self._cv(torch.cat([x0, x1], dim=1))
            else:
                out = x0 + x1
        if len(self._names) == 3:
            x0 = self._up(xs[self._names[0]])
            x1 = xs[self._names[1]]
            if self._mode == "concat":
                out = self._cv0(torch.cat([x0, x1], dim=1))
            else:
                out = x0 + x1
            x0 = self._up(out)
            x1 = xs[self._names[2]]
            if self._mode == "concat":
                out = self._cv1(torch.cat([x0, x1], dim=1))
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
    def __init__(self, heads, instance_norm=True, stride=8):
        super(FPNAF, self).__init__()
        print("FPN AF Stride: {}".format(stride))
        # TODO xxx
        if stride == 8:
            self.fpn_neck = FPNFusion(['s32', 's16', 's8'], 128, mode='concat')
        elif stride == 16:
            self.fpn_neck = FPNFusion(['s32', 's16'], 128, mode='concat')
        self.head = AFHeadRes(128, heads=heads)
        self.fpn = self.__init_fpn__(stride)
        if instance_norm:
            self.inn = nn.InstanceNorm2d(3, affine=True)
        else:
            self.inn = None

    def __init_fpn__(self):
        raise NotImplementedError

    def forward(self, x):
        if self.inn:
            x = self.inn(x)
        fpnr = self.fpn(x)
        feat = self.fpn_neck(fpnr)
        return self.head(feat)


class DLAFPNAF(FPNAF):
    def __init__(self, heads, instance_norm=False, stride=8):
        super(DLAFPNAF, self).__init__(heads, instance_norm, stride=stride)

    def __init_fpn__(self, stride):
        if stride == 8:
            bb = timm.create_model('dla34', pretrained=True)
            fpn = BackboneWithFPN(bb, return_layers={'level3': 's8', 'level4': 's16', 'level5': 's32'},
                                  in_channels_list=[128, 256, 512],
                                  out_channels=128)
            return fpn
        elif stride == 16:
            bb = timm.create_model('dla34', pretrained=True)
            fpn = BackboneWithFPN(bb, return_layers={'level5': 's32', 'level4': 's16'},
                                  in_channels_list=[256, 512],
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
    m = DLAFPNAF({"hm": 1, "vaf": 2, "haf": 1})
    d = torch.rand((1, 3, 512, 512))
    for k, v in m(d)[0].items():
        print(k, v.shape)
    # print(m)
