from collections import OrderedDict

import torch
import torch.nn as nn


class D4UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, init_features=32):
        super(D4UNet, self).__init__()

        features = init_features
        self.encoder1 = D4UNet._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = D4UNet._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = D4UNet._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = D4UNet._block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = D4UNet._block(features * 8, features * 16, name="bottleneck")

        self.upconv4 = nn.ConvTranspose2d(
            features * 16, features * 8, kernel_size=2, stride=2
        )
        self.decoder4 = D4UNet._block((features * 8) * 2, features * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose2d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.decoder3 = D4UNet._block((features * 4) * 2, features * 4, name="dec3")
        # self.upconv2 = nn.ConvTranspose2d(
        #     features * 4, features * 2, kernel_size=2, stride=2
        # )
        # self.decoder2 = D4UNet._block((features * 2) * 2, features * 2, name="dec2")
        # self.upconv1 = nn.ConvTranspose2d(
        #     features * 2, features, kernel_size=2, stride=2
        # )
        # self.decoder1 = D4UNet._block(features * 2, features, name="dec1")

        self.adapter = D4UNet._block(features * 4, features, name="ada")

        self.heads = {"hm": 1, "haf": 1, "vaf": 2}
        for head in self.heads:
            classes = self.heads[head]
            fc = nn.Conv2d(features, classes,
                           kernel_size=1, stride=1, bias=True)
            self.__setattr__(head, fc)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))  # /2
        enc3 = self.encoder3(self.pool2(enc2))  # /4
        enc4 = self.encoder4(self.pool3(enc3))  # /8

        bottleneck = self.bottleneck(self.pool4(enc4))  # /16

        dec4 = self.upconv4(bottleneck)  # /8
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)  # /4
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)

        feat = self.adapter(dec3)
        # dec2 = self.upconv2(dec3)
        # dec2 = torch.cat((dec2, enc2), dim=1)
        # dec2 = self.decoder2(dec2)
        # dec1 = self.upconv1(dec2)
        # dec1 = torch.cat((dec1, enc1), dim=1)
        # dec1 = self.decoder1(dec1)
        z = {}
        for head in self.heads:
            z[head] = self.__getattr__(head)(feat)
        return z

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                ]
            )
        )


if __name__ == '__main__':
    u = D4UNet(in_channels=3, out_channels=3, init_features=32)
    p = torch.rand(1, 3, 512, 512)
    r = u(p)

    for k, v in r.items():
        print(k, v.shape)
