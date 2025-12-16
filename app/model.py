import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights 
import torch.nn.functional as F

class ResNetColorizer(nn.Module):
    def __init__(self, n_output=2):
        super(ResNetColorizer, self).__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        self.encoder0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu) # 128 res 64 channels
        self.encoder1 = nn.Sequential(resnet.maxpool, resnet.layer1)  # 128 res 64 channels
        self.encoder2 = resnet.layer2  # 64 res 128 channels
        self.encoder3 = resnet.layer3  # 32 res 256 channels
        self.encoder4 = resnet.layer4  # 16 res 512 channels

        self.up1 = self._upsample_conv(512, 256)
        self.conv1 = self._double_conv(256 + 256, 256) 

        self.up2 = self._upsample_conv(256, 128)
        self.conv2 = self._double_conv(128 + 128, 128)

        self.up3 = self._upsample_conv(128, 64)
        self.conv3 = self._double_conv(64 + 64, 64)

        self.up4 = self._upsample_conv(64, 64)
        self.conv4 = self._double_conv(64 + 64, 64)

        self.final = nn.Sequential(
            nn.Conv2d(64, n_output, kernel_size=1),
            nn.Tanh()
        )

    def _upsample_conv(self, in_c, out_c):
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

    def _double_conv(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x_input = x.repeat(1, 3, 1, 1)

        e0 = self.encoder0(x_input) 
        e1 = self.encoder1(e0)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        d1 = self.up1(e4) # 16
        d1 = torch.cat([d1, e3], dim=1)
        d1 = self.conv1(d1) # 32

        d2 = self.up2(d1) # 32
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.conv2(d2) # 64

        d3 = self.up3(d2)
        d3 = torch.cat([d3, e1], dim=1)
        d3 = self.conv3(d3) # 128

        d4 = self.up4(d3)
        d4 = torch.cat([d4, e0], dim=1)
        d4 = self.conv4(d4) # 256

        out = self.final(d4)
        out = F.interpolate(out, size=x.shape[2:], mode='bilinear', align_corners=True) # stretch to 512
        return out