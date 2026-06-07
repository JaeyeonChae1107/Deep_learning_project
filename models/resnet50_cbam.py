"""
ResNet50 + CBAM for beef grade classification.
Uses ImageNet pretrained weights from torchvision + official CBAM module.

CBAM is inserted inside each Bottleneck block (after the 3rd conv, before residual add),
which matches the placement in the official model_resnet.py (Jongchan/attention-module).

Official CBAM source: models/attention_module/cbam.py
"""

import torch.nn as nn
import torchvision.models as models

from .attention_module.cbam import CBAM


class _BottleneckWithCBAM(nn.Module):
    """Wraps a torchvision Bottleneck block to insert CBAM before the residual add."""

    expansion = 4

    def __init__(self, bottleneck: nn.Module, planes: int):
        super().__init__()
        self.block = bottleneck
        self.cbam = CBAM(planes * self.expansion, reduction_ratio=16)

    def forward(self, x):
        identity = x
        out = self.block.conv1(x)
        out = self.block.bn1(out)
        out = self.block.relu(out)

        out = self.block.conv2(out)
        out = self.block.bn2(out)
        out = self.block.relu(out)

        out = self.block.conv3(out)
        out = self.block.bn3(out)

        if self.block.downsample is not None:
            identity = self.block.downsample(x)

        out = self.cbam(out)   # CBAM before residual add (official placement)
        out += identity
        out = self.block.relu(out)
        return out


def _inject_cbam_into_layer(layer: nn.Sequential, planes: int) -> nn.Sequential:
    """Replace every Bottleneck in a layer with _BottleneckWithCBAM."""
    new_blocks = []
    for block in layer:
        new_blocks.append(_BottleneckWithCBAM(block, planes))
    return nn.Sequential(*new_blocks)


class ResNet50CBAM(nn.Module):
    def __init__(self, num_classes: int = 5, pretrained: bool = True):
        super().__init__()
        weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.resnet50(weights=weights)

        self.layer0 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)

        # planes = base width per stage; output channels = planes * 4
        self.layer1 = _inject_cbam_into_layer(backbone.layer1, planes=64)   # out: 256
        self.layer2 = _inject_cbam_into_layer(backbone.layer2, planes=128)  # out: 512
        self.layer3 = _inject_cbam_into_layer(backbone.layer3, planes=256)  # out: 1024
        self.layer4 = _inject_cbam_into_layer(backbone.layer4, planes=512)  # out: 2048

        self.avgpool = backbone.avgpool
        self.fc = nn.Linear(backbone.fc.in_features, num_classes)

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x
