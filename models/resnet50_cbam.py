"""
ResNet50 + CBAM for beef grade classification.
CBAM is inserted after each residual stage (layer1~layer4),
following the placement convention in the original CBAM paper (Table 1, ResNet variant).
"""

import torch.nn as nn
import torchvision.models as models

from .cbam import CBAM


class ResNet50CBAM(nn.Module):
    def __init__(self, num_classes: int = 5, pretrained: bool = True, reduction_ratio: int = 16):
        super().__init__()
        weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.resnet50(weights=weights)

        self.layer0 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)

        # Each residual stage followed by CBAM
        # Output channels: layer1→256, layer2→512, layer3→1024, layer4→2048
        self.layer1 = backbone.layer1
        self.cbam1 = CBAM(256, reduction_ratio)

        self.layer2 = backbone.layer2
        self.cbam2 = CBAM(512, reduction_ratio)

        self.layer3 = backbone.layer3
        self.cbam3 = CBAM(1024, reduction_ratio)

        self.layer4 = backbone.layer4
        self.cbam4 = CBAM(2048, reduction_ratio)

        self.avgpool = backbone.avgpool
        self.fc = nn.Linear(backbone.fc.in_features, num_classes)

    def forward(self, x):
        x = self.layer0(x)

        x = self.layer1(x)
        x = self.cbam1(x)

        x = self.layer2(x)
        x = self.cbam2(x)

        x = self.layer3(x)
        x = self.cbam3(x)

        x = self.layer4(x)
        x = self.cbam4(x)

        x = self.avgpool(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x
