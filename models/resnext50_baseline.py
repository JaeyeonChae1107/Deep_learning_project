"""
Model A: ResNeXt-50 Baseline for Beef Grade Classification.

Architecture
------------
Torchvision pretrained ResNeXt-50 32x4d (ImageNet) with the final fully-connected
layer replaced by a 5-class head for beef grading (1++, 1+, 1, 2, 3).

The backbone is kept intact — only the last FC layer is re-initialised.

Training loss
-------------
Standard CrossEntropyLoss (optionally weighted for class imbalance).

Role in ablation
----------------
Establishes the clean-image accuracy baseline and the un-improved
perturbation stability that CBAM (Model B) and consistency loss (Model C) aim
to improve.

Usage
-----
    from models.resnext50_baseline import ResNeXt50Baseline
    model = ResNeXt50Baseline(num_classes=5, pretrained=True)
"""

import torch
import torch.nn as nn
import torchvision.models as tv_models


class ResNeXt50Baseline(nn.Module):
    """
    ResNeXt-50 32x4d with a 5-class classification head.

    Forward flow:
        input [B, 3, H, W]
        → stem (conv1 + bn1 + relu + maxpool)  → [B, 64, H/4, W/4]
        → layer1 (3 × Bottleneck)              → [B, 256, H/4, W/4]
        → layer2 (4 × Bottleneck, stride=2)    → [B, 512, H/8, W/8]
        → layer3 (6 × Bottleneck, stride=2)    → [B, 1024, H/16, W/16]
        → layer4 (3 × Bottleneck, stride=2)    → [B, 2048, H/32, W/32]
        → AdaptiveAvgPool2d(1, 1)              → [B, 2048, 1, 1]
        → flatten                              → [B, 2048]
        → fc (Linear 2048 → num_classes)       → [B, num_classes]

    Parameters
    ----------
    num_classes : int
        Number of output classes (5 for beef grading).
    pretrained : bool
        Load ImageNet-pretrained ResNeXt-50 backbone weights.
    """

    def __init__(self, num_classes: int = 5, pretrained: bool = True):
        super().__init__()
        weights = (
            tv_models.ResNeXt50_32X4D_Weights.IMAGENET1K_V1
            if pretrained else None
        )
        backbone = tv_models.resnext50_32x4d(weights=weights)

        # ── Backbone (pretrained) ──────────────────────────────────────────
        self.layer0 = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
        )
        self.layer1 = backbone.layer1   # output: 256 ch
        self.layer2 = backbone.layer2   # output: 512 ch
        self.layer3 = backbone.layer3   # output: 1024 ch
        self.layer4 = backbone.layer4   # output: 2048 ch
        self.avgpool = backbone.avgpool

        # ── Classification head (new, random-init) ─────────────────────────
        self.fc = nn.Linear(2048, num_classes)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor [B, 3, H, W]

        Returns
        -------
        logits : Tensor [B, num_classes]
        """
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)
