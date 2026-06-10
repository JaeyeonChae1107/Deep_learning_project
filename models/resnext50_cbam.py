"""
Model B: ResNeXt-50 + CBAM for Beef Grade Classification.

Architecture
------------
Identical to Model A (ResNeXt-50 backbone + 5-class FC), with a CBAM module
inserted **after each of the four residual stages** (layer1–4).

                            CBAM placement
    ┌──────────┐    ┌─────────────┐    ┌──────────┐    ┌─────────────┐
    │  layer1  │───▶│   CBAM(256) │───▶│  layer2  │───▶│   CBAM(512) │──▶ ...
    └──────────┘    └─────────────┘    └──────────┘    └─────────────┘

Each CBAM(C) applies:
  1. ChannelGate: squeeze (AvgPool + MaxPool) → shared MLP(C→C/16→C) → Sigmoid
     → scales each channel independently
  2. SpatialGate: channel-wise max+avg concat → 7×7 Conv → Sigmoid
     → scales each spatial location independently

All pretrained ResNeXt-50 weights are preserved; only CBAM and the new FC layer
introduce new parameters.

Training loss
-------------
Standard CrossEntropyLoss (same as Model A).

Role in ablation
----------------
Tests whether CBAM's selective attention improves prediction stability
under perturbation compared to the bare ResNeXt-50 (Model A).

Usage
-----
    from models.resnext50_cbam import ResNeXt50CBAM
    model = ResNeXt50CBAM(num_classes=5, pretrained=True)
"""

import torch
import torch.nn as nn
import torchvision.models as tv_models

from models.attention_module.cbam import CBAM


class ResNeXt50CBAM(nn.Module):
    """
    ResNeXt-50 32x4d with CBAM inserted after each residual stage.

    Channel configuration per CBAM:
        After layer1 → CBAM(gate_channels=256,  reduction_ratio=16)
        After layer2 → CBAM(gate_channels=512,  reduction_ratio=16)
        After layer3 → CBAM(gate_channels=1024, reduction_ratio=16)
        After layer4 → CBAM(gate_channels=2048, reduction_ratio=16)

    Forward flow:
        input [B, 3, H, W]
        → stem                              → [B, 64, H/4, W/4]
        → layer1 → CBAM(256)               → [B, 256, H/4, W/4]
        → layer2 → CBAM(512)               → [B, 512, H/8, W/8]
        → layer3 → CBAM(1024)              → [B, 1024, H/16, W/16]
        → layer4 → CBAM(2048)              → [B, 2048, H/32, W/32]
        → AdaptiveAvgPool2d → flatten → fc → [B, num_classes]

    Parameters
    ----------
    num_classes : int
        Number of output classes.
    pretrained : bool
        Load ImageNet-pretrained backbone weights.
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
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.avgpool = backbone.avgpool

        # ── CBAM modules (new, random-init) ───────────────────────────────
        self.cbam1 = CBAM(gate_channels=256,  reduction_ratio=16)
        self.cbam2 = CBAM(gate_channels=512,  reduction_ratio=16)
        self.cbam3 = CBAM(gate_channels=1024, reduction_ratio=16)
        self.cbam4 = CBAM(gate_channels=2048, reduction_ratio=16)

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
        x = self.cbam1(x)   # channel + spatial attention on 256-ch feature map

        x = self.layer2(x)
        x = self.cbam2(x)   # channel + spatial attention on 512-ch feature map

        x = self.layer3(x)
        x = self.cbam3(x)   # channel + spatial attention on 1024-ch feature map

        x = self.layer4(x)
        x = self.cbam4(x)   # channel + spatial attention on 2048-ch feature map

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)
