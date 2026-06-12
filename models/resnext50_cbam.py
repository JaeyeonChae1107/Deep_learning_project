"""
Model B: ResNeXt-50 + CBAM for Beef Grade Classification.

Architecture
------------
Identical to Model A (ResNeXt-50 backbone + 5-class FC), with CBAM injected
**inside each Bottleneck block** before the residual addition, matching the
original placement in Woo et al. (ECCV 2018).

                    CBAM placement (inside every block)
    shortcut x ──────────────────────────────────────────────────┐
    main path → conv1 → bn1 → relu → conv2 → bn2 → relu          │
              → conv3 → bn3 → CBAM → (+) → relu                  │
                                     ↑                            │
                              residual add  ◄──────────────────────┘

Each CBAM(C) applies:
  1. ChannelGate: squeeze (AvgPool + MaxPool) → shared MLP(C→C/16→C) → Sigmoid
     → scales each channel independently
  2. SpatialGate: channel-wise max+avg concat → 7×7 Conv → Sigmoid
     → scales each spatial location independently

All pretrained ResNeXt-50 weights are preserved; only CBAM and the new FC layer
introduce new parameters.  ResNeXt-50 has [3, 4, 6, 3] blocks → 16 CBAM modules.

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
from torchvision.models.resnet import Bottleneck as TorchBottleneck

from models.attention_module.cbam import CBAM


class CBAMBottleneckWrapper(nn.Module):
    """
    Wraps a pretrained torchvision Bottleneck and injects CBAM before the
    residual add, matching Woo et al. (ECCV 2018).

    Pretrained parameters (conv1/2/3, bn1/2/3, downsample) are shared
    references from the original block — no weight copying is needed.
    """

    def __init__(self, block: TorchBottleneck):
        super().__init__()
        self.conv1      = block.conv1
        self.bn1        = block.bn1
        self.conv2      = block.conv2
        self.bn2        = block.bn2
        self.conv3      = block.conv3
        self.bn3        = block.bn3
        self.relu       = block.relu
        self.downsample = block.downsample
        self.stride     = block.stride
        self.cbam = CBAM(gate_channels=block.bn3.num_features, reduction_ratio=16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out = self.cbam(out)          # ← CBAM before residual add (Woo et al. 2018)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


def _inject_cbam(layer: nn.Sequential) -> nn.Sequential:
    """Replace every Bottleneck in a residual stage with CBAMBottleneckWrapper."""
    return nn.Sequential(*[CBAMBottleneckWrapper(block) for block in layer])


class ResNeXt50CBAM(nn.Module):
    """
    ResNeXt-50 32x4d with CBAM injected inside every Bottleneck block.

    ResNeXt-50 has [3, 4, 6, 3] blocks → 16 CBAM modules in total:
        layer1: 3 × CBAM(256)
        layer2: 4 × CBAM(512)
        layer3: 6 × CBAM(1024)
        layer4: 3 × CBAM(2048)

    Forward flow:
        input [B, 3, H, W]
        → stem                                      → [B, 64, H/4, W/4]
        → layer1 (3 × Bottleneck+CBAM(256))         → [B, 256, H/4, W/4]
        → layer2 (4 × Bottleneck+CBAM(512))         → [B, 512, H/8, W/8]
        → layer3 (6 × Bottleneck+CBAM(1024))        → [B, 1024, H/16, W/16]
        → layer4 (3 × Bottleneck+CBAM(2048))        → [B, 2048, H/32, W/32]
        → AdaptiveAvgPool2d → flatten → fc          → [B, num_classes]

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

        # ── Backbone stem (pretrained) ─────────────────────────────────────
        self.layer0 = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
        )

        # ── Residual stages: CBAM injected inside every block ─────────────
        self.layer1 = _inject_cbam(backbone.layer1)
        self.layer2 = _inject_cbam(backbone.layer2)
        self.layer3 = _inject_cbam(backbone.layer3)
        self.layer4 = _inject_cbam(backbone.layer4)
        self.avgpool = backbone.avgpool

        # ── Classification head (new, random-init) ─────────────────────────
        self.fc = nn.Linear(2048, num_classes)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)
