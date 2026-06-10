"""
Official ResNeXt-50 (32x4d) implementation from torchvision.
Source: https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py
License: BSD 3-Clause

ResNeXt-50 32x4d is ResNet-50 with grouped convolutions:
  groups=32, width_per_group=4  (hence "32x4d")

Key difference from ResNet-50:
  In each Bottleneck, the 3×3 conv uses groups=32 instead of groups=1.
  The effective bottleneck width becomes:
    width = int(planes * (base_width / 64.0)) * groups
           = int(planes * (4 / 64.0)) * 32
           = planes * 2   (for planes=64 → 128, planes=128 → 256, ...)

Channel sizes at each stage output:
  stem (conv1+bn1+relu+maxpool) : [B, 64,  H/4,  W/4]
  layer1 (3 Bottlenecks)        : [B, 256, H/4,  W/4]
  layer2 (4 Bottlenecks)        : [B, 512, H/8,  W/8]
  layer3 (6 Bottlenecks)        : [B, 1024,H/16, W/16]
  layer4 (3 Bottlenecks)        : [B, 2048,H/32, W/32]
  avgpool + fc                  : [B, num_classes]

Usage (torchvision API):
  from torchvision.models import resnext50_32x4d, ResNeXt50_32X4D_Weights
  model = resnext50_32x4d(weights=ResNeXt50_32X4D_Weights.IMAGENET1K_V1)
"""

from typing import Any, Callable, List, Optional, Type, Union

import torch
import torch.nn as nn
from torch import Tensor


# ── building blocks ───────────────────────────────────────────────────────────

def conv3x3(
    in_planes: int,
    out_planes: int,
    stride: int = 1,
    groups: int = 1,
    dilation: int = 1,
) -> nn.Conv2d:
    """3×3 convolution with padding."""
    return nn.Conv2d(
        in_planes, out_planes,
        kernel_size=3, stride=stride,
        padding=dilation, groups=groups,
        bias=False, dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1×1 convolution."""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    """Two-conv residual block (used in ResNet-18/34, NOT in ResNeXt-50)."""
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)


class Bottleneck(nn.Module):
    """
    Three-conv bottleneck block (used in ResNet-50/101 and ResNeXt-50/101).

    For ResNeXt-50 32x4d:
      groups=32, base_width=4
      → width = int(planes * (4/64)) * 32 = planes // 2 * ... see formula

    The grouped 3×3 conv at the centre is what makes it ResNeXt.
    """
    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # Effective bottleneck width (wider than ResNet for the same planes)
        width = int(planes * (base_width / 64.0)) * groups

        self.conv1 = conv1x1(inplanes, width)
        self.bn1   = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)   # ← grouped conv
        self.bn2   = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3   = norm_layer(planes * self.expansion)
        self.relu  = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        return self.relu(out)


# ── ResNet / ResNeXt backbone ─────────────────────────────────────────────────

class ResNet(nn.Module):
    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.inplanes   = 64
        self.dilation   = 1
        self.groups     = groups
        self.base_width = width_per_group

        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]

        # Stem
        self.conv1   = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1     = norm_layer(self.inplanes)
        self.relu    = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Residual stages
        self.layer1 = self._make_layer(block, 64,  layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc      = nn.Linear(512 * block.expansion, num_classes)

        # Weight initialisation
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer        = self._norm_layer
        downsample        = None
        previous_dilation = self.dilation

        if dilate:
            self.dilation *= stride
            stride = 1

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = [
            block(
                self.inplanes, planes, stride, downsample,
                self.groups, self.base_width, previous_dilation, norm_layer,
            )
        ]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes, planes,
                    groups=self.groups, base_width=self.base_width,
                    dilation=self.dilation, norm_layer=norm_layer,
                )
            )
        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


# ── factory ───────────────────────────────────────────────────────────────────

def resnext50_32x4d(
    num_classes: int = 1000,
    pretrained: bool = False,
    **kwargs: Any,
) -> ResNet:
    """
    ResNeXt-50 32x4d.

    Architecture: Bottleneck × [3, 4, 6, 3] with groups=32, width_per_group=4.
    For pretrained weights use torchvision:
        from torchvision.models import resnext50_32x4d, ResNeXt50_32X4D_Weights
        model = resnext50_32x4d(weights=ResNeXt50_32X4D_Weights.IMAGENET1K_V1)
    """
    kwargs.setdefault("groups", 32)
    kwargs.setdefault("width_per_group", 4)
    model = ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, **kwargs)
    return model
