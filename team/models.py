from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import ResNeXt50_32X4D_Weights, resnext50_32x4d


class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden = max(channels // reduction, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        return x * self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor, return_attention: bool = False) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attention = self.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))
        out = x * attention
        if return_attention:
            return out, attention
        return out


class CBAM(nn.Module):
    """Convolutional Block Attention Module: channel attention then spatial attention."""

    def __init__(self, channels: int, reduction: int = 16, spatial_kernel_size: int = 7) -> None:
        super().__init__()
        self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention(spatial_kernel_size)

    def forward(self, x: torch.Tensor, return_attention: bool = False) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        x = self.channel_attention(x)
        if return_attention:
            x, attention = self.spatial_attention(x, return_attention=True)
            return x, attention
        return self.spatial_attention(x)


class ResNeXtCBAM(nn.Module):
    """ResNeXt-50 with CBAM inserted after the final convolutional stage."""

    def __init__(
        self,
        num_classes: int,
        pretrained: bool = True,
        cbam_reduction: int = 16,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        weights = ResNeXt50_32X4D_Weights.IMAGENET1K_V2 if pretrained else None
        backbone = resnext50_32x4d(weights=weights)
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.cbam = CBAM(channels=2048, reduction=cbam_reduction)
        self.avgpool = backbone.avgpool
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc = nn.Linear(backbone.fc.in_features, num_classes)

    def forward_features(self, x: torch.Tensor, return_attention: bool = False) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return self.cbam(x, return_attention=return_attention)

    def forward(self, x: torch.Tensor, return_attention: bool = False) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if return_attention:
            x, attention = self.forward_features(x, return_attention=True)
        else:
            attention = None
            x = self.forward_features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        logits = self.fc(x)
        if return_attention:
            return logits, attention
        return logits

    def class_cam(
        self,
        features: torch.Tensor,
        target: torch.Tensor,
        output_size: tuple[int, int] | None = None,
    ) -> torch.Tensor:
        weights = self.fc.weight[target].unsqueeze(-1).unsqueeze(-1)
        cam = torch.sum(features * weights, dim=1, keepdim=True)
        cam = F.relu(cam)
        if output_size is not None:
            cam = F.interpolate(cam, size=output_size, mode="bilinear", align_corners=False)
        cam_sum = cam.sum(dim=(2, 3), keepdim=True).clamp_min(1e-8)
        return cam / cam_sum

    def forward_logits_and_cam(
        self,
        x: torch.Tensor,
        target: torch.Tensor,
        cam_size: tuple[int, int] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.forward_features(x)
        pooled = self.avgpool(features)
        pooled = torch.flatten(pooled, 1)
        pooled = self.dropout(pooled)
        logits = self.fc(pooled)
        cam = self.class_cam(features, target, output_size=cam_size)
        return logits, cam

    @property
    def target_layer(self) -> nn.Module:
        return self.layer4[-1]


class ResNeXtSimCLR(nn.Module):
    """ResNeXt-50 encoder with a SimCLR projection head."""

    def __init__(
        self,
        pretrained: bool = True,
        projection_dim: int = 128,
        hidden_dim: int = 2048,
    ) -> None:
        super().__init__()
        weights = ResNeXt50_32X4D_Weights.IMAGENET1K_V2 if pretrained else None
        self.encoder = resnext50_32x4d(weights=weights)
        feature_dim = self.encoder.fc.in_features
        self.encoder.fc = nn.Identity()
        self.projection_head = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, projection_dim),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encode(x)
        projections = self.projection_head(features)
        return F.normalize(projections, dim=1)


@dataclass(frozen=True)
class ModelSpec:
    name: str
    uses_cbam: bool
    uses_consistency_loss: bool
    uses_attention_consistency_loss: bool = False


MODEL_SPECS = {
    "resnext50": ModelSpec("resnext50", uses_cbam=False, uses_consistency_loss=False),
    "resnext50_cbam": ModelSpec("resnext50_cbam", uses_cbam=True, uses_consistency_loss=False),
    "resnext50_cbam_consistency": ModelSpec(
        "resnext50_cbam_consistency", uses_cbam=True, uses_consistency_loss=True
    ),
    "resnext50_cbam_attention_consistency": ModelSpec(
        "resnext50_cbam_attention_consistency",
        uses_cbam=True,
        uses_consistency_loss=True,
        uses_attention_consistency_loss=True,
    ),
}


def create_model(
    variant: str,
    num_classes: int = 5,
    pretrained: bool = True,
    dropout: float = 0.0,
) -> nn.Module:
    """Create one of the planned project models."""

    if variant not in MODEL_SPECS:
        valid = ", ".join(sorted(MODEL_SPECS))
        raise ValueError(f"Unknown model variant '{variant}'. Valid variants: {valid}")

    spec = MODEL_SPECS[variant]
    if spec.uses_cbam:
        return ResNeXtCBAM(num_classes=num_classes, pretrained=pretrained, dropout=dropout)

    weights = ResNeXt50_32X4D_Weights.IMAGENET1K_V2 if pretrained else None
    model = resnext50_32x4d(weights=weights)
    in_features = model.fc.in_features
    classifier: nn.Module
    if dropout > 0:
        classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, num_classes))
    else:
        classifier = nn.Linear(in_features, num_classes)
    model.fc = classifier
    model.target_layer = model.layer4[-1]  # type: ignore[attr-defined]
    return model


def create_simclr_model(
    pretrained: bool = True,
    projection_dim: int = 128,
    hidden_dim: int = 2048,
) -> ResNeXtSimCLR:
    return ResNeXtSimCLR(pretrained=pretrained, projection_dim=projection_dim, hidden_dim=hidden_dim)


def load_simclr_encoder_weights(
    model: nn.Module,
    checkpoint_path: str,
    map_location: torch.device | str = "cpu",
) -> nn.Module:
    """Load a SimCLR ResNeXt encoder into a supervised ResNeXt classifier."""

    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    encoder_state = checkpoint.get("encoder_state_dict", checkpoint.get("model_state_dict", checkpoint))
    target_state = model.state_dict()
    compatible_state = {}

    for key, value in encoder_state.items():
        if key.startswith("encoder."):
            key = key.removeprefix("encoder.")
        if key.startswith("fc.") or key.startswith("projection_head."):
            continue
        if key in target_state and target_state[key].shape == value.shape:
            compatible_state[key] = value

    missing, unexpected = model.load_state_dict(compatible_state, strict=False)
    if not compatible_state:
        raise ValueError(f"No compatible SimCLR encoder weights found in: {checkpoint_path}")
    print(
        "loaded_simclr_encoder:",
        {
            "checkpoint": checkpoint_path,
            "loaded_tensors": len(compatible_state),
            "missing_tensors": len(missing),
            "unexpected_tensors": len(unexpected),
        },
    )
    return model
