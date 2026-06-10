from .resnext50_baseline import ResNeXt50Baseline
from .resnext50_cbam import ResNeXt50CBAM
from .resnext50_cbam_loss import ResNeXt50CBAMLoss, compute_loss_model_c
from .attention_module.cbam import CBAM, ChannelGate, SpatialGate
import torch.nn as nn


def build_model(
    model_type: str,
    num_classes: int = 5,
    pretrained: bool = True,
) -> nn.Module:
    """
    Factory function for the three model variants.

    model_type  'baseline'   → Model A: ResNeXt-50 + CE loss
                'cbam'       → Model B: ResNeXt-50 + CBAM + CE loss
                'cbam_loss'  → Model C: ResNeXt-50 + CBAM + consistency loss
    """
    if model_type == "baseline":
        return ResNeXt50Baseline(num_classes, pretrained)
    elif model_type == "cbam":
        return ResNeXt50CBAM(num_classes, pretrained)
    elif model_type == "cbam_loss":
        return ResNeXt50CBAMLoss(num_classes, pretrained)
    raise ValueError(
        f"Unknown model_type '{model_type}'. "
        "Expected: 'baseline', 'cbam', or 'cbam_loss'."
    )
