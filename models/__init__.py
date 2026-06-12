from .resnext50_baseline import ResNeXt50Baseline
from .resnext50_baseline_loss import ResNeXt50BaselineLoss
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
    Factory function for the four model variants.

    model_type  'baseline'       → Model A: ResNeXt-50 + CE loss
                'cbam'           → Model B: ResNeXt-50 + CBAM + CE loss
                'cbam_loss'      → Model C: ResNeXt-50 + CBAM + consistency loss
                'baseline_loss'  → Model D: ResNeXt-50 + consistency loss (no CBAM)

    Ablation matrix:
        B - A = CBAM effect (no KL)
        D - A = KL loss effect (no CBAM)
        C - B = KL added on top of CBAM
        C - D = CBAM added on top of KL
    """
    if model_type == "baseline":
        return ResNeXt50Baseline(num_classes, pretrained)
    elif model_type == "cbam":
        return ResNeXt50CBAM(num_classes, pretrained)
    elif model_type == "cbam_loss":
        return ResNeXt50CBAMLoss(num_classes, pretrained)
    elif model_type == "baseline_loss":
        return ResNeXt50BaselineLoss(num_classes, pretrained)
    raise ValueError(
        f"Unknown model_type '{model_type}'. "
        "Expected: 'baseline', 'cbam', 'cbam_loss', or 'baseline_loss'."
    )
