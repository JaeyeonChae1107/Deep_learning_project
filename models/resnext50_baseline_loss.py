"""
Model D: ResNeXt-50 + Perturbation Consistency Loss (no CBAM).

Architecture
------------
Identical to Model A (ResNeXt-50 baseline), with the KL consistency loss
added during training.  CBAM is intentionally absent, isolating the effect
of the consistency loss from the effect of CBAM.

Training loss
-------------
Same as Model C:
    L = CE(p, y) + CE(p', y) + lambda_kl * KL(p || p')

Role in ablation
----------------
Control for separating CBAM's contribution from the KL consistency loss:

    B - A = CBAM effect alone (no KL)
    D - A = KL loss effect alone (no CBAM)
    C - B = KL added on top of CBAM
    C - D = CBAM added on top of KL

Without Model D, it is impossible to tell whether Model C's reliability
improvement comes from CBAM or from the consistency training objective.

Usage
-----
    from models.resnext50_baseline_loss import ResNeXt50BaselineLoss
    model = ResNeXt50BaselineLoss(num_classes=5, pretrained=True)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models


class ResNeXt50BaselineLoss(nn.Module):
    """
    ResNeXt-50 32x4d trained with perturbation consistency loss (no CBAM).

    Architecture is identical to ResNeXt50Baseline (Model A).
    The only addition is the `consistency_loss` method for Model D training.

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

    def consistency_loss(
        self,
        x_orig:    torch.Tensor,
        x_pert:    torch.Tensor,
        targets:   torch.Tensor,
        ce_fn:     nn.CrossEntropyLoss,
        lambda_kl: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Perturbation consistency loss: L = CE(p, y) + CE(p', y) + lambda_kl * KL(p || p').

        Parameters
        ----------
        x_orig    : Tensor [B, 3, H, W]  — clean images
        x_pert    : Tensor [B, 3, H, W]  — perturbed images
        targets   : Tensor [B]
        ce_fn     : CrossEntropyLoss
        lambda_kl : float

        Returns
        -------
        total_loss   : scalar Tensor
        logits_clean : Tensor [B, num_classes]
        """
        logits_orig = self(x_orig)
        logits_pert = self(x_pert)

        loss_ce_orig = ce_fn(logits_orig, targets)
        loss_ce_pert = ce_fn(logits_pert, targets)

        log_p_pert = F.log_softmax(logits_pert, dim=1)
        p_orig     = F.softmax(logits_orig, dim=1)
        kl_loss    = F.kl_div(log_p_pert, p_orig, reduction="batchmean")

        total_loss = loss_ce_orig + loss_ce_pert + lambda_kl * kl_loss
        return total_loss, logits_orig
