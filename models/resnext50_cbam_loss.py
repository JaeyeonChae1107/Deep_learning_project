"""
Model C: ResNeXt-50 + CBAM + Perturbation Consistency Loss.

Architecture
------------
Identical to Model B (ResNeXt-50 + CBAM inside each Bottleneck block, Woo et al. 2018).
The only difference is the training objective: in addition to the standard
cross-entropy loss on clean images, the model is trained with a KL-divergence
term that pushes the prediction distribution on perturbed images toward that
on the clean image.

Training loss
-------------
For each mini-batch:
    x  = original image
    x' = randomly perturbed version of x   (one perturbation type, random strength)
    p  = softmax(model(x))                 (clean prediction)
    p' = softmax(model(x'))                (perturbed prediction)

    L_total = CE(p,  y)          ← clean classification
            + CE(p', y)          ← perturbed classification
            + λ · KL(p ‖ p')     ← consistency regularisation

    where CE = CrossEntropyLoss, KL = KL divergence, λ = lambda_kl (default 1.0).

The KL term is asymmetric: p is the reference distribution, and p' is penalised
for deviating from p. This directly incentivises the model to produce stable
probability outputs under photometric and geometric changes.

Role in ablation
----------------
Tests whether adding the consistency regularisation objective on top of CBAM
further improves prediction stability (lower KL/JSD, higher Top-1 agreement)
compared to Model B.

Usage
-----
    from models.resnext50_cbam_loss import ResNeXt50CBAMLoss

    model = ResNeXt50CBAMLoss(num_classes=5, pretrained=True)

    # Inside training loop:
    loss, logits_clean = model.consistency_loss(x, x_prime, y, ce_fn, lambda_kl=1.0)
    loss.backward()
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models

from models.resnext50_cbam import CBAMBottleneckWrapper, _inject_cbam  # noqa: F401


class ResNeXt50CBAMLoss(nn.Module):
    """
    ResNeXt-50 + CBAM, designed for training with perturbation consistency loss.

    The architecture is identical to ResNeXt50CBAM (Model B) — CBAM is injected
    inside each Bottleneck block before the residual add (Woo et al. 2018).
    The only addition is the `consistency_loss` helper method.

    Channel / shape flow: same as Model B — see resnext50_cbam.py.

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

    # ── forward ───────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)

    # ── Model C training objective ─────────────────────────────────────────

    def consistency_loss(
        self,
        x_orig:    torch.Tensor,
        x_pert:    torch.Tensor,
        targets:   torch.Tensor,
        ce_fn:     nn.CrossEntropyLoss,
        lambda_kl: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the perturbation consistency loss for Model C.

            L = CE(p, y) + CE(p', y) + λ · KL(p ‖ p')

        KL direction: p is the reference (clean); p' should match p.
        Using F.kl_div(log_q, p) = sum_i p_i · log(p_i / q_i) = KL(p ‖ q).

        Parameters
        ----------
        x_orig    : Tensor [B, 3, H, W]  — clean images (normalised)
        x_pert    : Tensor [B, 3, H, W]  — perturbed images (same normalisation)
        targets   : Tensor [B]           — ground-truth class indices
        ce_fn     : CrossEntropyLoss     — shared loss function (may carry class weights)
        lambda_kl : float                — weight of the KL consistency term

        Returns
        -------
        total_loss    : scalar Tensor
        logits_clean  : Tensor [B, num_classes]  — for accuracy logging
        """
        logits_orig = self(x_orig)
        logits_pert = self(x_pert)

        # Cross-entropy on both clean and perturbed
        loss_ce_orig = ce_fn(logits_orig, targets)
        loss_ce_pert = ce_fn(logits_pert, targets)

        # KL(p ‖ p') — F.kl_div expects log-probabilities as first argument
        # KL(p ‖ q) = sum p * log(p / q) = sum p * (log p - log q)
        #           = F.kl_div(log_q, p, reduction='batchmean')
        log_p_pert = F.log_softmax(logits_pert, dim=1)   # log q
        p_orig     = F.softmax(logits_orig, dim=1)        # p (reference)
        kl_loss    = F.kl_div(log_p_pert, p_orig, reduction="batchmean")

        total_loss = loss_ce_orig + loss_ce_pert + lambda_kl * kl_loss
        return total_loss, logits_orig
