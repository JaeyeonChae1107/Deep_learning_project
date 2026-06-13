import torch
import torch.nn as nn
import torchvision.models as tv_models


class ResNeXt50Baseline(nn.Module):


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
