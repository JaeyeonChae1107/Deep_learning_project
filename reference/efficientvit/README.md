# EfficientViT Reference — Lim & Song (2025)

## Paper

**Beef Carcass Grading with EfficientViT**
Lim, J. & Song, H. Applied Sciences, 2025.

## Role in This Project

EfficientViT is used **as a reference baseline only**, not reimplemented.
The project does NOT aim to beat EfficientViT in classification accuracy.
Reference numbers for Task 1 context:

| Metric           | EfficientViT (5-class) | EfficientViT (1++ binary) |
|------------------|------------------------|---------------------------|
| Accuracy         | 98.46%                 | 99.24%                    |
| F1-score         | 0.9867                 | 0.9866                    |
| Inference Latency| 3.92 ms                | —                         |

**Note:** The scatter plot (Figure 5) in the original paper shows 1++ binary accuracy
(98.95–99.25%), which differs from the 5-class accuracy (98.46%). Do not conflate these.

## Key Architectural Details

- **Model**: EfficientViT-B (Vision Transformer with efficient attention)
- **Attention type**: ReLU linear attention (NOT softmax self-attention)
  - Structural consequence: cannot form sharp, concentrated attention distributions
  - "Broad attention" is partly a property of the mechanism, not only model understanding
- **Training data**: AI-Hub beef carcass dataset (77,899 RGB images, 5 classes)
- **Visualization**: attention map (NOT Grad-CAM) — asymmetric comparison vs CNN baselines

## Acknowledged Limitations (from Section 5, Discussion)

> "This study is based on a curated dataset from AI Hub, and further validation is needed
> to assess generalizability under more variable industrial conditions such as differing
> lighting, background clutter, or carcass positioning."

This exact gap motivates Task 2 (perturbation-based reliability analysis) of this project.

## Official EfficientViT Code

Original EfficientViT architecture (Microsoft Research):
- Repository: https://github.com/microsoft/efficientvit
- Paper: Liu et al., CVPR 2023 — https://arxiv.org/abs/2205.14756

The beef-grading specific model from Lim & Song 2025 is not publicly released.
