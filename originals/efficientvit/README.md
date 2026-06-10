# EfficientViT Reference

## 두 가지 EfficientViT

이 프로젝트에서 "EfficientViT"는 두 개의 서로 다른 논문/구현을 가리킨다.

| 구분 | 논문 | 저자 | 코드 |
|------|------|------|------|
| **원 논문** (아키텍처) | EfficientViT: Multi-Scale Linear Attention for High-Resolution Dense Prediction, CVPR 2023 | Liu et al. (MIT Han Lab) | https://github.com/mit-han-lab/efficientvit |
| **기존 연구** (참조 대상) | Beef Carcass Grading with EfficientViT, Appl. Sci. 2025 | Lim & Song | 미공개 |

Lim & Song (2025)는 위 MIT Han Lab의 EfficientViT를 beef grading에 적용한 논문이다.
본 프로젝트는 Lim & Song (2025)를 기존 연구로 참조하고, 이 논문의 한계를 출발점으로 삼는다.

---

## MIT Han Lab EfficientViT 공식 코드

### 수집 파일 (이 디렉토리)

| 파일 | 출처 | 설명 |
|------|------|------|
| `backbone.py` | mit-han-lab/efficientvit | EfficientViT backbone (B/L 시리즈) |
| `cls.py` | mit-han-lab/efficientvit | Classification head + 모델 factory |

### 주요 특징 (아키텍처 이해용)

- **Attention 메커니즘**: ReLU linear attention (softmax self-attention이 아님)
  - Softmax 없이 `ReLU(Q) · (ReLU(K)ᵀ · V)` 형태로 연산
  - 구조적으로 sharp한 집중 분포를 만들지 못함 → "넓은 attention"의 원인
  - 이것이 본 프로젝트에서 "넓음 ≠ 안정성"을 주장하는 근거
- **구조**: Local CNN blocks + EfficientViT blocks (후반 stage에만 attention 적용)
- **모델 시리즈**: B0~B3 (경량), L0~L3 (대형)

### 실행 환경 설정

`backbone.py`, `cls.py`는 내부 모듈(`efficientvit.models.nn`, `efficientvit.models.utils`)에 의존하므로,
실행하려면 전체 패키지 설치가 필요하다.

```bash
pip install efficientvit
# 또는
git clone https://github.com/mit-han-lab/efficientvit
```

---

## Lim & Song (2025) — 기존 연구 핵심 수치

| Metric | EfficientViT (5-class) | EfficientViT (1++ binary) |
|--------|------------------------|---------------------------|
| Accuracy | 98.46% | 99.24% |
| F1-score | 0.9867 | 0.9866 |
| Inference Latency | 3.92 ms | — |

**주의**: scatter plot (Figure 5)은 1++ 이진 분류 accuracy (98.95~99.25%)를 보여주며,
전체 5등급 accuracy (98.46%)와 다르다. 발표 시 혼용 금지.

### 저자가 직접 인정한 한계 (Section 5, Discussion)

> "This study is based on a curated dataset from AI Hub, and further validation is needed
> to assess generalizability under more variable industrial conditions such as
> **differing lighting, background clutter, or carcass positioning.**"
>
> — Lim & Song (2025)

이 한계가 본 프로젝트의 perturbation 실험이 직접 답하는 질문이다.

### 비교 시각화 방법론의 비대칭 문제

기존 논문은 CNN 모델에 Grad-CAM, EfficientViT에 attention map을 사용 → 직접 비교 불공정.
본 프로젝트는 ResNet50 / ResNet50+CBAM 양쪽에 Grad-CAM을 균일 적용하여 이 문제를 해결한다.

---

## ResNet-50

ResNet-50은 torchvision에 내장되어 있어 별도 코드 수집 불필요.

```python
import torchvision.models as models
resnet50 = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
```

원 논문: He et al., "Deep Residual Learning for Image Recognition", CVPR 2016.
공식 구현: https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py
