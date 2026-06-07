# CBAM 기반 고기 등급 평가 프로젝트

**EL3005 Final Project** | "From accurate prediction to reliable decision-making."

## 연구 질문

> Can CBAM improve prediction and attention stability in CNN-based beef grading?

기존 EfficientViT 기반 연구(Lim & Song, 2025)는 높은 정확도와 attention 시각화를 제시했지만,
입력 변화에 대한 예측·attention의 **안정성(reliability)** 은 검증하지 않았다.
본 프로젝트는 수업에서 배운 CBAM을 ResNet50에 적용하고, perturbation 기반 consistency 분석으로
모델 신뢰성을 정량 평가한다.

## 레포 구조 (사전 작업)

```
.
├── models/
│   └── attention_module/        # 공식 CBAM 코드 (Jongchan/attention-module, MIT)
│       ├── cbam.py              # CBAM 원본 코드
│       └── model_resnet.py      # ResNet + CBAM 원본 코드
├── reference/
│   └── efficientvit/            # EfficientViT 참조 코드 (mit-han-lab/efficientvit, MIT)
│       ├── backbone.py          # EfficientViT backbone 원본
│       ├── cls.py               # EfficientViT classification 원본
│       └── README.md            # 기존 연구(Lim & Song 2025) 정리
├── configs/
│   └── config.yaml              # 학습·평가·perturbation 설정값
└── requirements.txt             # 의존성
```

## 수집한 공식 공개 코드

| 모델 | 출처 | 위치 |
|------|------|------|
| CBAM (Woo et al., ECCV 2018) | github.com/Jongchan/attention-module | `models/attention_module/` |
| EfficientViT (Liu et al., CVPR 2023) | github.com/mit-han-lab/efficientvit | `reference/efficientvit/` |
| ResNet-50 (He et al., CVPR 2016) | torchvision 내장 | `torchvision.models.resnet50` |

## 참고 문헌

- Lim & Song (2025). Beef Carcass Grading with EfficientViT. *Applied Sciences*.
- Woo et al. (2018). CBAM: Convolutional Block Attention Module. *ECCV 2018*.
- Liu et al. (2023). EfficientViT: Multi-Scale Linear Attention. *CVPR 2023*.
- He et al. (2016). Deep Residual Learning for Image Recognition. *CVPR 2016*.
