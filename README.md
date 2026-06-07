# CBAM 기반 고기 등급 평가 프로젝트

**EL3005 Final Project** | "From accurate prediction to reliable decision-making."

## 연구 질문

> Can CBAM improve prediction and attention stability in CNN-based beef grading?

기존 EfficientViT 기반 연구(Lim & Song, 2025)는 높은 정확도와 attention 시각화를 제시했지만,
입력 변화에 대한 예측·attention의 **안정성(reliability)** 은 검증하지 않았다.
본 프로젝트는 수업에서 배운 CBAM을 ResNet50에 적용하고, perturbation 기반 consistency 분석으로
모델 신뢰성을 정량 평가한다.

## 프로젝트 구조

```
.
├── configs/config.yaml          # 학습·평가 설정
├── data/
│   └── dataset.py               # AI-Hub 데이터셋 로더
├── models/
│   ├── cbam.py                  # CBAM (Woo et al., ECCV 2018)
│   ├── resnet50_baseline.py     # ResNet50 (Baseline)
│   └── resnet50_cbam.py         # ResNet50 + CBAM (Proposed)
├── utils/
│   ├── gradcam.py               # Grad-CAM (Selvaraju et al., ICCV 2017)
│   ├── perturbation.py          # 5가지 perturbation 함수
│   └── metrics.py               # Prediction / Attention consistency metrics
├── experiments/
│   └── verify_setup.py          # 환경 검증 스크립트
├── reference/
│   └── efficientvit/README.md   # 기존 연구 (Lim & Song 2025) 요약
└── requirements.txt
```

## 모델 비교

| 구분      | 모델             | 역할                             |
|-----------|------------------|----------------------------------|
| Baseline  | ResNet50         | 기본 CNN 고기 등급 분류           |
| Proposed  | ResNet50 + CBAM  | Channel + Spatial Attention 강화 |
| Reference | EfficientViT     | 기존 SOTA (분류 성능 참조만)      |

## Task 구조

**Task 1 — 분류 성능**: Accuracy, F1-score, Confusion Matrix (두 모델 간 비교)

**Task 2 — 신뢰성 분석**: Perturbation 전후 consistency 정량 비교
- Prediction Consistency: 예측 일치율 + KL divergence
- Attention Consistency: SSIM(H, H') + Top-20% pixel IoU
- Attention Entropy: activation map의 집중도

### Perturbation 유형

| Perturbation   | 범위               | 대응 현장 변화        |
|----------------|--------------------|-----------------------|
| Brightness     | ±20%               | 조명 변화             |
| Contrast       | factor 0.8–1.2     | 카메라 설정 차이      |
| Gaussian Noise | σ = 0.05           | 센서 노이즈           |
| Random Crop    | 90% 영역 유지      | 촬영 위치 변화        |
| Rotation       | ±10°               | 절단면 각도 차이      |

## 데이터셋

AI-Hub 한우 도체 이미지 데이터셋 (77,899장, 5등급: 1++, 1+, 1, 2, 3)

다운로드: https://aihub.or.kr  
디렉토리 구조: `dataset/{1++,1+,1,2,3}/*.jpg`

## 환경 설정

```bash
pip install -r requirements.txt
python experiments/verify_setup.py   # 환경 검증
```

## 참고 문헌

- Lim & Song (2025). Beef Carcass Grading with EfficientViT. *Applied Sciences*.
- Woo et al. (2018). CBAM: Convolutional Block Attention Module. *ECCV 2018*.
- Selvaraju et al. (2017). Grad-CAM. *ICCV 2017*.
- He et al. (2016). Deep Residual Learning for Image Recognition. *CVPR 2016*.
