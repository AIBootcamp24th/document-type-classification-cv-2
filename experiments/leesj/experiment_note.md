# Experiment Note (예시)

* 본 기록지는 팀원 간 공유하고 최종적으로 어떤 부분을 수정해야 성능을 올릴 수 있는지 확인할 수 있도록 만들었습니다.

## 1. 실험 개요

* 실험 목적:
* 실험 일자:
* 실험자:

## 2. 변경 사항 (Baseline 대비)

### Preprocess

* (예: grayscale 적용 여부)

### Augmentation

* (예: brightness 추가, affine 제거 등)

### Model

* (예: resnet50 → efficientnet_b0)

### Training 설정

* (예: lr 변경, batch size 변경)

## 3. 사용 Config

```text
data.yaml:
train.yaml:
inference.yaml:
model.yaml:
```

## 4. 결과

### 주요 metric

| metric   | train | valid |
| -------- | ----- | ----- |
| accuracy |       |       |
| f1_macro |       |       |
| loss     |       |       |

### Best Epoch

* epoch:
* valid_f1_macro:

## 5. 결과 해석

* 모델이 잘 학습되었는가:
* overfitting 여부:
* f1_macro 기준 성능:

## 6. 비교 (Baseline 대비)

* 좋아진 점:
* 나빠진 점:
* 차이 원인 추정:

## 7. 결론

* 유지할 것:
* 버릴 것:
* 다음 실험 방향:

## 8. 메모

* 자유롭게 기록
