# 문서 이미지 분류 대회 (Document Type Classification)

## 팀원

* 송병찬
* 유준우
* 이수지
* 이재석(팀장)
* 최유정

## 대회 개요

다양한 문서 이미지 데이터를 기반으로 문서 타입을 분류하는 Computer Vision 프로젝트입니다.
모델 학습부터 평가까지 전체 과정을 경험하며 실험 기반으로 성능을 개선하는 것을 목표로 합니다.

## 데이터 구성

### 학습 데이터

* 총 17개 클래스
* 총 1,570장

### 테스트 데이터

* 총 3,140장
* 다양한 노이즈 포함

## 평가 방법

* Metric: Macro F1 Score
* Confusion Matrix, Precision, Recall 활용

반드시 F1 Macro 기준으로 모델 평가

## 환경 설정 (Miniconda)

### 1. 환경 생성

```bash
conda create -n cv_comp python=3.11 -y
conda activate cv_comp
```

### 2. PyTorch 설치

환경에 맞게 하나 선택

```bash
# CPU
pip install torch torchvision

# GPU (CUDA 12.x)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 3. 의존성 설치

```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

### 4. Jupyter 커널 등록

```bash
python -m ipykernel install --user --name cv_comp --display-name "Python(cv_comp)"
```

## 설치 확인 (터미널)

```bash
python - <<'PY'
import torch
import torchvision
import albumentations
import yaml
import timm
import wandb

print("torch:", torch.__version__)
print("cuda:", torch.cuda.is_available())
print("environment setup ok")
PY
```

## 환경 변수 설정 (.env)

```bash
cp .env.example .env
```

`.env` 파일 수정:

```bash
WANDB_API_KEY=your_api_key
WANDB_ENTITY=your_entity
WANDB_PROJECT=document-type-classification
```

## 데이터 준비

```bash
tar -xvf data/data.tar.gz -C data/
```

```text
data/raw/
├── train/
├── test/
├── train.csv
└── sample_submission.csv
```

## 프로젝트 구조

```text
configs/
src/
experiments/
  template/
  {개인폴더}
```

## 실험 실행 방법

### 구조 개념

```text
configs/                → 공용 설정 (최종 제출 기준)
  base.yaml
  data.yaml
  train.yaml
  inference.yaml
  ensemble.yaml
  model/
    resnet50.yaml
    efficientnet_b0.yaml
    convnext_tiny.yaml
    deit_tiny.yaml

src/                    → 공용 코드 (모든 실행 기준)

experiments/{name}/     → 개인 실험 공간
  configs/              → 개인 실험용 config
  models/               → (선택) 모델 수정용
```

- 공용 config: 팀 합의 후 최종 제출에 사용
- 개인 config: 각자 실험 및 연구용
- 실행은 항상 `src` 기준
- 모델 정의는 반드시 `configs/model/` 사용

---

### 1. 실험 폴더 생성

```bash
cp -r experiments/template experiments/{your_name}
```

---

### 2. 개인 config 수정

```text
experiments/{your_name}/configs/
  data.yaml
  train.yaml
  inference.yaml
  model.yaml
```

---

## 개인 실험

### 3. 단일 모델 학습

```bash
python -m src.train \
  --base configs/base.yaml \
  --data experiments/{your_name}/configs/data.yaml \
  --train experiments/{your_name}/configs/train.yaml \
  --inference experiments/{your_name}/configs/inference.yaml \
  --model experiments/{your_name}/configs/model.yaml
```

---

### 4. 단일 모델 추론

```bash
python -m src.infer \
  --base configs/base.yaml \
  --data experiments/{your_name}/configs/data.yaml \
  --train experiments/{your_name}/configs/train.yaml \
  --inference experiments/{your_name}/configs/inference.yaml \
  --model experiments/{your_name}/configs/model.yaml
```

---

### 5. 검증용 앙상블 (KFold 평균)

```bash
python -m src.infer_valid_ensemble \
  --data experiments/{your_name}/configs/data.yaml \
  --train experiments/{your_name}/configs/train.yaml \
  --inference experiments/{your_name}/configs/inference.yaml \
  --model experiments/{your_name}/configs/model.yaml
```

---

### 6. [공용] 다중 모델 학습

```bash
python -m src.train_all_models --model_dir configs/model
```
---

### 7. [공용] 다중 모델 추론 (선택)
```bash
python -m src.infer_all_models --model_dir configs/model
```

---

### 8. [공용] 앙상블 (최종 제출 파일 생성)

```bash
python -m src.infer_ensemble \
  --data configs/data.yaml \
  --train configs/train.yaml \
  --inference configs/inference.yaml \
  --ensemble configs/ensemble.yaml
```

---

## 실행 흐름

```text
개인 실험:
train → infer → infer_valid_ensemble

공용 모델 학습:
train_all_models

(선택) 개별 모델 결과 확인:
infer_all_models

최종 제출:
infer_ensemble
```

---

## 모델 사용 규칙

```text
기본 원칙:
- 모델 정의는 configs/model/ 공용 폴더 사용
- 개인 폴더에 모델 복사 금지

예외:
- 모델 구조 변경 시만 복사 허용

주의:
- 제출 전 반드시 공용 모델 기준으로 재실행
```

---

## 핵심 규칙

```text
- src 코드는 공용이며 수정 금지 (merge 전 반드시 팀 논의)
- 실험은 개인 config에서만 수행
- 최종 제출은 공용 config 기준으로 실행
```

---

## 한 줄 요약

```text
개인 실험 → 성능 확인 → 팀 합의 → 공용 config로 재실행 → 제출
```