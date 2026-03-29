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

### 1. 실험 폴더 생성

```bash
cp -r experiments/template experiments/{your_name}
```

### 2. config 수정

```text
experiments/{your_name}/configs/
```

### 3. 학습

```bash
python -m src.train \
  --data experiments/{your_name}/configs/data.yaml \
  --train experiments/{your_name}/configs/train.yaml \
  --inference experiments/{your_name}/configs/inference.yaml \
  --model experiments/{your_name}/configs/model.yaml
```

### 4. 추론

```bash
python -m src.infer \
  --data experiments/{your_name}/configs/data.yaml \
  --train experiments/{your_name}/configs/train.yaml \
  --inference experiments/{your_name}/configs/inference.yaml \
  --model experiments/{your_name}/configs/model.yaml
```

## 결과 저장 위치

```text
experiments/{your_name}/outputs/
├── checkpoints/
├── logs/
├── metrics.csv
└── final_submission.csv
```

## 실험 흐름

1. Baseline 확인
2. Augmentation 실험
3. Preprocess 실험
4. 모델 변경
5. 최종 제출

## 실험 기록

```text
experiments/{your_name}/experiment_note.md
```

## 작업 규칙

* src 코드 수정 금지
* config 기반 실험
* 개인 폴더에서 실행
* 실험 결과 반드시 기록

## 폴더 역할

| 폴더 | 역할 |
| -------- | -------------- |
| template | 복사용 (직접 실행 금지) |
| 개인폴더 | 실험용 |

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

## 문제 해결

상세 내용:

```text
docs/setup.md
```

## 한 줄 요약

template 복사 --> config 수정 --> train --> infer --> 분석
