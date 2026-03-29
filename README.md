# 문서 이미지 분류 대회 (Document Type Classification)

## 팀원

* 송병찬
* 유준우
* 이수지
* 이재석(팀장)
* 최유정

## 대회 개요

이 대회는 다양한 문서 이미지 데이터를 바탕으로 문서 타입을 분류하는 컴퓨터 비전 경진대회입니다.
참가자는 모델 학습부터 평가에 이르는 전체 프로세스를 경험하며 모델 개선 과정을 학습하는 것을 목표로 합니다.

## 데이터 구성

### 학습 데이터

* 총 17개의 클래스
* 총 1,570장의 이미지 (클래스당 46~100장)

### 테스트 데이터

* 총 3,140장의 이미지
* 구겨짐, 빛번짐 등 다양한 노이즈 포함
* 난이도 조절을 위한 augmentation 적용

## 평가 방법

* **평가 지표**: Macro F1 Score
* **분석 도구**: Confusion Matrix, Precision, Recall

모델 평가는 반드시 **F1 Macro 기준으로 진행**

# 환경 설정 (Miniconda)

## 1. conda 환경 생성

```
conda create -n cv_comp python=3.11 -y
```

## 2. 환경 활성화

```bash
conda activate cv_comp
```

## 3. 의존성 설치

```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

# 환경 변수 설정 (.env)

본 프로젝트는 실험 기록을 위해 Weights & Biases를 사용합니다.

## 1. .env 파일 생성 (.env.example 복사)

```bash
cp .env.example .env
```

## 2. API Key 입력

`.env` 파일을 열어서 아래 값 수정

```bash
WANDB_API_KEY=your_api_key_here
WANDB_ENTITY=team_name
WANDB_PROJECT=document-type-classification
```

## 3. 주의사항

* `.env` 파일은 Git에 업로드 금지
* `.env.example`은 템플릿 역할
* 실제 API Key는 `.env`에만 작성

# 프로젝트 구조

```text
configs/
src/
experiments/
  template/
  {개인폴더}
```

# 실험 실행 방법

## 1. 실험 폴더 생성

```bash
cp -r experiments/template experiments/{your_name}
```

예:

```bash
cp -r experiments/template experiments/yoojw
```

## 2. config 수정

```text
experiments/{your_name}/configs/
```

수정 대상:

* data.yaml
* train.yaml
* inference.yaml
* model.yaml

## 3. 학습 (Train)

```bash
python -m src.train \
  --data experiments/{your_name}/configs/data.yaml \
  --train experiments/{your_name}/configs/train.yaml \
  --inference experiments/{your_name}/configs/inference.yaml \
  --model experiments/{your_name}/configs/model.yaml
```

## 4. 추론 (Inference)

```bash
python -m src.infer \
  --data experiments/{your_name}/configs/data.yaml \
  --train experiments/{your_name}/configs/train.yaml \
  --inference experiments/{your_name}/configs/inference.yaml \
  --model experiments/{your_name}/configs/model.yaml
```

# 결과 저장 위치

```text
experiments/{your_name}/outputs/
  ├── checkpoints/
  │   └── best.pt
  ├── logs/
  ├── metrics.csv
  └── final_submission.csv
```

# 실험 진행 순서

1. Baseline 확인
2. Augmentation 실험
3. Preprocess 실험
4. 모델 변경
5. 최종 submission 생성

# 모델 설정 예시

```yaml
model:
  library: timm
  name: efficientnet_b0
  pretrained: false
  num_classes: 17
```

# 실험 기록

각 실험은 아래 파일에 기록 해주시면 다른 팀원들이 연구하는데 도움이 됩니다.
(저희 노션이나 구글 드라이브로 기록하셔도 됩니다.)

```text
experiments/{your_name}/experiment_note.md
```

# 작업 규칙

* src 코드 수정 금지
* config 기반 실험 진행
* 각자 폴더에서 독립적으로 실험 수행
* 실험 결과 반드시 기록

# 핵심 원칙

* 실험은 반드시 비교 가능해야 함
* 결과보다 **왜 좋아졌는지**가 중요
* config 기반으로 재현 가능해야 함

# 폴더 역할

| 폴더       | 역할             |
| -------- | -------------- |
| template | 복사용 기준 (수정 금지) |
| {개인폴더}   | 실제 실험 수행       |

# 한 줄 요약

template 복사 --> config 수정 --> train --> infer --> 결과 분석
