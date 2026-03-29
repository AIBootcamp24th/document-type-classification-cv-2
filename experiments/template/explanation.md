# CV 프로젝트 실행 가이드 (Template 기반)

본 설명은 `experiments/template` 폴더를 복사하여 각자 실험을 진행할 수 있도록 구조를 만들었습니다.

## 1. 실험 폴더 생성

```bash
cp -r experiments/template experiments/{본인폴더명}
```

예시:

```bash
cp -r experiments/template experiments/yoojw
```

## 2. 설정 파일 위치

각자 사용하는 config는 아래 경로에 있습니다.

```text
experiments/{본인폴더명}/configs/
  ├── data.yaml
  ├── train.yaml
  ├── inference.yaml
  └── model.yaml
```

## 3. 학습 (Train)

```bash
python -m src.train \
  --data experiments/{본인폴더명}/configs/data.yaml \
  --train experiments/{본인폴더명}/configs/train.yaml \
  --inference experiments/{본인폴더명}/configs/inference.yaml \
  --model experiments/{본인폴더명}/configs/model.yaml
```

## 4. 추론 (Inference)

```bash
python -m src.infer \
  --data experiments/{본인폴더명}/configs/data.yaml \
  --train experiments/{본인폴더명}/configs/train.yaml \
  --inference experiments/{본인폴더명}/configs/inference.yaml \
  --model experiments/{본인폴더명}/configs/model.yaml
```

## 5. 결과 저장 위치

실행 결과는 자동으로 아래 경로에 저장됩니다.

```text
experiments/{본인폴더명}/outputs/
  ├── checkpoints/
  │   └── best.pt
  ├── logs/
  ├── metrics.csv
  └── final_submission.csv
```

## 6. 작업 규칙

* 각자 폴더에서 독립적으로 실험 진행
* 실험 결과는 `outputs` 폴더 기준으로 확인
* 루트 베이스라인 코드 수정 X (src 폴더 건드리지 않기)
* 팀원들 간 충분한 상의 후 config 파일만 수정 O

## 7. 권장 실험 순서

1. baseline 실행 (정상 동작 확인)
2. augmentation 변경
3. preprocess 적용 여부 판단
4. 모델 변경 (resnet / efficientnet 등)
5. 최종 submission 생성

## 한 줄 요약

template 복사 --> config 수정 --> train --> infer --> 결과 확인

## Template 폴더 실행 방법 (터미널)

template 폴더로 학습 및 추론까지 완료 후 최종 final_submission.csv 파일까지 저장 하는 방법

```bash
python -m src.train \
  --data experiments/template/configs/data.yaml \
  --train experiments/template/configs/train.yaml \
  --inference experiments/template/configs/inference.yaml \
  --model experiments/template/configs/model.yaml
```

```bash
python -m src.infer \
  --data experiments/template/configs/data.yaml \
  --train experiments/template/configs/train.yaml \
  --inference experiments/template/configs/inference.yaml \
  --model experiments/template/configs/model.yaml
```

실행 완료되면 `outputs` 폴더에 `final_submission.csv`로 저장 완료
