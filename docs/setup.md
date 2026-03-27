# Environment Setup Guide (팀 공통 기준)

## 1. 가상환경 생성 (conda)

conda는 환경 생성 용도로만 사용합니다.

    conda create -n cv_comp python=3.11 -y
    conda activate cv_comp

## 2. CUDA 버전 확인 (GPU 환경인 경우)

원격 서버에서 CUDA 드라이버 버전을 먼저 확인합니다.

    nvidia-smi

출력에서 CUDA Version을 확인합니다. (예: 12.2)

※ CUDA 드라이버가 12.2여도, PyTorch는 보통 cu121 빌드를 사용합니다.

## 3. PyTorch 설치

자신의 환경에 맞는 명령어 하나만 실행합니다.

    # CPU 환경
    pip install torch torchvision

    # GPU 환경 (CUDA 12.x 권장)
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

## 4. 프로젝트 패키지 설치

requirements 파일을 기준으로 패키지를 설치합니다.

    pip install -r requirements.txt
    pip install -r requirements-dev.txt

## 5. Jupyter 커널 등록

노트북 사용을 위해 커널을 등록합니다.

    python -m ipykernel install --user --name cv_comp --display-name "Python(cv_comp)"

## 6. 환경 변수 설정 (W&B)

.env 파일을 생성하고 값을 입력합니다.

    cp .env.example .env

.env 파일 내용:

    WANDB_API_KEY=your_api_key
    WANDB_ENTITY=your_entity
    WANDB_PROJECT=document-type-classification

## 7. 설치 확인

아래 코드를 실행하여 정상 설치 여부를 확인합니다.

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

## 핵심 원칙

- conda는 환경 생성만 사용
- 패키지 설치는 pip + requirements로 통일
- torch는 CPU/GPU 환경에 맞게 선택 설치
- 하드코딩 금지, config 기반으로 관리
