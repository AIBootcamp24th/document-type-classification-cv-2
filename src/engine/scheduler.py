from __future__ import annotations

from typing import Any

from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, LambdaLR

def build_scheduler(cfg: Any, optimizer: Optimizer):
    """
    scheduler 생성 함수

    반환 형태:
    - warmup OFF → scheduler 객체 반환
    - warmup ON  → dict 반환
        {
            "warmup": LambdaLR,
            "main": base_scheduler,
            "warmup_epochs": int
        }
    """

    # ---------------------------
    # scheduler 사용 안함
    # ---------------------------
    if not cfg.scheduler.use:
        return None

    scheduler_name = cfg.scheduler.name.lower()

    # ---------------------------
    # base scheduler 생성
    # ---------------------------
    if scheduler_name == "cosineannealinglr":
        base_scheduler = CosineAnnealingLR(
            optimizer=optimizer,
            T_max=cfg.scheduler.t_max,
            eta_min=cfg.scheduler.eta_min,
        )

    elif scheduler_name == "steplr":
        base_scheduler = StepLR(
            optimizer=optimizer,
            step_size=cfg.scheduler.step_size,
            gamma=cfg.scheduler.gamma,
        )

    else:
        msg = f"Unsupported scheduler: {cfg.scheduler.name}"
        raise ValueError(msg)

    # ---------------------------
    # warmup 설정 확인
    # ---------------------------
    warmup_cfg = getattr(cfg.scheduler, "warmup", None)

    # warmup OFF → base scheduler 그대로 사용
    if warmup_cfg is None or not warmup_cfg.use:
        return base_scheduler

    warmup_epochs = warmup_cfg.warmup_epochs

    if warmup_epochs <= 0:
        raise ValueError("warmup_epochs must be > 0")

    # ---------------------------
    # warmup scheduler
    # ---------------------------
    def lr_lambda(current_epoch: int):
        """
        LambdaLR은 '비율'을 반환해야 함

        current_epoch:
        - 0부터 시작
        """
        if current_epoch < warmup_epochs:
            # 0 → 1까지 선형 증가
            return float(current_epoch + 1) / float(warmup_epochs)

        return 1.0

    warmup_scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

    # ---------------------------
    # wrapper 형태 반환
    # ---------------------------
    return {
        "warmup": warmup_scheduler,
        "main": base_scheduler,
        "warmup_epochs": warmup_epochs,
    }