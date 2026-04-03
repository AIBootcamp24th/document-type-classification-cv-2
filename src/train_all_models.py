from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from src.utils.logger import setup_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run training for all models")
    parser.add_argument("--base", type=str, default="configs/base.yaml")
    parser.add_argument("--data", type=str, default="configs/data.yaml")
    parser.add_argument("--train", type=str, default="configs/train.yaml")
    parser.add_argument("--inference", type=str, default="configs/inference.yaml")
    parser.add_argument("--model_dir", type=str, default="configs/model")
    return parser.parse_args()


def resolve_execution_root(*path_candidates: str | Path) -> Path:
    for path_candidate in path_candidates:
        path = Path(path_candidate).resolve()
        config_dir = path.parent if path.is_file() else path

        if (
            config_dir.name == "configs"
            and config_dir.parent.parent.name == "experiments"
        ):
            return config_dir.parent

    return Path.cwd()


def get_log_dir(
    data_path: str,
    train_path: str,
    inference_path: str,
) -> str:
    execution_root = resolve_execution_root(
        data_path,
        train_path,
        inference_path,
    )
    return str(execution_root / "outputs" / "run_logs")


def get_model_paths(model_dir: str) -> list[str]:
    model_dir_path = Path(model_dir).resolve()

    if not model_dir_path.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    # 공용: configs/model/*.yaml
    if model_dir_path.name == "model" and model_dir_path.parent.name == "configs":
        model_paths = sorted(model_dir_path.glob("*.yaml"))
        if not model_paths:
            raise ValueError(f"No model yaml files found in: {model_dir}")
        return [str(p) for p in model_paths]

    # 개인: experiments/{name}/configs/model.yaml
    if (
        model_dir_path.name == "configs"
        and model_dir_path.parent.parent.name == "experiments"
    ):
        model_path = model_dir_path / "model.yaml"
        if not model_path.exists():
            raise FileNotFoundError(f"Model config not found: {model_path}")
        return [str(model_path)]

    raise ValueError(
        "Unsupported model_dir. Use either 'configs/model' or "
        "'experiments/{name}/configs'."
    )


def run_command(cmd: list[str], logger) -> None:
    logger.info(f"[RUN] {' '.join(cmd)}")

    process = subprocess.run(cmd, check=False)

    if process.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")


def main() -> None:
    args = parse_args()
    logger = setup_logger(
        "train_all_models",
        get_log_dir(args.data, args.train, args.inference),
    )

    model_paths = get_model_paths(args.model_dir)

    for model_path in model_paths:
        logger.info(f"[TRAIN START] {model_path}")

        run_command(
            [
                "python",
                "-m",
                "src.train",
                "--base",
                args.base,
                "--data",
                args.data,
                "--train",
                args.train,
                "--inference",
                args.inference,
                "--model",
                model_path,
            ],
            logger,
        )

    logger.info("[ALL TRAIN DONE]")


if __name__ == "__main__":
    main()
