from __future__ import annotations

import json
import logging
import subprocess
import time
from pathlib import Path

from .base import StageContext, StageName, StageResult


class TrainStage:
    """Обучение двухкаскадной модели wake-word на GPU через Rust и Burn (T163).

    Запускает бинарник `wake-trainer` (WGPU/Vulkan на RX 5600 XT),
    генерирует C-заголовки L1/L2 и регистрирует их в CAS.
    """

    name = StageName.TRAIN

    def run(self, ctx: StageContext) -> StageResult:
        started = time.monotonic()
        logger = ctx.logger

        # 1. Поиск матрицы признаков
        features_path = ctx.paths.artifacts_dir / "cache" / "features.safetensors"
        if not features_path.exists():
            raise FileNotFoundError(
                f"Матрица признаков не найдена: {features_path}. "
                "Сначала запустите стадию подготовки: wake-lab stage prepare"
            )

        # 2. Поиск бинарника wake-trainer
        trainer_bin = ctx.paths.project_root / "crates" / "wake-trainer" / "target" / "release" / "wake-trainer"
        if not trainer_bin.exists():
            logger.info("Бинарник wake-trainer не найден, сборка cargo build --release...")
            cmd_build = ["cargo", "build", "--release"]
            build_res = subprocess.run(
                cmd_build,
                cwd=str(ctx.paths.project_root / "crates" / "wake-trainer"),
                capture_output=True,
                text=True,
            )
            if build_res.returncode != 0:
                raise RuntimeError(
                    f"Ошибка сборки wake-trainer:\n{build_res.stderr}\n{build_res.stdout}"
                )

        # 3. Формирование аргументов запуска
        out_dir = ctx.paths.artifacts_dir / "models"
        out_dir.mkdir(parents=True, exist_ok=True)

        train_cfg = {**ctx.config.get("train", {}), **ctx.params}
        epochs_l1 = int(train_cfg.get("epochs_l1", 20))
        epochs_l2 = int(train_cfg.get("epochs_l2", 20))
        batch_size = int(train_cfg.get("batch_size", 64))
        lr_l1 = float(train_cfg.get("lr_l1", 1.5e-3))
        lr_l2 = float(train_cfg.get("lr_l2", 1.0e-3))
        th_hard = float(train_cfg.get("th_hard", 0.35))
        err_repeat = int(train_cfg.get("err_repeat", 3))
        seed = int(train_cfg.get("seed", 42))
        k_easy = float(train_cfg.get("k_easy", 0.0))
        gamma_easy = float(train_cfg.get("gamma_easy", 1.0))
        sweep_theories = bool(train_cfg.get("sweep_theories", False))

        cmd = [
            str(trainer_bin),
            "--features", str(features_path),
            "--out-dir", str(out_dir),
            "--epochs-l1", str(epochs_l1),
            "--epochs-l2", str(epochs_l2),
            "--batch-size", str(batch_size),
            "--lr-l1", str(lr_l1),
            "--lr-l2", str(lr_l2),
            "--th-hard", str(th_hard),
            "--err-repeat", str(err_repeat),
            "--seed", str(seed),
            "--k-easy", str(k_easy),
            "--gamma-easy", str(gamma_easy),
        ]

        if "neg_cap_l2" in train_cfg:
            cmd.extend(["--neg-cap-l2", str(int(train_cfg["neg_cap_l2"]))])

        if sweep_theories:
            cmd.append("--sweep-theories")

        logger.info(f"Запуск GPU обучения: {' '.join(cmd)}")
        run_res = subprocess.run(cmd, capture_output=True, text=True)
        if run_res.returncode != 0:
            raise RuntimeError(
                f"wake-trainer завершился с ошибкой (код {run_res.returncode}):\n{run_res.stderr}\n{run_res.stdout}"
            )

        logger.info(run_res.stdout)

        # 4. Проверка и публикация артефактов в CAS
        l1_header = out_dir / "wake_model_ey_gin_l1.h"
        l2_header = out_dir / "wake_model_ey_gin_l2.h"
        metrics_file = out_dir / "metrics.json"

        if not l1_header.exists() or not l2_header.exists():
            raise FileNotFoundError("Си-заголовки модели не были созданы wake-trainer")

        ref_l1 = ctx.artifacts.put_file(l1_header, name="wake_model_ey_gin_l1.h")
        ref_l2 = ctx.artifacts.put_file(l2_header, name="wake_model_ey_gin_l2.h")

        metrics_data = {}
        if metrics_file.exists():
            try:
                metrics_data = json.loads(metrics_file.read_text(encoding="utf-8"))
                ctx.artifacts.put_file(metrics_file, name="metrics.json")
            except Exception as e:
                logger.warning(f"Не удалось прочесть metrics.json: {e}")

        elapsed = time.monotonic() - started
        logger.info(
            f"Стадия train успешно завершена за {elapsed:.2f}с: "
            f"L1 CAS {ref_l1.cas_hash[:12]}, L2 CAS {ref_l2.cas_hash[:12]}"
        )

        return StageResult(
            stage=self.name,
            status="ok",
            outputs={
                "l1_header": str(l1_header),
                "l2_header": str(l2_header),
                "l1_cas_hash": ref_l1.cas_hash,
                "l2_cas_hash": ref_l2.cas_hash,
                "metrics_file": str(metrics_file),
            },
            metrics=metrics_data,
            warnings=[],
            elapsed_s=elapsed,
        )
