from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any

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
        bin_name = "wake-trainer.exe" if sys.platform == "win32" else "wake-trainer"
        trainer_bin = ctx.paths.project_root / "crates" / "wake-trainer" / "target" / "release" / bin_name
        if not trainer_bin.exists():
            logger.info("Бинарник wake-trainer не найден, сборка cargo build --release...")
            cmd_build = ["cargo", "build", "--release"]
            build_res = subprocess.run(
                cmd_build,
                cwd=str(ctx.paths.project_root / "crates" / "wake-trainer"),
                capture_output=True,
                text=True,
                check=False,
            )
            if build_res.returncode != 0:
                raise RuntimeError(
                    f"Ошибка сборки wake-trainer:\n{build_res.stderr}\n{build_res.stdout}"
                )

        # 3. Формирование аргументов запуска
        train_cfg = {**ctx.config.get("train", {}), **ctx.params}
        out_dir_override = train_cfg.get("out_dir")
        out_dir = (ctx.paths.project_root / out_dir_override).resolve() if out_dir_override else ctx.paths.artifacts_dir / "models"
        out_dir.mkdir(parents=True, exist_ok=True)

        theories_list = train_cfg.get("theories")
        if isinstance(theories_list, list) and theories_list:
            metrics_data, bench_json, bench_md = self._run_custom_theories_sweep(
                trainer_bin=trainer_bin,
                features_path=features_path,
                out_dir=out_dir,
                base_cfg=train_cfg,
                theories=theories_list,
                logger=logger,
            )
        else:
            epochs_override = train_cfg.get("epochs")
            epochs_l1 = int(epochs_override if epochs_override is not None else train_cfg.get("epochs_l1", 60))
            epochs_l2 = int(epochs_override if epochs_override is not None else train_cfg.get("epochs_l2", 60))
            batch_size = int(train_cfg.get("batch_size", 64))
            lr_l1 = float(train_cfg.get("lr_l1", 1.5e-3))
            lr_l2 = float(train_cfg.get("lr_l2", 1.0e-3))
            th_hard = float(train_cfg.get("th_hard", 0.50))
            err_th_l2 = float(train_cfg.get("err_th_l2", 0.65))
            err_repeat = int(train_cfg.get("err_repeat", 3))
            seed = int(train_cfg.get("seed", 42))
            k_easy = float(train_cfg.get("k_easy", 0.15))
            gamma_easy = float(train_cfg.get("gamma_easy", 1.0))
            pos_weight_l1 = float(train_cfg.get("pos_weight_l1", 2.5))
            pos_weight_l2 = float(train_cfg.get("pos_weight_l2", 2.0))
            conf_weight_l2 = float(train_cfg.get("conf_weight_l2", 1.0))
            conf_cap_ratio = float(train_cfg.get("conf_cap_ratio", 0.70))
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
                "--err-th-l2", str(err_th_l2),
                "--seed", str(seed),
                "--k-easy", str(k_easy),
                "--gamma-easy", str(gamma_easy),
                "--pos-weight-l1", str(pos_weight_l1),
                "--pos-weight-l2", str(pos_weight_l2),
                "--conf-weight-l2", str(conf_weight_l2),
                "--conf-cap-ratio", str(conf_cap_ratio),
                "--l2-c0", str(int(train_cfg.get("l2_c0", 16))),
                "--l2-c1", str(int(train_cfg.get("l2_c1", 32))),
                "--l2-c2", str(int(train_cfg.get("l2_c2", 64))),
                "--l2-fc1", str(int(train_cfg.get("l2_fc1", 32))),
            ]

            if "neg_cap_l2" in train_cfg:
                cmd.extend(["--neg-cap-l2", str(int(train_cfg["neg_cap_l2"]))])

            if sweep_theories:
                cmd.append("--sweep-theories")

            logger.info(f"Запуск GPU обучения: {' '.join(cmd)}")
            run_res = subprocess.run(cmd, capture_output=True, text=True, check=False)
            if run_res.returncode != 0:
                raise RuntimeError(
                    f"wake-trainer завершился с ошибкой (код {run_res.returncode}):\n{run_res.stderr}\n{run_res.stdout}"
                )

            logger.info(run_res.stdout)

            metrics_file = out_dir / "metrics.json"
            metrics_data = {}
            if metrics_file.exists():
                try:
                    metrics_data = json.loads(metrics_file.read_text(encoding="utf-8"))
                except Exception as e:
                    logger.warning(f"Не удалось прочесть metrics.json: {e}")

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

        bench_json = out_dir / "theories_benchmark.json"
        bench_md = out_dir / "theories_benchmark.md"
        if bench_json.exists():
            ctx.artifacts.put_file(bench_json, name="theories_benchmark.json")
        if bench_md.exists():
            ctx.artifacts.put_file(bench_md, name="theories_benchmark.md")

        elapsed = time.monotonic() - started
        logger.info(
            f"Стадия train успешно завершена за {elapsed:.2f}с: "
            f"L1 CAS {ref_l1.cas_hash[:12]}, L2 CAS {ref_l2.cas_hash[:12]}"
        )

        outputs = {
            "l1_header": str(l1_header),
            "l2_header": str(l2_header),
            "l1_cas_hash": ref_l1.cas_hash,
            "l2_cas_hash": ref_l2.cas_hash,
            "metrics_file": str(metrics_file),
        }
        if bench_json.exists():
            outputs["theories_benchmark_json"] = str(bench_json)
        if bench_md.exists():
            outputs["theories_benchmark_md"] = str(bench_md)

        return StageResult(
            stage=self.name,
            status="ok",
            outputs=outputs,
            metrics=metrics_data,
            warnings=[],
            elapsed_s=elapsed,
        )

    def _run_custom_theories_sweep(
        self,
        trainer_bin: Path,
        features_path: Path,
        out_dir: Path,
        base_cfg: dict[str, Any],
        theories: list[dict[str, Any]],
        logger: Any,
    ) -> tuple[dict[str, Any], Path, Path]:
        results: list[dict[str, Any]] = []
        best_score = -1e9
        best_theory_id = ""
        best_theory_dir: Path | None = None

        logger.info(f"Запуск кастомного свипа из {len(theories)} теорий...")

        for idx, t in enumerate(theories):
            t_id = str(t.get("id", f"T{idx+1:02d}"))
            t_name = str(t.get("name", t_id))
            t_hyp = str(t.get("hypothesis", ""))
            logger.info(f">>> [ТЕОРИЯ {idx+1}/{len(theories)}] {t_name} ({t_id})")

            t_out = out_dir / t_id
            t_out.mkdir(parents=True, exist_ok=True)

            merged = {**base_cfg, **t}
            epochs_l1 = int(merged.get("epochs_l1", 15))
            epochs_l2 = int(merged.get("epochs_l2", 15))

            cmd = [
                str(trainer_bin),
                "--features", str(features_path),
                "--out-dir", str(t_out),
                "--epochs-l1", str(epochs_l1),
                "--epochs-l2", str(epochs_l2),
                "--batch-size", str(int(merged.get("batch_size", 64))),
                "--lr-l1", str(float(merged.get("lr_l1", 1.5e-3))),
                "--lr-l2", str(float(merged.get("lr_l2", 1.0e-3))),
                "--th-hard", str(float(merged.get("th_hard", 0.50))),
                "--err-repeat", str(int(merged.get("err_repeat", 2))),
                "--err-th-l2", str(float(merged.get("err_th_l2", 0.65))),
                "--seed", str(int(merged.get("seed", 42))),
                "--k-easy", str(float(merged.get("k_easy", 0.15))),
                "--gamma-easy", str(float(merged.get("gamma_easy", 1.0))),
                "--pos-weight-l1", str(float(merged.get("pos_weight_l1", 2.5))),
                "--pos-weight-l2", str(float(merged.get("pos_weight_l2", 2.0))),
                "--conf-weight-l2", str(float(merged.get("conf_weight_l2", 1.0))),
                "--conf-cap-ratio", str(float(merged.get("conf_cap_ratio", 0.70))),
                "--l2-c0", str(int(merged.get("l2_c0", 16))),
                "--l2-c1", str(int(merged.get("l2_c1", 32))),
                "--l2-c2", str(int(merged.get("l2_c2", 64))),
                "--l2-fc1", str(int(merged.get("l2_fc1", 32))),
            ]

            if "neg_cap_l2" in merged:
                cmd.extend(["--neg-cap-l2", str(int(merged["neg_cap_l2"]))])

            res = subprocess.run(cmd, capture_output=True, text=True, check=False)
            if res.returncode != 0:
                raise RuntimeError(
                    f"Ошибка выполнения теории {t_id}:\n{res.stderr}\n{res.stdout}"
                )

            metrics_file = t_out / "metrics.json"
            m_data: dict[str, Any] = {}
            if metrics_file.exists():
                try:
                    m_data = json.loads(metrics_file.read_text(encoding="utf-8"))
                except Exception as e:
                    logger.warning(f"Не удалось прочесть metrics.json теории {t_id}: {e}")

            rec = float(m_data.get("multiplicative_recall", 0.0))
            fpr = float(m_data.get("multiplicative_fpr", 0.0))
            score = rec * 100.0 - (fpr * 100.0 * 5.0)

            results.append({
                "id": t_id,
                "name": t_name,
                "hypothesis": t_hyp,
                "k_easy": float(merged.get("k_easy", 0.15)),
                "gamma_easy": float(merged.get("gamma_easy", 1.0)),
                "th_hard": float(merged.get("th_hard", 0.50)),
                "pos_weight_l1": float(merged.get("pos_weight_l1", 2.5)),
                "pos_weight_l2": float(merged.get("pos_weight_l2", 2.0)),
                "conf_weight_l2": float(merged.get("conf_weight_l2", 1.0)),
                "conf_cap_ratio": float(merged.get("conf_cap_ratio", 0.70)),
                "err_repeat": int(merged.get("err_repeat", 2)),
                "epochs_l1": epochs_l1,
                "epochs_l2": epochs_l2,
                "metrics": m_data,
                "score": score,
            })

            logger.info(f"  Итог {t_id}: Mult Rec={rec*100:.2f}%, FPR={fpr*100:.2f}% | Счёт={score:.2f}")

            if score > best_score:
                best_score = score
                best_theory_id = t_id
                best_theory_dir = t_out

        if best_theory_dir is not None:
            logger.info(f">>> ЛУЧШАЯ ТЕОРИЯ: {best_theory_id} (Счёт: {best_score:.2f})")
            for fn in ["wake_model_ey_gin_l1.h", "wake_model_ey_gin_l2.h", "metrics.json", "predictions.json"]:
                src_f = best_theory_dir / fn
                if src_f.exists():
                    shutil.copy2(src_f, out_dir / fn)

        md_lines = [
            f"# Сводный бенчмарк теорий Wake-Word ({len(theories)} гипотез)\n",
            "| № | ID | Теория | $k_{easy}$ | $\\gamma$ | $th_{hard}$ | $pos_{L1/L2}$ | $w_{conf}$ | L1 AUC | L2 AUC | Hard Rec | Hard FPR | Mult Rec | Mult FPR | Счёт |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        for i, r in enumerate(results):
            win_mark = " **(WIN)**" if r["id"] == best_theory_id else ""
            m = r["metrics"]
            l1_auc = m.get("l1", {}).get("auc", 0.0)
            l2_auc = m.get("l2", {}).get("auc", 0.0)
            h_rec = m.get("hard_cascade_recall", 0.0) * 100.0
            h_fpr = m.get("hard_cascade_fpr", 0.0) * 100.0
            m_rec = m.get("multiplicative_recall", 0.0) * 100.0
            m_fpr = m.get("multiplicative_fpr", 0.0) * 100.0
            md_lines.append(
                f"| {i+1} | `{r['id']}`{win_mark} | {r['name']} | {r['k_easy']:.2f} | {r['gamma_easy']:.1f} | {r['th_hard']:.2f} | {r['pos_weight_l1']:.1f}/{r['pos_weight_l2']:.1f} | {r['conf_weight_l2']:.1f} | {l1_auc:.4f} | {l2_auc:.4f} | {h_rec:.2f}% | {h_fpr:.2f}% | {m_rec:.2f}% | {m_fpr:.2f}% | **{r['score']:.2f}** |"
            )

        bench_md = out_dir / "theories_benchmark.md"
        bench_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

        bench_json = out_dir / "theories_benchmark.json"
        report_data = {
            "timestamp": time.time(),
            "best_theory_id": best_theory_id,
            "best_score": best_score,
            "results": results,
        }
        bench_json.write_text(json.dumps(report_data, indent=2, ensure_ascii=False), encoding="utf-8")

        win_metrics = {}
        winner_metrics_file = out_dir / "metrics.json"
        if winner_metrics_file.exists():
            try:
                win_metrics = json.loads(winner_metrics_file.read_text(encoding="utf-8"))
            except Exception:
                pass

        return win_metrics, bench_json, bench_md
