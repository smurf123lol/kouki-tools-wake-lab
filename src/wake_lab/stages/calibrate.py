from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np

from .base import StageContext, StageName, StageResult


class CalibrateStage:
    """Стадия калибровки пороговых кривых (Парето-фронтир) и генерации

    wake_model_profiles.json для прошивки станции.
    """

    name = StageName.CALIBRATE

    def run(self, ctx: StageContext) -> StageResult:
        started = time.monotonic()
        logger = ctx.logger
        warnings: list[str] = []

        models_dir = ctx.paths.artifacts_dir / "models"
        preds_path = models_dir / "predictions.json"

        if not preds_path.exists():
            raise FileNotFoundError(
                f"Не найден файл предсказаний holdout: {preds_path}. "
                "Сначала выполните обучение: wake-lab stage train"
            )

        logger.info(f"Загрузка предсказаний holdout из: {preds_path}")
        with preds_path.open("r", encoding="utf-8") as f:
            preds_data = json.load(f)

        p1 = np.array(preds_data["p1"], dtype=np.float32)
        p2 = np.array(preds_data["p2"], dtype=np.float32)
        y = np.array(preds_data["y"], dtype=np.float32)
        y_l1 = np.array(preds_data["y_l1"], dtype=np.float32)

        pos_mask = y >= 0.99
        conf_mask = (y < 0.01) & (y_l1 > 0.05)
        neg_mask = (y < 0.01) & (y_l1 <= 0.05)
        non_pos_mask = ~pos_mask

        n_pos = int(np.sum(pos_mask))
        n_conf = int(np.sum(conf_mask))
        n_neg = int(np.sum(neg_mask))
        logger.info(
            "Выборка калибровки: %d позитивов, %d конфузеров, %d чистых негативов",
            n_pos,
            n_conf,
            n_neg,
        )

        # 3D Сетка параметров каскада Multiplicative:
        # p1 >= l1_invoke & p2 >= l2_floor & (p1 * p2) >= combo_th
        l1_grid = [0.25, 0.30, 0.35, 0.40, 0.45]
        l2_floor_grid = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
        combo_grid = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]

        candidates = []
        for l1_inv in l1_grid:
            for l2_fl in l2_floor_grid:
                for c_th in combo_grid:
                    passed = (p1 >= l1_inv) & (p2 >= l2_fl) & ((p1 * p2) >= c_th)
                    rec = float(np.mean(passed[pos_mask])) if n_pos > 0 else 0.0
                    fpr_conf = float(np.mean(passed[conf_mask])) if n_conf > 0 else 0.0
                    fpr_neg = float(np.mean(passed[neg_mask])) if n_neg > 0 else 0.0
                    fpr_all = float(np.mean(passed[non_pos_mask])) if len(non_pos_mask) > 0 else 0.0

                    candidates.append({
                        "l1_invoke": l1_inv,
                        "l2_floor": l2_fl,
                        "combo_th": c_th,
                        "recall": rec,
                        "conf_fpr": fpr_conf,
                        "neg_fpr": fpr_neg,
                        "total_fpr": fpr_all,
                    })

        # Парето-фронтир (максимизация Recall, минимизация total_fpr)
        candidates.sort(key=lambda c: (-c["recall"], c["total_fpr"]))
        pareto = []
        min_fpr_seen = float("inf")
        for c in candidates:
            if c["total_fpr"] < min_fpr_seen:
                pareto.append(c)
                min_fpr_seen = c["total_fpr"]

        logger.info("Найдено %d точек на Парето-фронтире", len(pareto))

        # Отбор 3 профилей:
        def pick_closest(target_rec: float, pool: list[dict[str, Any]]) -> dict[str, Any]:
            valid = [c for c in pool if c["recall"] >= target_rec]
            if valid:
                return min(valid, key=lambda c: c["total_fpr"])
            return min(pool, key=lambda c: abs(c["recall"] - target_rec))

        prof_recall = pick_closest(0.75, pareto)
        prof_standard = pick_closest(0.72, [c for c in pareto if c["recall"] <= 0.74] or pareto)
        prof_strict = pick_closest(0.66, [c for c in pareto if c["recall"] <= 0.70] or pareto)

        # Вычисление sha256 весов моделей
        l1_path = models_dir / "wake_model_ey_gin_l1.h"
        l2_path = models_dir / "wake_model_ey_gin_l2.h"
        l1_sha = hashlib.sha256(l1_path.read_bytes()).hexdigest() if l1_path.exists() else ""
        l2_sha = hashlib.sha256(l2_path.read_bytes()).hexdigest() if l2_path.exists() else ""

        # Метрики AUC из metrics.json
        metrics_file = models_dir / "metrics.json"
        l1_auc = 0.0
        l2_auc = 0.0
        if metrics_file.exists():
            try:
                m_data = json.loads(metrics_file.read_text(encoding="utf-8"))
                l1_auc = m_data.get("l1", {}).get("auc", 0.0)
                l2_auc = m_data.get("l2", {}).get("auc", 0.0)
            except Exception as e:
                warnings.append(f"Не удалось прочитать metrics.json: {e}")

        k_easy_val = ctx.config.get("train", {}).get("k_easy", 0.15)
        profiles_doc = {
            "model_version": f"t164-burn_wgpu-dyn_k{k_easy_val}",
            "name": "ey_gin_two_stage",
            "arch_l1": "base",
            "decision_mode": "multiplicative",
            "weights_sha256": {
                "l1": l1_sha,
                "l2": l2_sha,
            },
            "calibration_metrics": {
                "zero_leak_auc_all": round((l1_auc + l2_auc) / 2.0, 4),
                "auc_l1": round(l1_auc, 4),
                "auc_l2": round(l2_auc, 4),
                "val_samples": len(y),
            },
            "profiles": {
                "1": {
                    "id": "recall",
                    "name": "Recall",
                    "description": "Высокая чувствительность — тихий голос и расстояние",
                    "l1_invoke": prof_recall["l1_invoke"],
                    "l1_th": prof_recall["l1_invoke"],
                    "l2_floor": prof_recall["l2_floor"],
                    "l2_nom": round(min(0.95, prof_recall["l2_floor"] + 0.15), 2),
                    "combo_th": prof_recall["combo_th"],
                    "measured": {
                        "recall_val": round(prof_recall["recall"], 4),
                        "neg_fpr": round(prof_recall["neg_fpr"], 4),
                        "conf_fpr": round(prof_recall["conf_fpr"], 4),
                        "total_fpr": round(prof_recall["total_fpr"], 4),
                    },
                },
                "2": {
                    "id": "standard",
                    "name": "Стандарт",
                    "description": "Сбалансированная калибровка",
                    "l1_invoke": prof_standard["l1_invoke"],
                    "l1_th": prof_standard["l1_invoke"],
                    "l2_floor": prof_standard["l2_floor"],
                    "l2_nom": round(min(0.95, prof_standard["l2_floor"] + 0.10), 2),
                    "combo_th": prof_standard["combo_th"],
                    "measured": {
                        "recall_val": round(prof_standard["recall"], 4),
                        "neg_fpr": round(prof_standard["neg_fpr"], 4),
                        "conf_fpr": round(prof_standard["conf_fpr"], 4),
                        "total_fpr": round(prof_standard["total_fpr"], 4),
                    },
                },
                "3": {
                    "id": "lowest_fpr",
                    "name": "Строгий",
                    "description": "Минимум ложных срабатываний (отсечение ТВ и шума)",
                    "l1_invoke": prof_strict["l1_invoke"],
                    "l1_th": prof_strict["l1_invoke"],
                    "l2_floor": prof_strict["l2_floor"],
                    "l2_nom": round(min(0.95, prof_strict["l2_floor"] + 0.05), 2),
                    "combo_th": prof_strict["combo_th"],
                    "measured": {
                        "recall_val": round(prof_strict["recall"], 4),
                        "neg_fpr": round(prof_strict["neg_fpr"], 4),
                        "conf_fpr": round(prof_strict["conf_fpr"], 4),
                        "total_fpr": round(prof_strict["total_fpr"], 4),
                    },
                },
            },
        }

        # Запись в artifacts/models/wake_model_profiles.json
        out_profiles_path = models_dir / "wake_model_profiles.json"
        out_profiles_path.write_text(
            json.dumps(profiles_doc, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        logger.info("Паспорт профилей сохранен в: %s", out_profiles_path)

        # Отчет калибровки calibration_report.md
        report_md = self._generate_report_md(profiles_doc, pareto[:15])
        report_path = models_dir / "calibration_report.md"
        report_path.write_text(report_md, encoding="utf-8")
        logger.info("Отчет калибровки сохранен в: %s", report_path)

        # Синхронизация со станцией, если каталог существует
        station_profiles = (
            ctx.paths.project_root.parent
            / "kouki-station"
            / "firmware"
            / "main"
            / "wake_model_profiles.json"
        )
        if station_profiles.parent.exists():
            station_profiles.write_text(
                json.dumps(profiles_doc, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            logger.info("Паспорт профилей синхронизирован со станцией: %s", station_profiles)

        return StageResult(
            stage=self.name,
            status="ok",
            outputs={
                "profiles_path": str(out_profiles_path),
                "report_path": str(report_path),
                "profiles": profiles_doc["profiles"],
            },
            metrics={
                "recall_prof_recall": profiles_doc["profiles"]["1"]["measured"]["recall_val"],
                "recall_prof_standard": profiles_doc["profiles"]["2"]["measured"]["recall_val"],
                "recall_prof_strict": profiles_doc["profiles"]["3"]["measured"]["recall_val"],
                "fpr_prof_strict": profiles_doc["profiles"]["3"]["measured"]["total_fpr"],
            },
            warnings=warnings,
            elapsed_s=round(time.monotonic() - started, 3),
        )

    def _generate_report_md(self, doc: dict[str, Any], pareto: list[dict[str, Any]]) -> str:
        profs = doc["profiles"]
        md = []
        md.append("# Отчет калибровки профилей Wake-Word (Multiplicative)\n")
        md.append(f"Версия модели: `{doc['model_version']}`\n")
        md.append("## 1. Откалиброванные профили для станции\n")
        md.append("| Профиль | L1 invoke | L2 floor | Combo Th | Recall (Holdout) | Neg FPR | Confuser FPR | Total FPR |")
        md.append("|---|---|---|---|---|---|---|---|")
        for k in ["1", "2", "3"]:
            p = profs[k]
            m = p["measured"]
            md.append(
                f"| **{p['name']} ({k})** | {p['l1_invoke']:.2f} | {p['l2_floor']:.2f} | {p['combo_th']:.2f} | "
                f"**{m['recall_val']*100:.2f}%** | {m['neg_fpr']*100:.2f}% | {m['conf_fpr']*100:.2f}% | {m['total_fpr']*100:.2f}% |"
            )

        md.append("\n## 2. Точки Парето-фронтира (Recall vs FPR)\n")
        md.append("| L1 invoke | L2 floor | Combo Th | Recall | Neg FPR | Confuser FPR | Total FPR |")
        md.append("|---|---|---|---|---|---|---|")
        for pt in pareto:
            md.append(
                f"| {pt['l1_invoke']:.2f} | {pt['l2_floor']:.2f} | {pt['combo_th']:.2f} | "
                f"{pt['recall']*100:.2f}% | {pt['neg_fpr']*100:.2f}% | {pt['conf_fpr']*100:.2f}% | {pt['total_fpr']*100:.2f}% |"
            )
        return "\n".join(md) + "\n"
