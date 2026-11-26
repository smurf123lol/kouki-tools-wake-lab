from __future__ import annotations

import json
import time
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from safetensors.numpy import save_file

from ..features import (
    AudioAugmenter,
    compute_logmel_batch,
    place_in_window,
    read_audio_pcm,
    reduce_to_l1_batch,
)
from .base import StageContext, StageName, StageResult


@dataclass
class AudioItem:
    path: Path
    category: str       # TP, CONFUSER, SPEECH_NEG, NOISE
    is_positive: bool
    is_confuser: bool
    split: str          # train / holdout
    source: str = "raw" # raw / augmented


@dataclass(frozen=True)
class SampleWeights:
    """Единое описание меток и весов для L1 и L2 по категории сэмпла."""
    y: float
    y_l1: float
    w_l1: float
    w_l2: float


def get_sample_weights(is_positive: bool, is_confuser: bool) -> SampleWeights:
    """Единая функция расчета целевых меток и базовых весов потерь L1/L2."""
    if is_positive:
        return SampleWeights(y=1.0, y_l1=1.0, w_l1=1.0, w_l2=1.0)
    if is_confuser:
        return SampleWeights(y=0.0, y_l1=0.10, w_l1=0.40, w_l2=1.0)
    # Фоновая речь / шум
    return SampleWeights(y=0.0, y_l1=0.0, w_l1=0.20, w_l2=0.15)


class WindowCollector:
    """Единый коллектор фреймов с автоматическим назначением меток и весов L1/L2."""

    def __init__(self) -> None:
        self.windows: list[np.ndarray] = []
        self.y: list[float] = []
        self.y_l1: list[float] = []
        self.w_l1: list[float] = []
        self.w_l2: list[float] = []
        self.split: list[int] = []

    def add(
        self,
        window: np.ndarray,
        is_positive: bool,
        is_confuser: bool,
        split: int,
    ) -> None:
        meta = get_sample_weights(is_positive, is_confuser)
        self.windows.append(window)
        self.y.append(meta.y)
        self.y_l1.append(meta.y_l1)
        self.w_l1.append(meta.w_l1)
        self.w_l2.append(meta.w_l2)
        self.split.append(split)

    def __len__(self) -> int:
        return len(self.windows)


class PrepareStage:
    """Стадия подготовки признаков (П2): расчет Log-Mel спектрограмм L1/L2,

    аугментации и экспорт матрицы тензоров в features.safetensors.
    """

    name = StageName.PREPARE

    def run(self, ctx: StageContext) -> StageResult:
        started = time.monotonic()
        warnings: list[str] = []
        logger = ctx.logger

        # 1. Поиск promoted корпуса
        dataset_path = self._find_dataset_manifest(ctx)
        if dataset_path is None or not dataset_path.exists():
            raise FileNotFoundError(
                "Не найден размеченный датасет после стадии promote (artifacts/dataset_*.jsonl). "
                "Сначала запустите: wake-lab stage promote"
            )

        logger.info(f"Загрузка корпуса из: {dataset_path}")
        items: list[AudioItem] = []
        noise_pool: list[np.ndarray] = []

        with dataset_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    path_str = rec.get("path")
                    if not path_str:
                        continue
                    p = Path(path_str)
                    if not p.is_absolute():
                        p = ctx.paths.project_root / p
                    if not p.exists():
                        warnings.append(f"Файл не найден: {p}")
                        continue

                    cat = rec.get("category", "NOISE").upper()
                    is_pos = (cat == "TP" or rec.get("label") == "positive")
                    is_conf = (cat == "CONFUSER" or "confuser" in rec.get("label", ""))
                    split = rec.get("split", "train")
                    source = rec.get("source", "raw")

                    items.append(AudioItem(
                        path=p,
                        category=cat,
                        is_positive=is_pos,
                        is_confuser=is_conf,
                        split=split,
                        source=source,
                    ))

                    if cat == "NOISE" or "noise" in rec.get("label", ""):
                        try:
                            pcm_noise = read_audio_pcm(p)
                            if len(pcm_noise) >= 8000:
                                noise_pool.append(pcm_noise)
                        except Exception as e:
                            warnings.append(f"Не удалось прочесть шум {p}: {e}")

                except json.JSONDecodeError:
                    continue

        # Также подключаем неизменяемый baseline из data/baseline если есть
        baseline_dir = ctx.paths.baseline_dir
        if baseline_dir.exists():
            self._scan_baseline_storage(baseline_dir, items, noise_pool, warnings, ctx.paths.project_root)

        if not items:
            raise ValueError("Корпус пуст: нет доступных аудиофайлов для извлечения признаков")

        logger.info(
            f"Собрано {len(items)} исходных сэмплов (шумовой пул: {len(noise_pool)} файлов)"
        )

        # 2. Нарезка окон и аугментации
        rng = np.random.default_rng(42)
        augmenter = AudioAugmenter(noise_pool, rng)
        collector = WindowCollector()

        train_variants_raw_pos = int(ctx.config.get("prepare", {}).get("variants_raw_pos", 11))
        train_variants_raw_conf = int(ctx.config.get("prepare", {}).get("variants_raw_conf", 3))

        for it in items:
            try:
                pcm = read_audio_pcm(it.path)
            except Exception as e:
                warnings.append(f"Ошибка чтения {it.path}: {e}")
                continue

            if len(pcm) < 1600:  # менее 100 мс — мусор
                continue

            # Длинный фон (> 3 с): нарезка скользящими окнами с шагом 1.0 с и энергогейтом
            if not it.is_positive and not it.is_confuser and len(pcm) >= 48000:
                stride = 16000  # 1.0 с
                n_sliced = 0
                for st in range(0, len(pcm) - 24000 + 1, stride):
                    chunk = pcm[st : st + 24000]
                    # Энергогейт прошивки: пиковая амплитуда >= 500 из 32768
                    if np.max(np.abs(chunk)) * 32768.0 >= 500.0:
                        collector.add(chunk, is_positive=False, is_confuser=False, split=(0 if it.split == "train" else 1))
                        n_sliced += 1
                        if n_sliced >= 300:  # ограничение на один длинный файл
                            break
                continue

            if it.split == "train":
                # Базовое окно
                base_win = place_in_window(pcm, rng, it.is_positive)
                collector.add(base_win, it.is_positive, it.is_confuser, split=0)

                # Аугментации применяются в один шаг ко всем сырым данным (baseline raw и telemetry).
                # Повторная аугментация уже пред-аугментированных файлов (augmented/) запрещена.
                if it.source != "augmented":
                    reps = train_variants_raw_pos if it.is_positive else (train_variants_raw_conf if it.is_confuser else 0)
                    for _ in range(reps):
                        aug_win = augmenter.augment(base_win, it.is_positive)
                        collector.add(aug_win, it.is_positive, it.is_confuser, split=0)
            else:
                # Holdout: без случайных аугментаций (ровно 1 окно)
                win = place_in_window(pcm, rng, it.is_positive)
                collector.add(win, it.is_positive, it.is_confuser, split=1)

        total_windows = len(collector)
        logger.info(f"Сформировано {total_windows} окон по 24000 сэмплов (1.5 с)")

        # 3. Расчет Mel-спектрограмм батчами
        batch_size = 256
        x_l2_batches: list[np.ndarray] = []
        windows_arr = np.array(collector.windows, dtype=np.float32)

        for i in range(0, total_windows, batch_size):
            chunk = windows_arr[i : i + batch_size]
            x_l2_batches.append(compute_logmel_batch(chunk))

        X_l2 = np.concatenate(x_l2_batches, axis=0) if x_l2_batches else np.zeros((0, 148, 40), dtype=np.float32)
        X_l1 = reduce_to_l1_batch(X_l2)

        y = np.array(collector.y, dtype=np.float32)
        y_l1 = np.array(collector.y_l1, dtype=np.float32)
        w_l1 = np.array(collector.w_l1, dtype=np.float32)
        w_l2 = np.array(collector.w_l2, dtype=np.float32)
        split = np.array(collector.split, dtype=np.uint8)

        # 4. Сохранение в features.safetensors
        cache_dir = ctx.paths.artifacts_dir / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        out_safetensors = cache_dir / "features.safetensors"

        tensors = {
            "X_l1": X_l1,
            "X_l2": X_l2,
            "y": y,
            "y_l1": y_l1,
            "w_l1": w_l1,
            "w_l2": w_l2,
            "split": split,
        }

        save_file(tensors, str(out_safetensors))

        # Запись в CAS
        ref = ctx.artifacts.put_file(out_safetensors, name="features.safetensors")

        train_count = int(np.sum(split == 0))
        val_count = int(np.sum(split == 1))
        pos_count = int(np.sum(y == 1.0))

        metrics = {
            "total_windows": total_windows,
            "train_windows": train_count,
            "holdout_windows": val_count,
            "positive_windows": pos_count,
            "l1_shape": list(X_l1.shape),
            "l2_shape": list(X_l2.shape),
            "cas_hash": ref.cas_hash,
            "file_size_bytes": out_safetensors.stat().st_size,
        }

        elapsed = time.monotonic() - started
        logger.info(
            f"Стадия prepare завершена за {elapsed:.2f}с: {out_safetensors} "
            f"({metrics['file_size_bytes'] / (1024 * 1024):.1f} МБ, CAS {ref.cas_hash[:12]})"
        )

        return StageResult(
            stage=self.name,
            status="ok",
            outputs={"path": str(out_safetensors), "cas_hash": ref.cas_hash},
            metrics=metrics,
            warnings=warnings,
            elapsed_s=elapsed,
        )

    def _find_dataset_manifest(self, ctx: StageContext) -> Path | None:
        """Находит файл датасета в CAS или artifacts."""
        param_path = ctx.params.get("dataset")
        if param_path:
            return Path(param_path)

        index_file = ctx.paths.artifacts_dir / "index.jsonl"
        if index_file.exists():
            with index_file.open("r", encoding="utf-8") as f:
                for line in reversed(f.readlines()):
                    if not line.strip():
                        continue
                    try:
                        entry = json.loads(line)
                        if entry.get("name", "").startswith("dataset_"):
                            cas_hash = entry.get("cas_hash")
                            if cas_hash:
                                return ctx.artifacts.resolve(cas_hash)
                    except Exception:  # noqa: S112
                        continue
        return None

    def _scan_baseline_storage(
        self,
        base_dir: Path,
        items: list[AudioItem],
        noise_pool: list[np.ndarray],
        warnings: list[str],
        project_root: Path,
    ) -> None:
        """Сканирует только raw-файлы baseline storage (без augmented/, чтобы аугментация шла в один шаг)."""
        existing_paths = {it.path.resolve() for it in items}

        for p in sorted(base_dir.rglob("*.wav")):
            if "augmented" in p.parts:
                continue
            p_res = p.resolve()
            if p_res in existing_paths:
                continue

            path_str = str(p).lower()
            if "pos" in path_str or "positive" in path_str:
                cat = "TP"
                is_pos = True
                is_conf = False
            elif "confuser" in path_str or "hard_neg" in path_str:
                cat = "CONFUSER"
                is_pos = False
                is_conf = True
            else:
                cat = "NOISE" if "noise" in path_str else "SPEECH_NEG"
                is_pos = False
                is_conf = False

            # Детерминированное распределение 80% train / 20% holdout без утечки аугментаций
            split = "holdout" if (zlib.crc32(p.name.encode("utf-8")) % 100) < 20 else "train"
            source = "raw"

            items.append(AudioItem(
                path=p,
                category=cat,
                is_positive=is_pos,
                is_confuser=is_conf,
                split=split,
                source=source,
            ))

            if cat == "NOISE":
                try:
                    pcm = read_audio_pcm(p)
                    if len(pcm) >= 8000:
                        noise_pool.append(pcm)
                except Exception as e:
                    warnings.append(f"Ошибка чтения шума baseline {p}: {e}")
