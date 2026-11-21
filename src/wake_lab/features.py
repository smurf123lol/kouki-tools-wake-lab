from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import soundfile as sf

SAMPLE_RATE = 16000
TOTAL_SAMPLES = 24000  # 1.5 с при 16 кГц (148 кадров по 160 сэмплов)
FRAME_LEN = 480        # 30 мс
FRAME_STEP = 160       # 10 мс
N_FFT = 512
N_MELS = 40
NUM_FRAMES = 148
L1_FRAMES = 74
L1_MELS = 20
MEL_EPS = 1e-6
POWER_DIV = 480.0

# 40 треугольных mel-фильтров — 1:1 с прошивкой ESP32-S3 (app_wake.cpp: kMelBinEdges)
MEL_BIN_EDGES = [
    2, 4, 5, 7, 9, 10, 12, 15, 17, 19, 22, 24, 27, 30, 33, 36, 40, 44,
    48, 52, 56, 61, 66, 71, 77, 83, 89, 96, 103, 110, 118, 127, 135,
    145, 155, 165, 177, 188, 201, 214, 228, 243,
]

HANN_WINDOW = np.array(
    [0.5 * (1.0 - math.cos(2.0 * math.pi * i / (FRAME_LEN - 1))) for i in range(FRAME_LEN)],
    dtype=np.float32,
)


def build_mel_filterbank() -> np.ndarray:
    """Строит матрицу 40 mel-фильтров размерности (40, 257)."""
    fb = np.zeros((N_MELS, N_FFT // 2 + 1), dtype=np.float32)
    for m in range(N_MELS):
        left, center, right = MEL_BIN_EDGES[m], MEL_BIN_EDGES[m + 1], MEL_BIN_EDGES[m + 2]
        for k in range(left, center):
            fb[m, k] = (k - left) / (center - left)
        for k in range(center, right):
            fb[m, k] = (right - k) / (right - center)
    return fb


MEL_FB = build_mel_filterbank()


def compute_logmel_batch(windows: np.ndarray) -> np.ndarray:
    """Преобразует батч окон (B, 24000) в нормализованные mel-спектрограммы (B, 148, 40).

    Полностью повторяет расчет из firmware (app_wake.cpp) и train_wake_pair.py:
    мощность = |rfft|^2 / 480, mel = сумма по треугольникам, log(e + 1e-6),
    затем нормализация (x - mean) / std по 148x40 ячейкам окна.
    """
    windows = np.ascontiguousarray(windows, dtype=np.float32)
    b = windows.shape[0]
    frames = np.lib.stride_tricks.sliding_window_view(windows, FRAME_LEN, axis=1)[:, ::FRAME_STEP, :]
    frames = np.ascontiguousarray(frames[:, :NUM_FRAMES, :]) * HANN_WINDOW[None, None, :]
    spectrum = np.fft.rfft(frames, n=N_FFT, axis=2)
    power = (spectrum.real * spectrum.real + spectrum.imag * spectrum.imag) / POWER_DIV
    mel = power @ MEL_FB.T
    spec = np.log(mel + MEL_EPS)
    flat = spec.reshape(b, -1)
    mean = flat.mean(axis=1, keepdims=True)
    std = np.sqrt(np.maximum(flat.var(axis=1, keepdims=True), 1e-6))
    return ((flat - mean) / std).reshape(b, NUM_FRAMES, N_MELS).astype(np.float32)


def reduce_to_l1_batch(specs148x40: np.ndarray) -> np.ndarray:
    """Уменьшает спектрограммы L2 (B, 148, 40) до входа L1 (B, 74, 20) через 2x2 pooling."""
    b = specs148x40.shape[0]
    mel = specs148x40.reshape(b, NUM_FRAMES, L1_MELS, 2).mean(axis=3)
    return mel.reshape(b, L1_FRAMES, 2, L1_MELS).mean(axis=2).astype(np.float32)


def read_audio_pcm(path: Path | str) -> np.ndarray:
    """Загружает аудиофайл и возвращает одномерный массив float32 (16 кГц)."""
    data, sr = sf.read(path, dtype="float32", always_2d=True)
    mono = data.mean(axis=1) if data.shape[1] > 1 else data[:, 0]
    if sr != SAMPLE_RATE:
        # Простая интерполяция если частота дискретизации отличается
        num_target = round(len(mono) * SAMPLE_RATE / sr)
        mono = np.interp(
            np.linspace(0, len(mono), num_target, endpoint=False),
            np.arange(len(mono)),
            mono,
        ).astype(np.float32)
    return mono


def place_in_window(pcm: np.ndarray, rng: np.random.Generator, is_positive: bool) -> np.ndarray:
    """Размещает аудиосигнал в окне фиксированного размера TOTAL_SAMPLES (24000)."""
    n = len(pcm)
    if n == TOTAL_SAMPLES:
        return pcm.astype(np.float32)

    if n < TOTAL_SAMPLES:
        diff = TOTAL_SAMPLES - n
        if is_positive:
            # Центрируем ключевое слово с небольшим случайным джиттером
            max_jitter = min(diff // 4, 1200)
            jitter = rng.integers(-max_jitter, max_jitter + 1) if max_jitter > 0 else 0
            left = max(0, min(diff, diff // 2 + jitter))
            right = diff - left
            return np.pad(pcm, (left, right)).astype(np.float32)
        else:
            left = rng.integers(0, diff + 1)
            return np.pad(pcm, (left, diff - left)).astype(np.float32)
    else:
        # n > TOTAL_SAMPLES
        if is_positive:
            # Ищем пик энергии и берем срез вокруг него
            win = 480
            env = np.convolve(np.abs(pcm), np.ones(win, dtype=np.float32) / win, mode="same")
            peak = int(np.argmax(env))
            start = max(0, min(n - TOTAL_SAMPLES, peak - TOTAL_SAMPLES // 2))
            return pcm[start : start + TOTAL_SAMPLES].astype(np.float32)
        else:
            start = rng.integers(0, n - TOTAL_SAMPLES + 1)
            return pcm[start : start + TOTAL_SAMPLES].astype(np.float32)


class AudioAugmenter:
    """Накладывает аудио-аугментации на этапе подготовки признаков (для train split)."""

    def __init__(self, noise_pool: Sequence[np.ndarray], rng: np.random.Generator):
        self.noise_pool = list(noise_pool)
        self.rng = rng

    def augment(self, window: np.ndarray, is_positive: bool) -> np.ndarray:
        out = window.copy()

        # 1. Случайное изменение громкости (gain 0.7 .. 1.3)
        gain = self.rng.uniform(0.7, 1.3)
        out = out * gain

        # 2. Добавление фонового шума с SNR 10..25 dB
        if self.noise_pool and self.rng.random() < 0.65:
            noise_idx = self.rng.integers(0, len(self.noise_pool))
            noise_sample = self.noise_pool[noise_idx]
            noise_win = place_in_window(noise_sample, self.rng, is_positive=False)

            p_sig = np.mean(out * out) + 1e-12
            p_noise = np.mean(noise_win * noise_win) + 1e-12
            target_snr_db = self.rng.uniform(10.0, 25.0)
            snr_linear = 10.0 ** (target_snr_db / 10.0)

            scale = math.sqrt(p_sig / (p_noise * snr_linear))
            out = out + noise_win * scale

        # Клиппинг для предотвращения артефактов
        return np.clip(out, -1.0, 1.0).astype(np.float32)
