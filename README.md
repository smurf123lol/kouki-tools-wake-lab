# Wake Lab

Инструмент подготовки датасетов и двухкаскадного детектора wake-word («эй гин») для проекта Kouki. Вынесен в отдельный репозиторий `modules/kouki-tools-wake-lab` (задачи T159, T163, T164).

---

## 1. Конвейер стадий

Стадии регистрируются в `src/wake_lab/stages/registry.py`. Список стадий и статус: `wake-lab stage list`.

| Стадия | Доступна | Что делает |
|---|---|---|
| `harvest` | да | Сбор размеченных аудио-сэмплов через API Kouki (`GET /api/v1/telemetry/labeled`, `--limit` обязателен) |
| `promote` | да | Дедупликация (SHA256 + IoU), GroupShuffleSplit (train / holdout) и регистрация корпуса в CAS |
| `prepare` | да | Извлечение Log-Mel спектрограмм L1/L2, нарезка фона, аугментация и экспорт в `features.safetensors` |
| `train` | да | GPU-обучение двухкаскадного детектора на Rust/Burn (`burn-wgpu` / Vulkan 1.3), INT8 квантование, экспорт Си-заголовков |
| `calibrate` | да | 3D-сетка порогов, Парето-фронтир, генерация `wake_model_profiles.json` (Recall, Стандарт, Строгий) и отчёта калибровки |
| `report` | да | JSON-сводка по текущему CAS-корпусу и выкачанным записям |

---

## 2. Быстрый старт

### На сервере (GPU AMD Radeon RX 5600 XT, Vulkan 1.3 RADV):

```bash
cd modules/kouki-tools-wake-lab

# 1. Проверить доступные стадии
/srv/kouki-workspace/_tools/uv/uv run wake-lab stage list

# 2. Выкачать данные из API Kouki (требуется KOUKI_API_TOKEN)
export KOUKI_API_TOKEN="$(sudo -n kouki-admin api-token)"
/srv/kouki-workspace/_tools/uv/uv run wake-lab stage harvest --limit 10

# 3. Дедуплицировать корпус и подготовить CAS-сплит
/srv/kouki-workspace/_tools/uv/uv run wake-lab stage promote

# 4. Рассчитать Log-Mel спектрограммы и аугментации в тензоры
/srv/kouki-workspace/_tools/uv/uv run wake-lab stage prepare

# 5. Обучить модель на GPU по дефолтному конфигу
/srv/kouki-workspace/_tools/uv/uv run wake-lab stage train

# 6. Откалибровать пороги профилей для станции
/srv/kouki-workspace/_tools/uv/uv run wake-lab stage calibrate

# 7. Посмотреть JSON-сводку
/srv/kouki-workspace/_tools/uv/uv run wake-lab report
```

### На Windows:

На рабочей Windows-машине уже установлены Python 3.12 и `uv`. Для прямого запуска с Windows:
```powershell
cd z:\modules\kouki-tools-wake-lab
uv run wake-lab stage list
```
*(Примечание: для стадии `train` на Windows требуется установленный Rust с MSVC Build Tools для компиляции `wake-trainer.exe`, либо запуск стадии на сервере через SSH).*

---

## 3. Конфигурация без правок кода

Все гиперпараметры обучения, веса категорий, пороги и структура каталогов задаются через YAML-конфиги в каталоге `configs/`. **Править исходный код Python или Rust не требуется.**

### Каталог `configs/`:
* `configs/default.yaml` — референсная конфигурация (базовые пути, параметры майнинга и динамического переноса).
* `configs/presets/production-synergy.yaml` — **Топ-1 (Synergy)**: $th=0.35, k=0.25, \gamma=2.0, w_{conf}=1.3$, 80 эпох (максимальный баланс качества, высокое подавление шума).
* `configs/presets/confuser-strict.yaml` — **Топ-2 (Confuser Penalty)**: $th=0.50, k=0.15, \gamma=1.0, w_{conf}=1.5$, 80 эпох (минимальный ложный пробой по ТВ и разговорам: $< 3\%$).
* `configs/presets/isolation-baseline.yaml` — Строгая изоляция ($k_{easy} = 0.0$).
* `configs/presets/fast-experiment.yaml` — Быстрый пробный прогон (15 эпох).
* `configs/experiments/custom-theories-example.yaml` — Пример декларативного свипа гипотез.

### Запуск готового пресета:
```bash
# Обучение пресета Топ-1
wake-lab stage train --config configs/presets/production-synergy.yaml

# Калибровка порогов для этого же пресета
wake-lab stage calibrate --config configs/presets/production-synergy.yaml
```

---

## 4. Параметры секции `train`

Секция `train` в конфигурационном файле поддерживает следующие настройки:

```yaml
train:
  out_dir: artifacts/models/my_experiment    # каталог артефактов (Си-заголовки, метрики)
  epochs_l1: 60                             # эпохи L1 микро-CNN
  epochs_l2: 60                             # эпохи L2 (Xtensa SIMD-16)
  batch_size: 64                            # размер батча
  lr_l1: 0.0015                             # стартовый LR L1 (Cosine Annealing)
  lr_l2: 0.0010                             # стартовый LR L2

  # Параметры майнинга и k-изоляции
  th_hard: 0.35                             # порог L1 для отбора hard negatives
  k_easy: 0.25                              # коэффициент переноса легких негативов (0.0 = изоляция)
  gamma_easy: 2.0                           # степень затухания: 1.0 = линейная, 2.0 = квадратичная

  # Веса категорий (Weighted BCE)
  pos_weight_l1: 2.5                        # вес позитивного класса в L1
  pos_weight_l2: 2.0                        # вес позитивного класса в L2
  conf_weight_l2: 1.3                       # штраф за конфузеры в лоссе L2
  conf_cap_ratio: 0.70                      # доля конфузеров в пуле негативов L2

  # Error Repeat (двухпрогонный перевзвес ошибок)
  err_repeat: 2                             # кратность дублирования ошибочных окон
  err_th_l2: 0.65                           # порог ошибки для перевзвеса
  seed: 42
```

---

## 5. Свип гипотез и бенчмарк теорий

Инструмент поддерживает два режима бенчмарка:

### Вариант А: Встроенный нативный свип (14 теорий на Rust)
Запускается одним флагом `--theories`:
```bash
# Бенчмарк 14 теорий по 15 эпох на теорию
wake-lab stage train --theories --epochs 15
```
Автоматически ранжирует модели по интегральному счёту, экспортирует заголовки лучшей модели и генерирует `theories_benchmark.md` и `theories_benchmark.json`.

### Вариант Б: Кастомный свип из YAML без компиляции Rust
В секции `train` конфига передаётся список `theories`:
```yaml
train:
  out_dir: artifacts/models/my_sweep
  theories:
    - id: EXP-01
      name: "Тест 1"
      th_hard: 0.35
      k_easy: 0.25
      gamma_easy: 2.0
      conf_weight_l2: 1.3
      epochs_l1: 15
      epochs_l2: 15
    - id: EXP-02
      name: "Тест 2"
      th_hard: 0.40
      k_easy: 0.15
      gamma_easy: 1.0
      conf_weight_l2: 1.6
      epochs_l1: 15
      epochs_l2: 15
```
Запуск:
```bash
wake-lab stage train --config configs/experiments/custom-theories-example.yaml
```
Каждая теория обучается в своём подкаталоге, затем строится единая сводная таблица, а лучшая модель копируется в `out_dir`.

---

## 6. Тесты и линтер

```bash
# Тесты Python (48 тестов)
uv run pytest

# Тесты нативного крейта Rust (8 тестов)
cd crates/wake-trainer && cargo test
```
