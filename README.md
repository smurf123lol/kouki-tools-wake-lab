# Wake Lab

Модульный инструментарий подготовки данных и обучения wake-word моделей для проекта Kouki.

## Структура
- `configs/` — конфигурационные YAML-файлы для стадий и экспериментов
- `src/wake_lab/` — исходный код пакета:
  - `paths.py` — PathGuard (проверка целостности и строгий read-only для `data/baseline`)
  - `artifacts.py` — ArtifactStore (CAS хранилище)
  - `stages/` — независимые стадии пайплайна (`promote`, `prepare`, `train_l1`, `train_l2`, etc.)
  - `cli.py` — CLI интерфейс
- `tests/` — модульные и регрессионные тесты

## Запуск
```bash
# Запуск конкретной стадии
python -m wake_lab.cli stage promote --config configs/default.yaml

# Или через wake-lab run
python -m wake_lab.cli run --config configs/default.yaml
```
