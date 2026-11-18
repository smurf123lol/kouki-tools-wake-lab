# Wake Lab

Инструмент подготовки датасетов wake-word для проекта Kouki. Вынесен из репозитория
`kouki-station` (`tools/wake-lab`) в отдельный репозиторий `modules/kouki-tools-wake-lab`
(задача T159 П1): инструмент ходит в API Kouki и работает с данными, поэтому живёт рядом
с модулями, а не в прошивке станции.

## Стадии

Единственное место, где перечислены стадии, — реестр `src/wake_lab/stages/registry.py`.
Состав стадий и их доступность:

| Стадия    | Доступна | Что делает |
| --------- | -------- | ---------- |
| `harvest` | нет      | сбор размеченных сэмплов через API Kouki — реализуется в П3 (T159) |
| `promote` | да       | дедуп, holdout-сплит и запись корпуса в CAS |
| `report`  | да       | сводка по корпусу и выкачанным данным (JSON) |
| `train`   | нет      | обучение модели wake-word — не входит в T159 (см. спеку T159 §3) |

Стадия `promote`:

- читает baseline из `data/baseline` (`manifest.jsonl` + `sha256`, read-only) и
  проверяет, что baseline не изменён (PathGuard);
- собирает размеченные сэмплы из `data/telemetry_harvested/labeled/<категория>`
  (категории: `TP`, `CONFUSER`, `SPEECH_NEG`, `NOISE`; файлы `.wav` 16 кГц, моно,
  PCM_16 и sidecar `.json` с `start_s`/`end_s` и служебными метаданными);
- дедуплицирует по sha256 и по IoU интервалов (порог `dedup_iou_threshold`);
- ограничивает долю confuser-сэмплов (`confuser_ratio`);
- делит запись на `train`/`holdout` по группам (`GroupShuffleSplit` по
  speaker/station/date или по файлу);
- коммитит результат в CAS-хранилище: `artifacts/dataset_v<N>.jsonl` +
  `artifacts/provenance.json`. Каталог `artifacts/` в git не входит.

Стадия `report` читает последний манифест `dataset_v<N>.jsonl` (по самому большому
`N`) и считает по нему: количество записей, разбивку по категориям, размеры
`train`/`holdout`, sha256 корпуса (sha256 от sha256 записей), время последнего
`promote`. Рядом отдаётся сводка по выкачанным данным в
`data/telemetry_harvested/labeled/<категория>` (число `.wav`). Если манифеста нет,
стадия не падает: `artifacts.latest = null`, `records = 0` и предупреждение
«сначала выполните stage promote».

Стадии `harvest` и `train` зарегистрированы, но недоступны: `available: false` и
понятная причина. Их запуск завершается **кодом 3** и текстом причины (это не
«unknown stage»). Неизвестное имя стадии — **код 2** и список доступных стадий.

## Конфигурация

`configs/default.yaml` валидируется схемой (`src/wake_lab/schema.py`) при каждом
запуске CLI:

- обязательные секции и ключи: `paths.baseline`, `paths.telemetry_labeled`,
  `paths.artifacts`, `api.base_url`;
- `promote.test_size` и `promote.dedup_iou_threshold` — строго в интервале `(0, 1)`;
- `promote.categories` — только `TP`, `CONFUSER`, `SPEECH_NEG`, `NOISE`;
- любой неизвестный ключ отвергается с полным путём ключа
  (например, `promote.unknown_option: неизвестный ключ`).

Токен API берётся только из окружения и в конфиг не пишется.

## Структура

```
configs/                 конфигурация стадий (default.yaml, stages/promote.yaml)
src/wake_lab/
  paths.py               PathGuard: пути проекта, read-only baseline, sha256 baseline
  artifacts.py           ArtifactStore: CAS, версии манифестов, provenance
  schema.py              валидация конфигурации (ошибки с путём ключа)
  stages/base.py         StageName/StageContext/StageResult/Stage (Protocol)
  stages/registry.py     реестр стадий: имена, описания, available, run
  stages/promote.py      стадия promote
  stages/harvest.py      каркас harvest (П3): недоступна, явная ошибка
  stages/report.py       стадия report: сводка JSON
  stages/train.py        train: недоступна, явная ошибка
  cli.py                 typer CLI (команды stage/report)
tests/                   тесты (pytest): artifacts, paths, promote, schema,
                         registry, report, cli
```

## Установка

Нужен Python >= 3.12. Через `uv`:

```bash
uv venv
uv pip install -e ".[dev]"
```

Или обычным pip:

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
```

## Тесты и линтер

```bash
pytest -q
ruff check .
```

## Запуск

```bash
# список стадий и их доступность
wake-lab stage list

# конкретная стадия (доступны promote и report)
wake-lab stage promote --config configs/default.yaml
wake-lab stage promote --config configs/default.yaml --dry-run

# сводка JSON по корпусу и выкачанным данным
wake-lab report --config configs/default.yaml
```

Результат стадии печатается JSON-ом (`StageResult`: stage/status/outputs/metrics/
warnings/elapsed_s), при `status=failed` CLI завершается кодом 1. Команды `stage list`,
`stage <name>` и `report` — единственный вход; отдельной команды `run` нет.

## Где лежат данные

Пути задаются в `configs/default.yaml` (`paths.baseline`, `paths.telemetry_labeled`,
`paths.artifacts`) относительно `project_root`. Каталоги `data/` и `artifacts/` — рабочие
данные и результаты, в git не входят.
