# Wake Lab

Инструмент подготовки датасетов wake-word для проекта Kouki. Вынесен из репозитория
`kouki-station` (`tools/wake-lab`) в отдельный репозиторий `modules/kouki-tools-wake-lab`
(задача T159 П1): инструмент ходит в API Kouki и работает с данными, поэтому живёт рядом
с модулями, а не в прошивке станции.

## Статус: что реально работает

Реализована **одна стадия — `promote`**. В исходной структуре упоминались `prepare`,
`train_l1`, `train_l2`, но их кода в репозитории нет: `wake-lab stage prepare` вернёт
ошибку `unknown stage: prepare`. Не считайте их существующими, пока они не появятся.

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

API-клиент, harvest и отчёты — следующие пакеты T159 (П2/П3), в этом репозитории их пока
нет.

## Структура

```
configs/                 конфигурация стадий (default.yaml, stages/promote.yaml)
src/wake_lab/
  paths.py               PathGuard: пути проекта, read-only baseline, sha256 baseline
  artifacts.py           ArtifactStore: CAS, версии манифестов, provenance
  stages/base.py         StageName/StageContext/StageResult/Stage (Protocol)
  stages/promote.py      стадия promote
  cli.py                 typer CLI (команды stage/run)
tests/                   4 теста в 3 файлах (pytest)
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

## Тесты

```bash
pytest -q
# 4 passed (test_artifacts, test_paths_guard, test_promote_stage)
```

## Запуск

```bash
# конкретная стадия (пока единственная — promote)
wake-lab stage promote --config configs/default.yaml

# сухой прогон без записи в artifacts/
wake-lab stage promote --config configs/default.yaml --dry-run

# общий вход: пока выполняет promote
wake-lab run --config configs/default.yaml
```

Результат печатается JSON-ом (`StageResult`: stage/status/outputs/metrics/warnings/
elapsed_s), при `status=failed` CLI завершается кодом 1.

## Где лежат данные

Пути задаются в `configs/default.yaml` (`paths.baseline`, `paths.telemetry_labeled`,
`paths.artifacts`) относительно `project_root`. Каталоги `data/` и `artifacts/` — рабочие
данные и результаты, в git не входят.
