# Инструкция для агентов: kouki-tools-wake-lab

Памятка для AI-агентов (Antigravity, DSH, Claude, Gemini), работающих с лабораторией wake-word Kouki.

---

## 1. Архитектура и среда исполнения

* **Файловая система:** Репозиторий находится на Linux-сервере (`/srv/kouki-workspace/modules/kouki-tools-wake-lab`), который смонтирован на Windows-машину как сетевой диск `Z:\modules\kouki-tools-wake-lab`.
* **Вычисления:** Обучение моделей (`stage train`) оптимизировано под нативный GPU-стек: Burn (`burn-wgpu`), Vulkan 1.3 / Mesa RADV на дискретной видеокарте **AMD Radeon RX 5600 XT** (`Navi 10`), физически установленной на сервере `192.168.1.157`.
* **Как запускать команды:**
  * Если агент находится в среде Windows (PowerShell): команды выполнения стадий и сборки нативного крейта отправляются на сервер через SSH:
    ```bash
    ssh smurf123lol@192.168.1.157 "cd /srv/kouki-workspace/modules/kouki-tools-wake-lab && /srv/kouki-workspace/_tools/uv/uv run wake-lab <команда>"
    ```
  * Если агент находится внутри DSH на сервере:
    ```bash
    cd /srv/kouki-workspace/modules/kouki-tools-wake-lab
    /srv/kouki-workspace/_tools/uv/uv run wake-lab <команда>
    ```

---

## 2. Главное правило: Не переписывать код ради экспериментов

**Никогда не хардкодьте параметры обучения в `pipeline.rs` или `train.py`.**  
Все параметры полностью управляются через YAML-файлы конфигурации в каталоге `configs/`:

* **Одиночные прогоны:** создайте или используйте пресет в `configs/presets/*.yaml` (например, `production-synergy.yaml` или `confuser-strict.yaml`).
* **Кастомный свип:** создайте файл в `configs/experiments/*.yaml` со списком `theories: [...]`. Конвейер автоматически прогонит каждую теорию, посчитает метрики, построит сводную таблицу в Markdown/JSON и скопирует веса победителя в `out_dir`.
* **Встроенный бенчмарк 14 теорий:** передайте флаг `--theories` и опционально `--epochs <N>`:
  ```bash
  uv run wake-lab stage train --theories --epochs 15
  ```

---

## 3. Типовой цикл работы над моделями

При запросе на переобучение или эксперименты с wake-word агент выполняет цикл:

1. **Проверка окружения и тестов:**
   ```bash
   uv run pytest
   (cd crates/wake-trainer && source $HOME/.cargo/env && cargo test)
   ```
2. **Подготовка признаков (если обновился датасет):**
   ```bash
   uv run wake-lab stage promote
   uv run wake-lab stage prepare
   ```
3. **Обучение:**
   ```bash
   uv run wake-lab stage train --config configs/presets/production-synergy.yaml
   ```
4. **Калибровка профилей станции (ОБЯЗАТЕЛЬНЫЙ ЭТАП):**
   ```bash
   uv run wake-lab stage calibrate --config configs/presets/production-synergy.yaml
   ```
   *Выход калибровки:* `artifacts/models/.../wake_model_profiles.json` и `calibration_report.md`.
5. **Проверка артефактов:**
   Убедиться, что сгенерированы Си-заголовки `wake_model_ey_gin_l1.h`, `wake_model_ey_gin_l2.h`, файл профилей `wake_model_profiles.json` и метрики `metrics.json`.

---

## 4. Подводные камни и их устранение

### 4.1 Права доступа (`Permission denied / os error 13`)
* **Причина:** Если кто-то выполнил глобальный `chmod 0664` на шаре, с бинарников и скриптов в `.venv/bin/` и `crates/wake-trainer/target/` слетает флаг исполнения (`+x`).
* **Решение:**
  ```bash
  chmod +x /srv/kouki-workspace/modules/kouki-tools-wake-lab/.venv/bin/*
  chmod +x /srv/kouki-workspace/modules/kouki-tools-wake-lab/crates/wake-trainer/target/release/wake-trainer
  ```

### 4.2 Cargo не найден в PATH на сервере
* **Причина:** Rust установлен в пользовательском каталоге `/home/smurf123lol/.cargo`.
* **Решение:** Перед вызовами `cargo` активировать окружение:
  ```bash
  source $HOME/.cargo/env && cargo ...
  ```

### 4.3 Токен API Kouki (`401 Unauthorized` на стадии `harvest`)
* **Правило:** Токен никогда не пишется в конфиги.
* **Решение:**
  ```bash
  export KOUKI_API_TOKEN="$(sudo -n kouki-admin api-token)"
  ```
