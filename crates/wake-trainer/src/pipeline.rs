use crate::data::Dataset;
use crate::export::save_headers;
use crate::metrics::CascadeMetrics;
use crate::model::{L2ArchConfig, WakeNetL1, WakeNetL2};
use crate::trainer::{predict_l1, predict_l2, train_l1, train_l2, TrainConfig};
use burn::tensor::Device;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::Write;
use std::path::Path;

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct PipelineConfig {
    pub epochs_l1: usize,
    pub epochs_l2: usize,
    pub batch_size: usize,
    pub lr_l1: f64,
    pub lr_l2: f64,
    pub pos_weight_l1: f32,
    pub pos_weight_l2: f32,
    pub conf_weight_l2: f32,
    pub conf_cap_ratio: f32,
    pub th_hard: f32,
    pub err_repeat: usize,
    pub err_th_l2: f32,
    pub seed: u64,
    pub neg_cap_l2: Option<usize>,
    /// Множитель динамического переноса легких негативов (k_easy).
    /// При k_easy = 0.0 математически сводится к строгой изоляции (strict isolation).
    pub k_easy: f32,
    /// Степень уверенности для динамического переноса: w = k_easy * (p1 / th_hard)^gamma_easy.
    /// 1.0 = линейный перенос, 2.0 = квадратичный перенос (фокус на near-miss окнах).
    pub gamma_easy: f32,
    pub l2_c0: usize,
    pub l2_c1: usize,
    pub l2_c2: usize,
    pub l2_fc1: usize,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            epochs_l1: 60,
            epochs_l2: 60,
            batch_size: 64,
            lr_l1: 1.5e-3,
            lr_l2: 1.0e-3,
            pos_weight_l1: 2.5,
            pos_weight_l2: 2.0,
            conf_weight_l2: 1.0,
            conf_cap_ratio: 0.70,
            th_hard: 0.50,
            err_repeat: 3,
            err_th_l2: 0.65,
            seed: 42,
            neg_cap_l2: None,
            k_easy: 0.15,
            gamma_easy: 1.0,
            l2_c0: 16,
            l2_c1: 32,
            l2_c2: 64,
            l2_fc1: 32,
        }
    }
}

pub fn error_repeat(
    ds: &Dataset,
    probs: &[f32],
    labels: &[f32],
    th_err: f32,
    repeat: usize,
    include_pos: bool,
) -> Dataset {
    if repeat <= 1 {
        return ds.clone();
    }

    let mut extra_indices = Vec::new();

    for i in 0..ds.n_samples {
        let y = labels[i];
        let p = probs[i];
        let peak = ds.peaks[i];

        // Энергогейт: окна с пиком < 500 прошивка не инференсит
        let pass_gate = peak >= 500.0;

        let wrong_neg = y <= 0.01 && p >= th_err && pass_gate;
        let wrong_pos = include_pos && (y >= 0.99 && p < th_err);

        if wrong_neg || wrong_pos {
            for _ in 0..(repeat - 1) {
                extra_indices.push(i);
            }
        }
    }

    if extra_indices.is_empty() {
        return ds.clone();
    }

    let mut all_indices: Vec<usize> = (0..ds.n_samples).collect();
    all_indices.extend(extra_indices);
    all_indices.sort();

    ds.filter_by_indices(&all_indices)
}

pub fn mine_hard_negatives(
    model_l1: &WakeNetL1,
    train_ds: &Dataset,
    cfg: &PipelineConfig,
    device: &Device,
    batch_size: usize,
) -> Dataset {
    let p1 = predict_l1(model_l1, train_ds, device, batch_size);

    let mut pos_indices = Vec::new();
    let mut conf_indices = Vec::new();
    let mut bg_hard_indices = Vec::new();
    let mut bg_easy_indices = Vec::new();

    for i in 0..train_ds.n_samples {
        let is_pos = train_ds.y[i] >= 0.99;
        let is_confuser = train_ds.y[i] <= 0.01 && train_ds.y_l1[i] > 0.01;
        let is_bg = train_ds.y[i] <= 0.01 && train_ds.y_l1[i] <= 0.01;

        if is_pos {
            pos_indices.push(i);
        } else if is_confuser {
            conf_indices.push(i);
        } else if is_bg {
            if p1[i] >= cfg.th_hard {
                bg_hard_indices.push(i);
            } else if cfg.k_easy > 0.0 {
                bg_easy_indices.push(i);
            }
        }
    }

    let n_pos = pos_indices.len();
    let effective_cap = cfg.neg_cap_l2.unwrap_or_else(|| n_pos * 3 / 2);
    let cap_conf = ((effective_cap as f32) * cfg.conf_cap_ratio).round() as usize;
    let cap_hard = effective_cap.saturating_sub(cap_conf).max(1);
    let cap_easy = cap_hard;

    let orig_conf = conf_indices.len();
    let orig_bg_hard = bg_hard_indices.len();
    let orig_bg_easy = bg_easy_indices.len();

    // Сортируем конфузеры по убыванию скора L1 (берем самые сложные для L1)
    if conf_indices.len() > cap_conf {
        conf_indices.sort_by(|&a, &b| p1[b].partial_cmp(&p1[a]).unwrap_or(std::cmp::Ordering::Equal));
        conf_indices.truncate(cap_conf);
    }

    // Сортируем фоновые hard negatives по убыванию скора L1 (берем самые жесткие пробои)
    if bg_hard_indices.len() > cap_hard {
        bg_hard_indices.sort_by(|&a, &b| p1[b].partial_cmp(&p1[a]).unwrap_or(std::cmp::Ordering::Equal));
        bg_hard_indices.truncate(cap_hard);
    }

    // При k_easy > 0: сортируем легкие негативы по убыванию p1 (берем околопороговые near-miss окна)
    if cfg.k_easy > 0.0 && bg_easy_indices.len() > cap_easy {
        bg_easy_indices.sort_by(|&a, &b| p1[b].partial_cmp(&p1[a]).unwrap_or(std::cmp::Ordering::Equal));
        bg_easy_indices.truncate(cap_easy);
    }

    println!("  L2 майнинг негативов (th_hard = {:.2}, k_easy = {:.2}, gamma = {:.2}, conf_ratio = {:.2}, w_conf = {:.2}):",
             cfg.th_hard, cfg.k_easy, cfg.gamma_easy, cfg.conf_cap_ratio, cfg.conf_weight_l2);
    println!("    Позитивов: {}, Конфузеров: {}/{} (кап {}), Hard BG: {}/{} (кап {}), Easy BG: {}/{} (кап {})",
             pos_indices.len(), conf_indices.len(), orig_conf, cap_conf,
             bg_hard_indices.len(), orig_bg_hard, cap_hard,
             if cfg.k_easy > 0.0 { bg_easy_indices.len() } else { 0 }, orig_bg_easy, cap_easy);

    let mut keep_indices = pos_indices;
    keep_indices.extend(conf_indices);
    keep_indices.extend(bg_hard_indices);
    if cfg.k_easy > 0.0 {
        keep_indices.extend(bg_easy_indices);
    }
    keep_indices.sort();

    let mut l2_ds = train_ds.filter_by_indices(&keep_indices);

    // Динамическое взвешивание образцов для L2:
    // Позитивы сохраняют w_l2 = 1.0.
    // Конфузеры получают cfg.conf_weight_l2.
    // Фоновые hard negatives (p1 >= th_hard): вес 1.0.
    // Фоновые easy negatives (p1 < th_hard при k_easy > 0): динамический вес k_easy * (p1 / th_hard)^gamma_easy.
    for i in 0..l2_ds.n_samples {
        let orig_idx = keep_indices[i];
        let is_confuser = l2_ds.y[i] <= 0.01 && l2_ds.y_l1[i] > 0.01;
        let is_bg = l2_ds.y[i] <= 0.01 && l2_ds.y_l1[i] <= 0.01;
        if is_confuser {
            l2_ds.w_l2[i] = cfg.conf_weight_l2;
        } else if is_bg {
            let p = p1[orig_idx];
            if p >= cfg.th_hard {
                l2_ds.w_l2[i] = 1.0;
            } else if cfg.k_easy > 0.0 {
                let ratio = (p / cfg.th_hard).clamp(0.0, 1.0);
                l2_ds.w_l2[i] = cfg.k_easy * ratio.powf(cfg.gamma_easy);
            }
        }
    }

    l2_ds
}

pub fn train_cascade(
    train_ds: &Dataset,
    val_ds: &Dataset,
    cfg: &PipelineConfig,
    device: &Device,
) -> (WakeNetL1, WakeNetL2, CascadeMetrics, HoldoutPredictions) {
    // 1. Обучение L1
    println!("\n[Этап 1] Обучение модели L1 (микро-CNN)...");
    let model_l1 = WakeNetL1::new(device);
    let l1_cfg = TrainConfig {
        epochs: cfg.epochs_l1,
        batch_size: cfg.batch_size,
        lr: cfg.lr_l1,
        pos_weight: cfg.pos_weight_l1,
        seed: cfg.seed,
    };

    let (model_l1, l1_metrics) = if cfg.err_repeat > 1 {
        println!("  Пробный прогон L1 для поиска ошибок...");
        let (probe_l1, _) = train_l1(model_l1, train_ds, val_ds, &l1_cfg, device);
        let train_probs = predict_l1(&probe_l1, train_ds, device, cfg.batch_size * 2);
        let train_ds_rep = error_repeat(train_ds, &train_probs, &train_ds.y_l1, cfg.th_hard, cfg.err_repeat, false);
        println!("  L1 перевзвес: окон {} -> {}", train_ds.n_samples, train_ds_rep.n_samples);

        let final_model_l1 = WakeNetL1::new(device);
        train_l1(final_model_l1, &train_ds_rep, val_ds, &l1_cfg, device)
    } else {
        train_l1(model_l1, train_ds, val_ds, &l1_cfg, device)
    };

    println!("  L1 Val ROC-AUC: {:.4}, Recall @ Zero-FP: {:.2}%",
        l1_metrics.auc, l1_metrics.recall_at_zero_fp * 100.0);

    // 2. Отбор Hard Negatives и динамических Easy Negatives для L2
    println!("\n[Этап 2] Отбор обучающей выборки L2 (th_hard = {:.2}, k_easy = {:.2})...", cfg.th_hard, cfg.k_easy);
    let train_l2_ds = mine_hard_negatives(
        &model_l1,
        train_ds,
        cfg,
        device,
        cfg.batch_size * 2,
    );
    println!("  L2 train dataset: {} окон", train_l2_ds.n_samples);

    // 3. Обучение L2
    println!("\n[Этап 3] Обучение модели L2 (Xtensa SIMD-16 alignment: c0={}, c1={}, c2={}, fc1={})...",
        cfg.l2_c0, cfg.l2_c1, cfg.l2_c2, cfg.l2_fc1);
    let l2_arch = L2ArchConfig {
        c0: cfg.l2_c0,
        c1: cfg.l2_c1,
        c2: cfg.l2_c2,
        fc1: cfg.l2_fc1,
    };
    let model_l2 = WakeNetL2::new_with_arch(device, l2_arch);
    let l2_cfg = TrainConfig {
        epochs: cfg.epochs_l2,
        batch_size: cfg.batch_size,
        lr: cfg.lr_l2,
        pos_weight: cfg.pos_weight_l2,
        seed: cfg.seed + 1,
    };

    let (model_l2, l2_metrics) = if cfg.err_repeat > 1 {
        println!("  Пробный прогон L2 для поиска ошибок...");
        let (probe_l2, _) = train_l2(model_l2, &train_l2_ds, val_ds, &l2_cfg, device);
        let train_probs = predict_l2(&probe_l2, &train_l2_ds, device, cfg.batch_size * 2);
        let train_l2_rep = error_repeat(&train_l2_ds, &train_probs, &train_l2_ds.y, cfg.err_th_l2, cfg.err_repeat, true);
        println!("  L2 перевзвес: окон {} -> {}", train_l2_ds.n_samples, train_l2_rep.n_samples);

        let final_model_l2 = WakeNetL2::new_with_arch(device, l2_arch);
        train_l2(final_model_l2, &train_l2_rep, val_ds, &l2_cfg, device)
    } else {
        train_l2(model_l2, &train_l2_ds, val_ds, &l2_cfg, device)
    };

    println!("  L2 Val ROC-AUC: {:.4}, Recall @ Zero-FP: {:.2}%",
        l2_metrics.auc, l2_metrics.recall_at_zero_fp * 100.0);

    // 4. Комбинированные метрики каскада на holdout
    println!("\n[Этап 4] Расчёт каскадных метрик на holdout (val)...");
    let val_p1 = predict_l1(&model_l1, val_ds, device, cfg.batch_size * 2);
    let val_p2 = predict_l2(&model_l2, val_ds, device, cfg.batch_size * 2);

    let th_l1_nom = 0.50f64;
    let th_l2_nom = 0.65f64;

    let mut hard_hits = 0;
    let mut mult_hits = 0;
    let mut hard_fps = 0;
    let mut mult_fps = 0;
    let mut total_pos = 0;
    let mut total_neg = 0;

    let mult_th = 0.325f64;

    for i in 0..val_ds.n_samples {
        let is_pos = val_ds.y[i] >= 0.99;
        let is_neg = val_ds.y[i] <= 0.01;

        let p1 = val_p1[i] as f64;
        let p2 = val_p2[i] as f64;

        let hard_pass = p1 >= th_l1_nom && p2 >= th_l2_nom;
        let mult_pass = (p1 * p2) >= mult_th;

        if is_pos {
            total_pos += 1;
            if hard_pass { hard_hits += 1; }
            if mult_pass { mult_hits += 1; }
        } else if is_neg {
            total_neg += 1;
            if hard_pass { hard_fps += 1; }
            if mult_pass { mult_fps += 1; }
        }
    }

    let hard_cascade_recall = hard_hits as f64 / total_pos.max(1) as f64;
    let hard_cascade_fpr = hard_fps as f64 / total_neg.max(1) as f64;
    let multiplicative_recall = mult_hits as f64 / total_pos.max(1) as f64;
    let multiplicative_fpr = mult_fps as f64 / total_neg.max(1) as f64;

    println!("  Hard Cascade: Recall = {:.2}%, FPR = {:.2}% (th_l1={:.2}, th_l2={:.2})",
        hard_cascade_recall * 100.0, hard_cascade_fpr * 100.0, th_l1_nom, th_l2_nom);
    println!("  Multiplicative: Recall = {:.2}%, FPR = {:.2}% (th_mult={:.2})",
        multiplicative_recall * 100.0, multiplicative_fpr * 100.0, mult_th);

    let cascade_metrics = CascadeMetrics {
        l1: l1_metrics,
        l2: l2_metrics,
        hard_cascade_recall,
        hard_cascade_fpr,
        multiplicative_recall,
        multiplicative_fpr,
        th_l1: th_l1_nom,
        th_l2: th_l2_nom,
    };

    let preds = HoldoutPredictions {
        p1: val_p1,
        p2: val_p2,
        y: val_ds.y.clone(),
        y_l1: val_ds.y_l1.clone(),
    };

    (model_l1, model_l2, cascade_metrics, preds)
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct HoldoutPredictions {
    pub p1: Vec<f32>,
    pub p2: Vec<f32>,
    pub y: Vec<f32>,
    pub y_l1: Vec<f32>,
}

pub fn run_pipeline<P: AsRef<Path>, Q: AsRef<Path>>(
    features_path: P,
    out_dir: Q,
    cfg: &PipelineConfig,
    device: &Device,
) -> Result<CascadeMetrics, Box<dyn std::error::Error>> {
    println!("=== Запуск полного конвейера обучения Wake-Word на GPU ===");
    println!("Загрузка признаков из {:?}...", features_path.as_ref());
    let full_ds = Dataset::load_safetensors(features_path)?;
    let (train_ds, val_ds) = full_ds.split_train_val();
    println!("Выборка: train = {} окон, val/holdout = {} окон", train_ds.n_samples, val_ds.n_samples);

    let (model_l1, model_l2, cascade_metrics, preds) = train_cascade(&train_ds, &val_ds, cfg, device);

    // 5. Квантование и экспорт C-заголовков
    println!("\n[Этап 5] Квантование INT8 и экспорт Си-заголовков в {:?}...", out_dir.as_ref());
    save_headers(&out_dir, &model_l1, &model_l2, cascade_metrics.th_l1 as f32, cascade_metrics.th_l2 as f32)?;

    // 6. Запись отчёта метрик
    let metrics_path = out_dir.as_ref().join("metrics.json");
    let mut mf = File::create(&metrics_path)?;
    let json_bytes = serde_json::to_vec_pretty(&cascade_metrics)?;
    mf.write_all(&json_bytes)?;
    println!("Метрики записаны в {:?}", metrics_path);

    // 7. Запись предсказаний holdout для калибровки профилей
    let preds_path = out_dir.as_ref().join("predictions.json");
    let mut pf = File::create(&preds_path)?;
    let p_bytes = serde_json::to_vec(&preds)?;
    pf.write_all(&p_bytes)?;
    println!("Предсказания holdout записаны в {:?}", preds_path);

    println!("=== Конвейер успешно завершён! ===");
    Ok(cascade_metrics)
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct TheoryResult {
    pub id: String,
    pub name: String,
    pub hypothesis: String,
    pub k_easy: f32,
    pub gamma_easy: f32,
    pub th_hard: f32,
    pub pos_weight_l1: f32,
    pub pos_weight_l2: f32,
    pub conf_weight_l2: f32,
    pub conf_cap_ratio: f32,
    pub err_repeat: usize,
    pub epochs_l1: usize,
    pub epochs_l2: usize,
    pub metrics: CascadeMetrics,
    pub score: f64,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct TheoriesBenchmarkReport {
    pub timestamp: String,
    pub best_theory_id: String,
    pub best_score: f64,
    pub results: Vec<TheoryResult>,
}

pub fn get_theories_pool(epochs_l1: usize, epochs_l2: usize) -> Vec<(&'static str, &'static str, &'static str, PipelineConfig)> {
    let base = PipelineConfig {
        epochs_l1,
        epochs_l2,
        err_repeat: 2,
        ..PipelineConfig::default()
    };
    vec![
        // --- Блок 1: Подходы к k-изоляции ---
        (
            "T01-iso-baseline",
            "Строгая изоляция (Baseline)",
            "Строгое отсечение легких негативов (k_easy=0.0). В L2 попадают только hard negatives L1 и конфузеры.",
            PipelineConfig { k_easy: 0.0, ..base.clone() }
        ),
        (
            "T02-iso-wide-th035",
            "Изоляция + широкий майнинг (th=0.35)",
            "Изоляция (k=0.0), но порог L1 снижен до 0.35 для захвата большего пула hard negatives в L2.",
            PipelineConfig { k_easy: 0.0, th_hard: 0.35, ..base.clone() }
        ),
        (
            "T03-iso-strict-th060",
            "Изоляция + узкий майнинг (th=0.60)",
            "Изоляция (k=0.0), в L2 допускаются только самые критичные пробои L1 (p1 >= 0.60).",
            PipelineConfig { k_easy: 0.0, th_hard: 0.60, ..base.clone() }
        ),
        (
            "T04-dyn-lin-k008",
            "Ультра-мягкий перенос (Linear k=0.08)",
            "Слабая подпитка фонового градиента (w <= 0.08 * p1/th). Минимальное возмущение разделения.",
            PipelineConfig { k_easy: 0.08, gamma_easy: 1.0, ..base.clone() }
        ),
        (
            "T05-dyn-lin-k015",
            "Стандартный перенос (Linear k=0.15)",
            "Базовый динамический перенос легких негативов: линейная шкала уверенности до 0.15.",
            PipelineConfig { k_easy: 0.15, gamma_easy: 1.0, ..base.clone() }
        ),
        (
            "T06-dyn-lin-k030",
            "Агрессивный перенос (Linear k=0.30)",
            "Повышенный вес легких негативов (до 0.30). Проверка границы между подавлением шума и размытием.",
            PipelineConfig { k_easy: 0.30, gamma_easy: 1.0, ..base.clone() }
        ),
        (
            "T07-dyn-quad-k020",
            "Квадратичный фокус (Quad k=0.20, gamma=2.0)",
            "Квадратичное затухание w = 0.20 * (p1/th)^2: фокус строго на near-miss окнах у границы.",
            PipelineConfig { k_easy: 0.20, gamma_easy: 2.0, ..base.clone() }
        ),
        (
            "T08-dyn-quad-k035",
            "Квадратичный усиленный (Quad k=0.35, gamma=2.0)",
            "Сильный градиент на самой границе срабатывания при быстром квадратичном затухании в тишине.",
            PipelineConfig { k_easy: 0.35, gamma_easy: 2.0, ..base.clone() }
        ),
        (
            "T09-dyn-pow15-k015",
            "Промежуточная степень (gamma=1.5, k=0.15)",
            "Степенной перенос между линейным и квадратичным: компромисс ширины охвата и подавления шума.",
            PipelineConfig { k_easy: 0.15, gamma_easy: 1.5, ..base.clone() }
        ),

        // --- Блок 2: Настройка весов категорий (Positive, Confuser, Background) ---
        (
            "T10-weight-pos-boost",
            "Приоритет активации (PosWeight 3.5 / 3.0)",
            "Увеличенный штраф за пропуск слова: pos_weight L1=3.5, L2=3.0 для максимизации Recall тихого голоса.",
            PipelineConfig { pos_weight_l1: 3.5, pos_weight_l2: 3.0, k_easy: 0.15, ..base.clone() }
        ),
        (
            "T11-weight-pos-suppress",
            "Приоритет чистоты (PosWeight 1.8 / 1.5)",
            "Сниженный вес позитивов для жесткого подавления ложных срабатываний (фокус на Precision).",
            PipelineConfig { pos_weight_l1: 1.8, pos_weight_l2: 1.5, k_easy: 0.15, ..base.clone() }
        ),
        (
            "T12-weight-confuser-15",
            "Усиленный штраф конфузеров (w_conf=1.5)",
            "Повышенный вес конфузеров в L2 лоссе (w=1.5) для подавления срабатываний на похожие слова.",
            PipelineConfig { conf_weight_l2: 1.5, k_easy: 0.15, ..base.clone() }
        ),
        (
            "T13-pool-conf-heavy",
            "Высокая доля конфузеров в пуле (85%)",
            "Увеличение квоты конфузеров в L2 с 70% до 85% от объема негативов.",
            PipelineConfig { conf_cap_ratio: 0.85, k_easy: 0.15, ..base.clone() }
        ),
        (
            "T14-synergy-quad-th035",
            "Синергия: th=0.35 + Quad k=0.25 + w_conf=1.3",
            "Комплексная связка: широкий майнинг hard negatives, квадратичный динамический перенос и упор на конфузеры.",
            PipelineConfig { th_hard: 0.35, k_easy: 0.25, gamma_easy: 2.0, conf_weight_l2: 1.3, ..base.clone() }
        ),
    ]
}

pub fn run_theories_sweep<P: AsRef<Path>, Q: AsRef<Path>>(
    features_path: P,
    out_dir: Q,
    epochs_l1: usize,
    epochs_l2: usize,
    device: &Device,
) -> Result<TheoriesBenchmarkReport, Box<dyn std::error::Error>> {
    println!("===============================================================================");
    println!(">>> ЗАПУСК СВИПА ТЕОРИЙ WAKE-WORD (GPU БЕНЧМАРК ГИПОТЕЗ: {} эпох) <<<", epochs_l1);
    println!("===============================================================================");
    let full_ds = Dataset::load_safetensors(features_path)?;
    let (train_ds, val_ds) = full_ds.split_train_val();
    println!("Выборка: train = {} окон, val/holdout = {} окон\n", train_ds.n_samples, val_ds.n_samples);

    let pool = get_theories_pool(epochs_l1, epochs_l2);
    let total = pool.len();

    let mut results = Vec::new();
    let mut best_score = -1e9f64;
    let mut best_theory_id = String::new();
    let mut best_models: Option<(WakeNetL1, WakeNetL2, CascadeMetrics, HoldoutPredictions)> = None;

    for (idx, (id, name, hypothesis, cfg)) in pool.into_iter().enumerate() {
        println!("\n-------------------------------------------------------------------------------");
        println!(">>> [ТЕОРИЯ {}/{}] {} ({})", idx + 1, total, name, id);
        println!("    Гипотеза: {}", hypothesis);
        println!("    Параметры: k_easy = {:.2}, gamma = {:.2}, th_hard = {:.2}, pos_w = {:.1}/{:.1}, conf_w = {:.2}, conf_ratio = {:.2}",
                 cfg.k_easy, cfg.gamma_easy, cfg.th_hard, cfg.pos_weight_l1, cfg.pos_weight_l2, cfg.conf_weight_l2, cfg.conf_cap_ratio);
        println!("-------------------------------------------------------------------------------");

        let (m1, m2, metrics, preds) = train_cascade(&train_ds, &val_ds, &cfg, device);

        // Интегральная оценка: приоритет Multiplicative Recall с жестким штрафом за False Positive Rate
        let score = metrics.multiplicative_recall * 100.0 - (metrics.multiplicative_fpr * 100.0 * 5.0);

        println!("  ИТОГ ТЕОРИИ {}: Multiplicative Recall = {:.2}%, FPR = {:.2}% | Счёт = {:.2}",
                 id, metrics.multiplicative_recall * 100.0, metrics.multiplicative_fpr * 100.0, score);

        if score > best_score {
            best_score = score;
            best_theory_id = id.to_string();
            best_models = Some((m1, m2, metrics.clone(), preds));
        }

        results.push(TheoryResult {
            id: id.to_string(),
            name: name.to_string(),
            hypothesis: hypothesis.to_string(),
            k_easy: cfg.k_easy,
            gamma_easy: cfg.gamma_easy,
            th_hard: cfg.th_hard,
            pos_weight_l1: cfg.pos_weight_l1,
            pos_weight_l2: cfg.pos_weight_l2,
            conf_weight_l2: cfg.conf_weight_l2,
            conf_cap_ratio: cfg.conf_cap_ratio,
            err_repeat: cfg.err_repeat,
            epochs_l1: cfg.epochs_l1,
            epochs_l2: cfg.epochs_l2,
            metrics,
            score,
        });
    }

    // Сохраняем C-заголовки лучшей теории
    if let Some((best_m1, best_m2, best_metrics, best_preds)) = best_models {
        println!("\n===============================================================================");
        println!(">>> ЛУЧШАЯ ТЕОРИЯ: {} (Счёт: {:.2})", best_theory_id, best_score);
        println!("Экспорт C-заголовков лучшей модели в {:?}...", out_dir.as_ref());
        save_headers(&out_dir, &best_m1, &best_m2, best_metrics.th_l1 as f32, best_metrics.th_l2 as f32)?;

        let metrics_path = out_dir.as_ref().join("metrics.json");
        let mut mf = File::create(&metrics_path)?;
        let json_bytes = serde_json::to_vec_pretty(&best_metrics)?;
        mf.write_all(&json_bytes)?;

        let preds_path = out_dir.as_ref().join("predictions.json");
        let mut pf = File::create(&preds_path)?;
        let p_bytes = serde_json::to_vec(&best_preds)?;
        pf.write_all(&p_bytes)?;
        println!("Предсказания лучшей теории записаны в {:?}", preds_path);
    }

    // Сводная таблица Markdown
    let mut md = String::new();
    md.push_str(&format!("# Сводный бенчмарк теорий Wake-Word (GPU, {} эпох)\n\n", epochs_l1));
    md.push_str("| № | ID | Теория | $k_{easy}$ | $\\gamma$ | $th_{hard}$ | $pos_{L1/L2}$ | $w_{conf}$ | L1 AUC | L2 AUC | Hard Rec | Hard FPR | Mult Rec | Mult FPR | Счёт |\n");
    md.push_str("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n");

    for (i, r) in results.iter().enumerate() {
        let is_best = if r.id == best_theory_id { " **(WIN)**" } else { "" };
        md.push_str(&format!(
            "| {} | `{}`{} | {} | {:.2} | {:.1} | {:.2} | {:.1}/{:.1} | {:.1} | {:.4} | {:.4} | {:.2}% | {:.2}% | {:.2}% | {:.2}% | **{:.2}** |\n",
            i + 1,
            r.id,
            is_best,
            r.name,
            r.k_easy,
            r.gamma_easy,
            r.th_hard,
            r.pos_weight_l1,
            r.pos_weight_l2,
            r.conf_weight_l2,
            r.metrics.l1.auc,
            r.metrics.l2.auc,
            r.metrics.hard_cascade_recall * 100.0,
            r.metrics.hard_cascade_fpr * 100.0,
            r.metrics.multiplicative_recall * 100.0,
            r.metrics.multiplicative_fpr * 100.0,
            r.score
        ));
    }

    println!("\n{}", md);

    // Запись файлов отчёта
    let md_path = out_dir.as_ref().join("theories_benchmark.md");
    let mut mdf = File::create(&md_path)?;
    mdf.write_all(md.as_bytes())?;

    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    let report = TheoriesBenchmarkReport {
        timestamp: format!("unix-timestamp: {}", now),
        best_theory_id: best_theory_id.clone(),
        best_score,
        results,
    };

    let json_path = out_dir.as_ref().join("theories_benchmark.json");
    let mut jf = File::create(&json_path)?;
    let j_bytes = serde_json::to_vec_pretty(&report)?;
    jf.write_all(&j_bytes)?;

    println!("Отчёт бенчмарка записан в:\n  {:?}\n  {:?}", md_path, json_path);
    Ok(report)
}
