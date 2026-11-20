use crate::data::Dataset;
use crate::export::save_headers;
use crate::metrics::CascadeMetrics;
use crate::model::{WakeNetL1, WakeNetL2};
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
    pub th_hard: f32,
    pub err_repeat: usize,
    pub err_th_l2: f32,
    pub seed: u64,
    pub neg_cap_l2: Option<usize>,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            epochs_l1: 20,
            epochs_l2: 20,
            batch_size: 64,
            lr_l1: 1.5e-3,
            lr_l2: 1.0e-3,
            pos_weight_l1: 2.5,
            pos_weight_l2: 2.0,
            th_hard: 0.35,
            err_repeat: 3,
            err_th_l2: 0.50,
            seed: 42,
            neg_cap_l2: None,
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
        let wrong_pos = include_pos && (y >= 0.99 && p < 0.50);

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
    th_hard: f32,
    device: &Device,
    batch_size: usize,
    neg_cap: Option<usize>,
) -> Dataset {
    let p1 = predict_l1(model_l1, train_ds, device, batch_size);

    let mut pos_indices = Vec::new();
    let mut conf_indices = Vec::new();
    let mut bg_hard_indices = Vec::new();

    for i in 0..train_ds.n_samples {
        let is_pos = train_ds.y[i] >= 0.99;
        let is_confuser = train_ds.y[i] <= 0.01 && train_ds.y_l1[i] > 0.01;
        let is_bg_hard = train_ds.y[i] <= 0.01 && train_ds.y_l1[i] <= 0.01 && p1[i] >= th_hard;

        if is_pos {
            pos_indices.push(i);
        } else if is_confuser {
            conf_indices.push(i);
        } else if is_bg_hard {
            bg_hard_indices.push(i);
        }
    }

    let n_pos = pos_indices.len();
    let effective_cap = neg_cap.unwrap_or_else(|| n_pos * 3 / 2);
    let cap_conf = ((effective_cap as f32) * 0.70) as usize;
    let cap_hard = effective_cap.saturating_sub(cap_conf).max(1);

    let orig_conf = conf_indices.len();
    let orig_bg_hard = bg_hard_indices.len();

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

    println!("  L2 майнинг hard negatives (порог L1 >= {:.2}):", th_hard);
    println!("    Позитивов: {}, Конфузеров: {}/{} (кап {}), Hard BG: {}/{} (кап {})",
             pos_indices.len(), conf_indices.len(), orig_conf, cap_conf,
             bg_hard_indices.len(), orig_bg_hard, cap_hard);

    let mut keep_indices = pos_indices;
    keep_indices.extend(conf_indices);
    keep_indices.extend(bg_hard_indices);
    keep_indices.sort();

    let mut l2_ds = train_ds.filter_by_indices(&keep_indices);

    // Усиливаем вес найденных hard negatives до 1.0 (вместо фонового 0.05)
    for i in 0..l2_ds.n_samples {
        if l2_ds.y[i] <= 0.01 && l2_ds.y_l1[i] <= 0.01 {
            l2_ds.w_l2[i] = 1.0;
        }
    }

    l2_ds
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
        let (probe_l1, _) = train_l1(model_l1, &train_ds, &val_ds, &l1_cfg, device);
        let train_probs = predict_l1(&probe_l1, &train_ds, device, cfg.batch_size * 2);
        let train_ds_rep = error_repeat(&train_ds, &train_probs, &train_ds.y_l1, cfg.th_hard, cfg.err_repeat, false);
        println!("  L1 перевзвес: окон {} -> {}", train_ds.n_samples, train_ds_rep.n_samples);

        let final_model_l1 = WakeNetL1::new(device);
        train_l1(final_model_l1, &train_ds_rep, &val_ds, &l1_cfg, device)
    } else {
        train_l1(model_l1, &train_ds, &val_ds, &l1_cfg, device)
    };

    println!("  L1 Val ROC-AUC: {:.4}, Recall @ Zero-FP: {:.2}%",
        l1_metrics.auc, l1_metrics.recall_at_zero_fp * 100.0);

    // 2. Отбор Hard Negatives для L2
    println!("\n[Этап 2] Отбор Hard Negatives по порогу L1 (th_hard = {:.2})...", cfg.th_hard);
    let train_l2_ds = mine_hard_negatives(&model_l1, &train_ds, cfg.th_hard, device, cfg.batch_size * 2, cfg.neg_cap_l2);
    println!("  L2 train dataset: {} окон (с hard negatives и конфузерами)", train_l2_ds.n_samples);

    // 3. Обучение L2
    println!("\n[Этап 3] Обучение модели L2 (Xtensa SIMD-16 alignment)...");
    let model_l2 = WakeNetL2::new(device);
    let l2_cfg = TrainConfig {
        epochs: cfg.epochs_l2,
        batch_size: cfg.batch_size,
        lr: cfg.lr_l2,
        pos_weight: cfg.pos_weight_l2,
        seed: cfg.seed + 1,
    };

    let (model_l2, l2_metrics) = if cfg.err_repeat > 1 {
        println!("  Пробный прогон L2 для поиска ошибок...");
        let (probe_l2, _) = train_l2(model_l2, &train_l2_ds, &val_ds, &l2_cfg, device);
        let train_probs = predict_l2(&probe_l2, &train_l2_ds, device, cfg.batch_size * 2);
        let train_l2_rep = error_repeat(&train_l2_ds, &train_probs, &train_l2_ds.y, cfg.err_th_l2, cfg.err_repeat, true);
        println!("  L2 перевзвес: окон {} -> {}", train_l2_ds.n_samples, train_l2_rep.n_samples);

        let final_model_l2 = WakeNetL2::new(device);
        train_l2(final_model_l2, &train_l2_rep, &val_ds, &l2_cfg, device)
    } else {
        train_l2(model_l2, &train_l2_ds, &val_ds, &l2_cfg, device)
    };

    println!("  L2 Val ROC-AUC: {:.4}, Recall @ Zero-FP: {:.2}%",
        l2_metrics.auc, l2_metrics.recall_at_zero_fp * 100.0);

    // 4. Комбинированные метрики каскада на holdout
    println!("\n[Этап 4] Расчёт каскадных метрик на holdout (val)...");
    let val_p1 = predict_l1(&model_l1, &val_ds, device, cfg.batch_size * 2);
    let val_p2 = predict_l2(&model_l2, &val_ds, device, cfg.batch_size * 2);

    let th_l1_nom = 0.40f64;
    let th_l2_nom = 0.80f64;

    let mut hard_hits = 0;
    let mut mult_hits = 0;
    let mut hard_fps = 0;
    let mut mult_fps = 0;
    let mut total_pos = 0;
    let mut total_neg = 0;

    let mult_th = 0.35f64;

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

    // 5. Квантование и экспорт C-заголовков
    println!("\n[Этап 5] Квантование INT8 и экспорт Си-заголовков в {:?}...", out_dir.as_ref());
    save_headers(&out_dir, &model_l1, &model_l2, th_l1_nom as f32, th_l2_nom as f32)?;

    // 6. Запись отчёта метрик
    let metrics_path = out_dir.as_ref().join("metrics.json");
    let mut mf = File::create(&metrics_path)?;
    let json_bytes = serde_json::to_vec_pretty(&cascade_metrics)?;
    mf.write_all(&json_bytes)?;
    println!("Метрики записаны в {:?}", metrics_path);

    println!("=== Конвейер успешно завершён! ===");
    Ok(cascade_metrics)
}
