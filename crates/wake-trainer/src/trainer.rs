use crate::data::Dataset;
use crate::loss::weighted_bce_with_logits;
use crate::metrics::{auc_score, recall_at_zero_fp, ModelMetrics};
use crate::model::{WakeNetL1, WakeNetL2};
use burn::optim::{AdamWConfig, GradientsParams};
use burn::tensor::activation::sigmoid;
use burn::tensor::{Device, Tensor, TensorData};
use rand::seq::SliceRandom;
use rand_chacha::rand_core::SeedableRng;
use rand_chacha::ChaCha8Rng;

pub struct TrainConfig {
    pub epochs: usize,
    pub batch_size: usize,
    pub lr: f64,
    pub pos_weight: f32,
    pub seed: u64,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            epochs: 20,
            batch_size: 64,
            lr: 1.5e-3,
            pos_weight: 2.5,
            seed: 42,
        }
    }
}

pub fn predict_l1(model: &WakeNetL1, ds: &Dataset, device: &Device, batch_size: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(ds.n_samples);
    let n = ds.n_samples;

    for i in (0..n).step_by(batch_size) {
        let end = (i + batch_size).min(n);
        let b = end - i;

        let start_offset = i * 74 * 20;
        let end_offset = end * 74 * 20;
        let slice = &ds.x_l1[start_offset..end_offset];

        let x = Tensor::<3>::from_data(TensorData::new(slice.to_vec(), [b, 74, 20]), device);
        let logits = model.forward(x);
        let probs = sigmoid(logits);
        let vec = probs.into_data().try_to_vec::<f32>().unwrap();
        out.extend_from_slice(&vec);
    }

    out
}

pub fn predict_l2(model: &WakeNetL2, ds: &Dataset, device: &Device, batch_size: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(ds.n_samples);
    let n = ds.n_samples;

    for i in (0..n).step_by(batch_size) {
        let end = (i + batch_size).min(n);
        let b = end - i;

        let start_offset = i * 148 * 40;
        let end_offset = end * 148 * 40;
        let slice = &ds.x_l2[start_offset..end_offset];

        let x = Tensor::<3>::from_data(TensorData::new(slice.to_vec(), [b, 148, 40]), device);
        let logits = model.forward(x);
        let probs = sigmoid(logits);
        let vec = probs.into_data().try_to_vec::<f32>().unwrap();
        out.extend_from_slice(&vec);
    }

    out
}

pub fn train_l1(
    mut model: WakeNetL1,
    train_ds: &Dataset,
    val_ds: &Dataset,
    cfg: &TrainConfig,
    device: &Device,
) -> (WakeNetL1, ModelMetrics) {
    let mut optim = AdamWConfig::new().init();
    let mut rng = ChaCha8Rng::seed_from_u64(cfg.seed);

    let n_train = train_ds.n_samples;
    let mut indices: Vec<usize> = (0..n_train).collect();

    let pi = std::f64::consts::PI;

    for ep in 1..=cfg.epochs {
        indices.shuffle(&mut rng);

        // Cosine annealing learning rate
        let progress = ep as f64 / cfg.epochs as f64;
        let lr_ep = 1e-5 + 0.5 * (cfg.lr - 1e-5) * (1.0 + (pi * progress).cos());

        let mut total_loss = 0.0f32;
        let mut n_batches = 0;

        for chunk in indices.chunks(cfg.batch_size) {
            let b = chunk.len();
            let mut batch_x = Vec::with_capacity(b * 74 * 20);
            let mut batch_y = Vec::with_capacity(b);
            let mut batch_w = Vec::with_capacity(b);

            for &idx in chunk {
                let start = idx * 74 * 20;
                let end = start + 74 * 20;
                batch_x.extend_from_slice(&train_ds.x_l1[start..end]);
                batch_y.push(train_ds.y_l1[idx]);
                batch_w.push(train_ds.w_l1[idx]);
            }

            let x = Tensor::<3>::from_data(TensorData::new(batch_x, [b, 74, 20]), device);
            let y = Tensor::<2>::from_data(TensorData::new(batch_y, [b, 1]), device);
            let w = Tensor::<2>::from_data(TensorData::new(batch_w, [b, 1]), device);

            let logits = model.forward(x);
            let loss = weighted_bce_with_logits(logits, y, w, cfg.pos_weight);

            let loss_val: f32 = loss.clone().into_scalar();
            total_loss += loss_val;
            n_batches += 1;

            let grads = GradientsParams::from_grads(loss.backward(), &model);
            model = optim.step(lr_ep, model, grads);
        }

        if ep % 5 == 0 || ep == cfg.epochs {
            let avg_loss = total_loss / (n_batches as f32).max(1.0);
            println!("  [L1] Epoch {:2}/{} - train loss: {:.4}", ep, cfg.epochs, avg_loss);
        }
    }

    // Evaluate on val
    let val_probs = predict_l1(&model, val_ds, device, cfg.batch_size * 2);
    let auc = auc_score(&val_probs, &val_ds.y).unwrap_or(0.0);
    let r_zero = recall_at_zero_fp(&val_probs, &val_ds.y);

    let metrics = ModelMetrics {
        auc,
        recall_at_zero_fp: r_zero,
        val_loss: 0.0,
        val_accuracy: 0.0,
    };

    (model, metrics)
}

pub fn train_l2(
    mut model: WakeNetL2,
    train_ds: &Dataset,
    val_ds: &Dataset,
    cfg: &TrainConfig,
    device: &Device,
) -> (WakeNetL2, ModelMetrics) {
    let mut optim = AdamWConfig::new().init();
    let mut rng = ChaCha8Rng::seed_from_u64(cfg.seed);

    let n_train = train_ds.n_samples;
    let mut indices: Vec<usize> = (0..n_train).collect();

    let pi = std::f64::consts::PI;

    for ep in 1..=cfg.epochs {
        indices.shuffle(&mut rng);

        // Cosine annealing learning rate
        let progress = ep as f64 / cfg.epochs as f64;
        let lr_ep = 1e-5 + 0.5 * (cfg.lr - 1e-5) * (1.0 + (pi * progress).cos());

        let mut total_loss = 0.0f32;
        let mut n_batches = 0;

        for chunk in indices.chunks(cfg.batch_size) {
            let b = chunk.len();
            let mut batch_x = Vec::with_capacity(b * 148 * 40);
            let mut batch_y = Vec::with_capacity(b);
            let mut batch_w = Vec::with_capacity(b);

            for &idx in chunk {
                let start = idx * 148 * 40;
                let end = start + 148 * 40;
                batch_x.extend_from_slice(&train_ds.x_l2[start..end]);
                batch_y.push(train_ds.y[idx]);
                batch_w.push(train_ds.w_l2[idx]);
            }

            let x = Tensor::<3>::from_data(TensorData::new(batch_x, [b, 148, 40]), device);
            let y = Tensor::<2>::from_data(TensorData::new(batch_y, [b, 1]), device);
            let w = Tensor::<2>::from_data(TensorData::new(batch_w, [b, 1]), device);

            let logits = model.forward(x);
            let loss = weighted_bce_with_logits(logits, y, w, cfg.pos_weight);

            let loss_val: f32 = loss.clone().into_scalar();
            total_loss += loss_val;
            n_batches += 1;

            let grads = GradientsParams::from_grads(loss.backward(), &model);
            model = optim.step(lr_ep, model, grads);
        }

        if ep % 5 == 0 || ep == cfg.epochs {
            let avg_loss = total_loss / (n_batches as f32).max(1.0);
            println!("  [L2] Epoch {:2}/{} - train loss: {:.4}", ep, cfg.epochs, avg_loss);
        }
    }

    // Evaluate on val
    let val_probs = predict_l2(&model, val_ds, device, cfg.batch_size * 2);
    let auc = auc_score(&val_probs, &val_ds.y).unwrap_or(0.0);
    let r_zero = recall_at_zero_fp(&val_probs, &val_ds.y);

    let metrics = ModelMetrics {
        auc,
        recall_at_zero_fp: r_zero,
        val_loss: 0.0,
        val_accuracy: 0.0,
    };

    (model, metrics)
}
