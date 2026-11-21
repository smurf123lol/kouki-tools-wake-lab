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
            epochs: 60,
            batch_size: 64,
            lr: 1.5e-3,
            pos_weight: 2.5,
            seed: 42,
        }
    }
}

macro_rules! impl_wake_predictor {
    ($fn_name:ident, $model_ty:ident, $feat_field:ident, $dim_t:expr, $dim_f:expr) => {
        pub fn $fn_name(model: &$model_ty, ds: &Dataset, device: &Device, batch_size: usize) -> Vec<f32> {
            let feat_len = $dim_t * $dim_f;
            let mut out = Vec::with_capacity(ds.n_samples);
            let n = ds.n_samples;

            for i in (0..n).step_by(batch_size) {
                let end = (i + batch_size).min(n);
                let b = end - i;

                let start_offset = i * feat_len;
                let end_offset = end * feat_len;
                let slice = &ds.$feat_field[start_offset..end_offset];

                let x = Tensor::<3>::from_data(TensorData::new(slice.to_vec(), [b, $dim_t, $dim_f]), device);
                let logits = model.forward(x);
                let probs = sigmoid(logits);
                let vec = probs.into_data().try_to_vec::<f32>().unwrap();
                out.extend_from_slice(&vec);
            }

            out
        }
    };
}

macro_rules! impl_wake_trainer {
    ($fn_name:ident, $pred_fn:ident, $model_ty:ident, $feat_field:ident, $label_field:ident, $weight_field:ident, $dim_t:expr, $dim_f:expr, $tag:expr) => {
        pub fn $fn_name(
            mut model: $model_ty,
            train_ds: &Dataset,
            val_ds: &Dataset,
            cfg: &TrainConfig,
            device: &Device,
        ) -> ($model_ty, ModelMetrics) {
            let feat_len = $dim_t * $dim_f;
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
                    let mut batch_x = Vec::with_capacity(b * feat_len);
                    let mut batch_y = Vec::with_capacity(b);
                    let mut batch_w = Vec::with_capacity(b);

                    for &idx in chunk {
                        let start = idx * feat_len;
                        let end = start + feat_len;
                        batch_x.extend_from_slice(&train_ds.$feat_field[start..end]);
                        batch_y.push(train_ds.$label_field[idx]);
                        batch_w.push(train_ds.$weight_field[idx]);
                    }

                    let x = Tensor::<3>::from_data(TensorData::new(batch_x, [b, $dim_t, $dim_f]), device);
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
                    println!("  [{}] Epoch {:2}/{} - train loss: {:.4}", $tag, ep, cfg.epochs, avg_loss);
                }
            }

            // Evaluate on val
            let val_probs = $pred_fn(&model, val_ds, device, cfg.batch_size * 2);
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
    };
}

impl_wake_predictor!(predict_l1, WakeNetL1, x_l1, 74, 20);
impl_wake_predictor!(predict_l2, WakeNetL2, x_l2, 148, 40);

impl_wake_trainer!(train_l1, predict_l1, WakeNetL1, x_l1, y_l1, w_l1, 74, 20, "L1");
impl_wake_trainer!(train_l2, predict_l2, WakeNetL2, x_l2, y, w_l2, 148, 40, "L2");
