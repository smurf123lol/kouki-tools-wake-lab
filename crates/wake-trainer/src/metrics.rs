use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ModelMetrics {
    pub auc: f64,
    pub recall_at_zero_fp: f64,
    pub val_loss: f64,
    pub val_accuracy: f64,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct CascadeMetrics {
    pub l1: ModelMetrics,
    pub l2: ModelMetrics,
    pub hard_cascade_recall: f64,
    pub hard_cascade_fpr: f64,
    pub multiplicative_recall: f64,
    pub multiplicative_fpr: f64,
    pub th_l1: f64,
    pub th_l2: f64,
}

/// Расчёт ROC-AUC без внешних библиотек (U-критерий Манна–Уитни с учётом связок)
pub fn auc_score(scores: &[f32], targets: &[f32]) -> Option<f64> {
    let n = scores.len();
    if n == 0 || targets.len() != n {
        return None;
    }

    let mut n_pos = 0usize;
    let mut n_neg = 0usize;
    for &y in targets {
        if y >= 0.99 {
            n_pos += 1;
        } else if y <= 0.01 {
            n_neg += 1;
        }
    }

    if n_pos == 0 || n_neg == 0 {
        return None;
    }

    // Сохраняем пары (индекс, скор, таргет)
    let mut items: Vec<(usize, f32, bool)> = scores
        .iter()
        .zip(targets.iter())
        .enumerate()
        .filter_map(|(idx, (&s, &y))| {
            if y >= 0.99 {
                Some((idx, s, true))
            } else if y <= 0.01 {
                Some((idx, s, false))
            } else {
                None // промежуточные таргеты (например, 0.10 для L1) не входят в чистый бинарный AUC
            }
        })
        .collect();

    // Сортировка по возрастанию скора (стабильная)
    items.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut ranks = vec![0.0f64; items.len()];
    let mut i = 0;
    while i < items.len() {
        let mut j = i;
        while j + 1 < items.len() && (items[j + 1].1 - items[i].1).abs() < 1e-9 {
            j += 1;
        }
        let avg_rank = (i + j) as f64 / 2.0 + 1.0;
        for k in i..=j {
            ranks[k] = avg_rank;
        }
        i = j + 1;
    }

    let mut sum_pos_ranks = 0.0f64;
    for (k, (_, _, is_pos)) in items.iter().enumerate() {
        if *is_pos {
            sum_pos_ranks += ranks[k];
        }
    }

    let u_pos = sum_pos_ranks - (n_pos * (n_pos + 1)) as f64 / 2.0;
    Some(u_pos / (n_pos * n_neg) as f64)
}

/// Расчёт Recall при нулевом уровне ложных тревог (FPR = 0)
pub fn recall_at_zero_fp(scores: &[f32], targets: &[f32]) -> f64 {
    let mut max_neg_score = -1e9f32;
    for (&s, &y) in scores.iter().zip(targets.iter()) {
        if y <= 0.01 && s > max_neg_score {
            max_neg_score = s;
        }
    }

    let mut pos_hits = 0;
    let mut total_pos = 0;
    for (&s, &y) in scores.iter().zip(targets.iter()) {
        if y >= 0.99 {
            total_pos += 1;
            if s > max_neg_score {
                pos_hits += 1;
            }
        }
    }

    if total_pos > 0 {
        pos_hits as f64 / total_pos as f64
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_perfect_auc() {
        let scores = vec![0.1, 0.2, 0.8, 0.9];
        let targets = vec![0.0, 0.0, 1.0, 1.0];
        let auc = auc_score(&scores, &targets).unwrap();
        assert!((auc - 1.0).abs() < 1e-6);

        let rec = recall_at_zero_fp(&scores, &targets);
        assert!((rec - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_inverted_auc() {
        let scores = vec![0.9, 0.8, 0.2, 0.1];
        let targets = vec![0.0, 0.0, 1.0, 1.0];
        let auc = auc_score(&scores, &targets).unwrap();
        assert!((auc - 0.0).abs() < 1e-6);
    }
}
