use burn::tensor::activation::relu;
use burn::tensor::Tensor;

/// Взвешенный BCE с логитами (1:1 совпадает с torch.nn.BCEWithLogitsLoss)
///
/// L_i = (1 - y_i) * z_i + (1 + (pos_weight - 1) * y_i) * (max(-z_i, 0) + log(1 + exp(-|z_i|)))
/// Итоговый лосс = mean(L_i * w_i)
pub fn weighted_bce_with_logits(
    logits: Tensor<2>,
    targets: Tensor<2>,
    weights: Tensor<2>,
    pos_weight: f32,
) -> Tensor<1> {
    let neg_logits = logits.clone().neg();
    let max_neg_zero = relu(neg_logits);
    let abs_neg = logits.clone().abs().neg();
    let log1p_exp = abs_neg.exp().add_scalar(1.0).log();

    let term1 = logits.mul(targets.clone().neg().add_scalar(1.0));
    let weight_factor = targets.mul_scalar(pos_weight - 1.0).add_scalar(1.0);
    let term2 = weight_factor.mul(max_neg_zero.add(log1p_exp));

    let loss_per_elem = term1.add(term2).mul(weights);
    loss_per_elem.mean()
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Device, DeviceKind, TensorData};

    #[test]
    fn test_bce_numerical_values() {
        let device = Device::wgpu(DeviceKind::DiscreteGpu(0)).autodiff();

        // 1. target = 1.0, logit = 0.0, pos_weight = 2.5
        let logits = Tensor::<2>::from_data(TensorData::new(vec![0.0f32], [1, 1]), &device);
        let targets = Tensor::<2>::from_data(TensorData::new(vec![1.0f32], [1, 1]), &device);
        let weights = Tensor::<2>::from_data(TensorData::new(vec![1.0f32], [1, 1]), &device);

        let loss = weighted_bce_with_logits(logits, targets, weights, 2.5);
        let val: f32 = loss.into_scalar();
        let expected = 2.5 * 2.0f32.ln();
        assert!((val - expected).abs() < 1e-4, "Expected {}, got {}", expected, val);

        // 2. target = 0.0, logit = 2.0, pos_weight = 2.5
        let logits = Tensor::<2>::from_data(TensorData::new(vec![2.0f32], [1, 1]), &device);
        let targets = Tensor::<2>::from_data(TensorData::new(vec![0.0f32], [1, 1]), &device);
        let weights = Tensor::<2>::from_data(TensorData::new(vec![1.0f32], [1, 1]), &device);

        let loss = weighted_bce_with_logits(logits, targets, weights, 2.5);
        let val: f32 = loss.into_scalar();
        let expected = 2.0 + (1.0 + (-2.0f32).exp()).ln();
        assert!((val - expected).abs() < 1e-4, "Expected {}, got {}", expected, val);
        println!("Weighted BCE numerical checks PASSED on GPU!");
    }
}
