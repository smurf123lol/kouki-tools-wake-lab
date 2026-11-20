use burn::module::Module;
use burn::nn::conv::Conv2dConfig;
use burn::nn::{BatchNormConfig, Linear, LinearConfig, PaddingConfig2d};
use burn::tensor::activation::relu;
use burn::tensor::{Device, Tensor, TensorData};

#[derive(Module, Debug)]
pub struct ConvBnRelu {
    pub conv: burn::nn::conv::Conv2d,
    pub bn: burn::nn::BatchNorm,
}

impl ConvBnRelu {
    pub fn new(
        in_c: usize,
        out_c: usize,
        k: [usize; 2],
        s: [usize; 2],
        p: [usize; 2],
        groups: usize,
        bias: bool,
        device: &Device,
    ) -> Self {
        let conv = Conv2dConfig::new([in_c, out_c], k)
            .with_stride(s)
            .with_padding(PaddingConfig2d::Explicit(p[0], p[0], p[1], p[1]))
            .with_groups(groups)
            .with_bias(bias)
            .init(device);
        let bn = BatchNormConfig::new(out_c).init(device);
        Self { conv, bn }
    }

    pub fn forward(&self, x: Tensor<4>) -> Tensor<4> {
        let x = self.conv.forward(x);
        let x = self.bn.forward(x);
        relu(x)
    }
}

#[derive(Module, Debug)]
pub struct DwPwBnRelu {
    pub dw: burn::nn::conv::Conv2d,
    pub pw: burn::nn::conv::Conv2d,
    pub bn: burn::nn::BatchNorm,
}

impl DwPwBnRelu {
    pub fn new(
        in_c: usize,
        dw_c: usize,
        pw_c: usize,
        k: [usize; 2],
        s: [usize; 2],
        p: [usize; 2],
        device: &Device,
    ) -> Self {
        let dw = Conv2dConfig::new([in_c, dw_c], k)
            .with_stride(s)
            .with_padding(PaddingConfig2d::Explicit(p[0], p[0], p[1], p[1]))
            .with_groups(dw_c)
            .with_bias(false)
            .init(device);
        let pw = Conv2dConfig::new([dw_c, pw_c], [1, 1])
            .with_bias(true)
            .init(device);
        let bn = BatchNormConfig::new(pw_c).init(device);
        Self { dw, pw, bn }
    }

    pub fn forward(&self, x: Tensor<4>) -> Tensor<4> {
        let x = self.dw.forward(x);
        let x = self.pw.forward(x);
        let x = self.bn.forward(x);
        relu(x)
    }
}

/// Создание матрицы проекции адаптивного пулинга [T, 4]
pub fn create_adaptive_pool_matrix(t_in: usize, device: &Device) -> Tensor<2> {
    let mut matrix = vec![0.0f32; t_in * 4];
    for i in 0..4 {
        let a = (i * t_in) / 4;
        let b = ((i + 1) * t_in + 4 - 1) / 4;
        let count = (b - a) as f32;
        for t in a..b {
            matrix[t * 4 + i] = 1.0 / count;
        }
    }
    Tensor::from_data(TensorData::new(matrix, [t_in, 4]), device)
}

#[derive(Module, Debug)]
pub struct WakeNetL1 {
    pub b0: ConvBnRelu,
    pub b1: DwPwBnRelu,
    pub b2: DwPwBnRelu,
    pub pool_matrix: Tensor<2>,
    pub fc1: Linear,
    pub fc2: Linear,
}

impl WakeNetL1 {
    pub fn new(device: &Device) -> Self {
        let b0 = ConvBnRelu::new(1, 8, [3, 3], [2, 2], [1, 1], 1, true, device);
        let b1 = DwPwBnRelu::new(8, 8, 16, [3, 3], [2, 2], [1, 1], device);
        let b2 = DwPwBnRelu::new(16, 16, 32, [3, 3], [2, 2], [1, 1], device);

        let pool_matrix = create_adaptive_pool_matrix(10, device);
        let fc1 = LinearConfig::new(32 * 4, 24).init(device);
        let fc2 = LinearConfig::new(24, 1).init(device);

        Self {
            b0,
            b1,
            b2,
            pool_matrix,
            fc1,
            fc2,
        }
    }

    pub fn forward(&self, x: Tensor<3>) -> Tensor<2> {
        // x: [B, 74, 20] -> unsqueeze(1) -> [B, 1, 74, 20]
        let x = x.unsqueeze_dim(1);
        let x = self.b0.forward(x);
        let x = self.b1.forward(x);
        let x = self.b2.forward(x); // [B, 32, 10, 3]

        // Mean across mel bins (dim 3) -> [B, 32, 10, 1] -> squeeze to [B, 32, 10]
        let t = x.mean_dim(3).squeeze_dim::<3>(3);

        // Adaptive avg pool 10 -> 4: t [B, 32, 10] * pool_matrix [10, 4] -> [B, 32, 4]
        let pool = t.matmul(self.pool_matrix.clone().unsqueeze_dim(0));

        // Flatten to [B, 128]
        let [b, c, p] = pool.dims();
        let flat = pool.reshape([b, c * p]);

        let h = relu(self.fc1.forward(flat));
        self.fc2.forward(h)
    }
}

#[derive(Module, Debug)]
pub struct WakeNetL2 {
    pub b0: ConvBnRelu,
    pub b1: DwPwBnRelu,
    pub b2: DwPwBnRelu,
    pub pool_matrix: Tensor<2>,
    pub fc1: Linear,
    pub fc2: Linear,
}

impl WakeNetL2 {
    pub fn new(device: &Device) -> Self {
        // T130: Проверка SIMD-выравнивания архитектуры L2 (каналы кратны 16 под Xtensa PIE 128-bit)
        assert_eq!(16 % 16, 0, "L2 SIMD assert: b0 oc=16");
        assert_eq!(32 % 16, 0, "L2 SIMD assert: b1 pw=32");
        assert_eq!(64 % 16, 0, "L2 SIMD assert: b2 pw=64");
        assert_eq!(32 % 16, 0, "L2 SIMD assert: fc1=32");

        let b0 = ConvBnRelu::new(1, 16, [5, 5], [2, 2], [2, 2], 1, true, device);
        let b1 = DwPwBnRelu::new(16, 16, 32, [3, 3], [2, 2], [1, 1], device);
        let b2 = DwPwBnRelu::new(32, 32, 64, [3, 3], [2, 2], [1, 1], device);

        let pool_matrix = create_adaptive_pool_matrix(19, device);
        let fc1 = LinearConfig::new(64 * 4, 32).init(device);
        let fc2 = LinearConfig::new(32, 1).init(device);

        Self {
            b0,
            b1,
            b2,
            pool_matrix,
            fc1,
            fc2,
        }
    }

    pub fn forward(&self, x: Tensor<3>) -> Tensor<2> {
        // x: [B, 148, 40] -> unsqueeze(1) -> [B, 1, 148, 40]
        let x = x.unsqueeze_dim(1);
        let x = self.b0.forward(x);
        let x = self.b1.forward(x);
        let x = self.b2.forward(x); // [B, 64, 19, 5]

        // Mean across mel bins (dim 3) -> [B, 64, 19, 1] -> squeeze to [B, 64, 19]
        let t = x.mean_dim(3).squeeze_dim::<3>(3);

        // Adaptive avg pool 19 -> 4: t [B, 64, 19] * pool_matrix [19, 4] -> [B, 64, 4]
        let pool = t.matmul(self.pool_matrix.clone().unsqueeze_dim(0));

        // Flatten to [B, 256]
        let [b, c, p] = pool.dims();
        let flat = pool.reshape([b, c * p]);

        let h = relu(self.fc1.forward(flat));
        self.fc2.forward(h)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::DeviceKind;

    #[test]
    fn test_wakenet_l1_shapes() {
        let device = Device::wgpu(DeviceKind::DiscreteGpu(0)).autodiff();
        let model = WakeNetL1::new(&device);

        // Input [4, 74, 20]
        let x = Tensor::<3>::zeros([4, 74, 20], &device);
        let logits = model.forward(x);
        assert_eq!(logits.dims(), [4, 1]);
        println!("WakeNetL1 forward successful on GPU!");
    }

    #[test]
    fn test_wakenet_l2_shapes() {
        let device = Device::wgpu(DeviceKind::DiscreteGpu(0)).autodiff();
        let model = WakeNetL2::new(&device);

        // Input [4, 148, 40]
        let x = Tensor::<3>::zeros([4, 148, 40], &device);
        let logits = model.forward(x);
        assert_eq!(logits.dims(), [4, 1]);
        println!("WakeNetL2 forward successful on GPU!");
    }
}
