use safetensors::SafeTensors;
use std::fs::File;
use std::io::Read;
use std::path::Path;

#[derive(Clone, Debug)]
pub struct Dataset {
    pub x_l1: Vec<f32>,
    pub x_l2: Vec<f32>,
    pub y: Vec<f32>,
    pub y_l1: Vec<f32>,
    pub w_l1: Vec<f32>,
    pub w_l2: Vec<f32>,
    pub split: Vec<u8>,
    pub peaks: Vec<f32>,
    pub n_samples: usize,
}

impl Dataset {
    pub fn load_safetensors<P: AsRef<Path>>(path: P) -> Result<Self, Box<dyn std::error::Error>> {
        let mut file = File::open(path)?;
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer)?;

        let st = SafeTensors::deserialize(&buffer)?;

        let x_l1_view = st.tensor("X_l1")?;
        let x_l2_view = st.tensor("X_l2")?;
        let y_view = st.tensor("y")?;
        let y_l1_view = st.tensor("y_l1")?;
        let w_l1_view = st.tensor("w_l1")?;
        let w_l2_view = st.tensor("w_l2")?;
        let split_view = st.tensor("split")?;

        let n_samples = y_view.shape()[0];

        let x_l1: &[f32] = bytemuck::cast_slice(x_l1_view.data());
        let x_l2: &[f32] = bytemuck::cast_slice(x_l2_view.data());
        let y: &[f32] = bytemuck::cast_slice(y_view.data());
        let y_l1: &[f32] = bytemuck::cast_slice(y_l1_view.data());
        let w_l1: &[f32] = bytemuck::cast_slice(w_l1_view.data());
        let w_l2: &[f32] = bytemuck::cast_slice(w_l2_view.data());
        let split: &[u8] = split_view.data();

        let peaks = if let Ok(p_view) = st.tensor("peaks") {
            let p_slice: &[f32] = bytemuck::cast_slice(p_view.data());
            p_slice.to_vec()
        } else {
            vec![1000.0f32; n_samples]
        };

        Ok(Self {
            x_l1: x_l1.to_vec(),
            x_l2: x_l2.to_vec(),
            y: y.to_vec(),
            y_l1: y_l1.to_vec(),
            w_l1: w_l1.to_vec(),
            w_l2: w_l2.to_vec(),
            split: split.to_vec(),
            peaks,
            n_samples,
        })
    }

    pub fn filter_by_indices(&self, indices: &[usize]) -> Self {
        let n = indices.len();
        let mut x_l1 = Vec::with_capacity(n * 74 * 20);
        let mut x_l2 = Vec::with_capacity(n * 148 * 40);
        let mut y = Vec::with_capacity(n);
        let mut y_l1 = Vec::with_capacity(n);
        let mut w_l1 = Vec::with_capacity(n);
        let mut w_l2 = Vec::with_capacity(n);
        let mut split = Vec::with_capacity(n);
        let mut peaks = Vec::with_capacity(n);

        for &i in indices {
            let l1_start = i * 74 * 20;
            let l1_end = l1_start + 74 * 20;
            x_l1.extend_from_slice(&self.x_l1[l1_start..l1_end]);

            let l2_start = i * 148 * 40;
            let l2_end = l2_start + 148 * 40;
            x_l2.extend_from_slice(&self.x_l2[l2_start..l2_end]);

            y.push(self.y[i]);
            y_l1.push(self.y_l1[i]);
            w_l1.push(self.w_l1[i]);
            w_l2.push(self.w_l2[i]);
            split.push(self.split[i]);
            peaks.push(self.peaks[i]);
        }

        Self {
            x_l1,
            x_l2,
            y,
            y_l1,
            w_l1,
            w_l2,
            split,
            peaks,
            n_samples: n,
        }
    }

    pub fn split_train_val(&self) -> (Self, Self) {
        let mut train_idx = Vec::new();
        let mut val_idx = Vec::new();

        for i in 0..self.n_samples {
            if self.split[i] == 0 {
                train_idx.push(i);
            } else {
                val_idx.push(i);
            }
        }

        (self.filter_by_indices(&train_idx), self.filter_by_indices(&val_idx))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load_features() {
        let path = "../../artifacts/cache/features.safetensors";
        if Path::new(path).exists() {
            let ds = Dataset::load_safetensors(path).expect("Failed to load safetensors");
            assert!(ds.n_samples > 0);
            assert_eq!(ds.x_l1.len(), ds.n_samples * 74 * 20);
            assert_eq!(ds.x_l2.len(), ds.n_samples * 148 * 40);
            println!("Loaded dataset: {} samples", ds.n_samples);
            let (train, val) = ds.split_train_val();
            println!("Split: train={}, val={}", train.n_samples, val.n_samples);
            assert!(train.n_samples > 0);
            assert!(val.n_samples > 0);
        }
    }
}
