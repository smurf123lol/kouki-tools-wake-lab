use crate::model::{ConvBnRelu, DwPwBnRelu, WakeNetL1, WakeNetL2};
use std::fs::File;
use std::io::Write;
use std::path::Path;

#[derive(Clone, Debug)]
pub struct LayerExport {
    pub kind: String, // "conv", "dwpw", "linear"
    pub w: Vec<i8>,
    pub w_scale: f32,
    pub b: Vec<f32>,
    pub w_dw: Vec<i8>,
    pub w_dw_scale: f32,
    pub w_pw: Vec<i8>,
    pub w_pw_scale: f32,
}

/// Квантование весов в симметричный INT8
pub fn quantize_int8(weights: &[f32]) -> (Vec<i8>, f32) {
    let mut max_val = 0.0f32;
    for &w in weights {
        let abs = w.abs();
        if abs > max_val {
            max_val = abs;
        }
    }
    max_val += 1e-12;
    let scale = max_val / 127.0;
    let mut q = Vec::with_capacity(weights.len());
    for &w in weights {
        let val = (w / scale).round();
        let clamped = val.clamp(-127.0, 127.0) as i8;
        q.push(clamped);
    }
    (q, scale)
}

/// Слияние BatchNorm в Conv2D (в double f64 для максимальной числовой точности)
pub fn fuse_bn_conv(
    conv_w: &[f32],
    conv_b: Option<&[f32]>,
    bn_gamma: &[f32],
    bn_beta: &[f32],
    bn_mean: &[f32],
    bn_var: &[f32],
    out_c: usize,
) -> (Vec<f32>, Vec<f32>) {
    let elem_per_channel = conv_w.len() / out_c;
    let mut fused_w = vec![0.0f32; conv_w.len()];
    let mut fused_b = vec![0.0f32; out_c];

    for c in 0..out_c {
        let gamma = bn_gamma[c] as f64;
        let beta = bn_beta[c] as f64;
        let mean = bn_mean[c] as f64;
        let var = bn_var[c] as f64;
        let scale = gamma / (var + 1e-5).sqrt();
        let bias = conv_b.map(|b| b[c] as f64).unwrap_or(0.0);

        fused_b[c] = ((bias - mean) * scale + beta) as f32;

        let start = c * elem_per_channel;
        let end = start + elem_per_channel;
        for i in start..end {
            fused_w[i] = (conv_w[i] as f64 * scale) as f32;
        }
    }

    (fused_w, fused_b)
}

pub fn export_conv_bn(block: &ConvBnRelu, out_c: usize) -> LayerExport {
    let conv_w = block.conv.weight.val().into_data().try_to_vec::<f32>().unwrap();
    let conv_b = block.conv.bias.as_ref().map(|b| b.val().into_data().try_to_vec::<f32>().unwrap());
    let bn_gamma = block.bn.gamma.val().into_data().try_to_vec::<f32>().unwrap();
    let bn_beta = block.bn.beta.val().into_data().try_to_vec::<f32>().unwrap();
    let bn_mean = block.bn.running_mean.value_sync().into_data().try_to_vec::<f32>().unwrap();
    let bn_var = block.bn.running_var.value_sync().into_data().try_to_vec::<f32>().unwrap();

    let (fused_w, fused_b) = fuse_bn_conv(
        &conv_w,
        conv_b.as_deref(),
        &bn_gamma,
        &bn_beta,
        &bn_mean,
        &bn_var,
        out_c,
    );

    let (q, s) = quantize_int8(&fused_w);
    LayerExport {
        kind: "conv".to_string(),
        w: q,
        w_scale: s,
        b: fused_b,
        w_dw: Vec::new(),
        w_dw_scale: 0.0,
        w_pw: Vec::new(),
        w_pw_scale: 0.0,
    }
}

pub fn export_dwpw_bn(block: &DwPwBnRelu, pw_c: usize) -> LayerExport {
    let dw_w = block.dw.weight.val().into_data().try_to_vec::<f32>().unwrap();
    let (q_dw, s_dw) = quantize_int8(&dw_w);

    let pw_w = block.pw.weight.val().into_data().try_to_vec::<f32>().unwrap();
    let pw_b = block.pw.bias.as_ref().map(|b| b.val().into_data().try_to_vec::<f32>().unwrap());
    let bn_gamma = block.bn.gamma.val().into_data().try_to_vec::<f32>().unwrap();
    let bn_beta = block.bn.beta.val().into_data().try_to_vec::<f32>().unwrap();
    let bn_mean = block.bn.running_mean.value_sync().into_data().try_to_vec::<f32>().unwrap();
    let bn_var = block.bn.running_var.value_sync().into_data().try_to_vec::<f32>().unwrap();

    let (fused_pw_w, fused_pw_b) = fuse_bn_conv(
        &pw_w,
        pw_b.as_deref(),
        &bn_gamma,
        &bn_beta,
        &bn_mean,
        &bn_var,
        pw_c,
    );

    let (q_pw, s_pw) = quantize_int8(&fused_pw_w);
    LayerExport {
        kind: "dwpw".to_string(),
        w: Vec::new(),
        w_scale: 0.0,
        b: fused_pw_b,
        w_dw: q_dw,
        w_dw_scale: s_dw,
        w_pw: q_pw,
        w_pw_scale: s_pw,
    }
}

pub fn export_linear(lin: &burn::nn::Linear) -> LayerExport {
    let w = lin.weight.val().into_data().try_to_vec::<f32>().unwrap();
    let b = lin.bias.as_ref().unwrap().val().into_data().try_to_vec::<f32>().unwrap();
    let (q, s) = quantize_int8(&w);

    LayerExport {
        kind: "linear".to_string(),
        w: q,
        w_scale: s,
        b,
        w_dw: Vec::new(),
        w_dw_scale: 0.0,
        w_pw: Vec::new(),
        w_pw_scale: 0.0,
    }
}

fn c_array_i8(name: &str, arr: &[i8], per_line: usize) -> String {
    let mut out = format!("static const int8_t {}[{}] = {{\n    ", name, arr.len());
    for (i, &v) in arr.iter().enumerate() {
        out.push_str(&format!("{:5}", v));
        if i < arr.len() - 1 {
            out.push_str(", ");
            if (i + 1) % per_line == 0 {
                out.push_str("\n    ");
            }
        }
    }
    out.push_str("\n};\n\n");
    out
}

fn c_array_f32(name: &str, arr: &[f32], per_line: usize) -> String {
    let mut out = format!("static const float {}[{}] = {{\n    ", name, arr.len());
    for (i, &v) in arr.iter().enumerate() {
        out.push_str(&format!("{:17.8e}f", v));
        if i < arr.len() - 1 {
            out.push_str(", ");
            if (i + 1) % per_line == 0 {
                out.push_str("\n    ");
            }
        }
    }
    out.push_str("\n};\n\n");
    out
}

pub fn generate_c_header_l1(model: &WakeNetL1) -> String {
    let l0 = export_conv_bn(&model.b0, 8);
    let l1 = export_dwpw_bn(&model.b1, 16);
    let l2 = export_dwpw_bn(&model.b2, 32);
    let l3 = export_linear(&model.fc1);
    let l4 = export_linear(&model.fc2);

    let layers = vec![l0, l1, l2, l3, l4];
    let total_len: usize = layers.iter().map(|l| {
        if l.kind == "conv" || l.kind == "linear" {
            l.w.len()
        } else {
            l.w_dw.len() + l.w_pw.len()
        }
    }).sum();
    let total_params: usize = total_len + layers.iter().map(|l| l.b.len()).sum::<usize>();

    let mut s = String::new();
    s.push_str(r#"/*
 * wake_model_ey_gin_l1.h — L1-уровень детектора «эй гин» (T112/T163 Burn WGPU)
 *
 * Сгенерирован `wake-trainer` на Burn/WGPU, руками не правится: переобучение
 * перезаписывает файл целиком. Веса — INT8 (по тензору на слой), смещения и
 * активации — float32, BatchNorm слит в свёртки.
 *
 * Вход: нормированная спектрограмма (74 кадров x 20 mel-бинов,
 * (x-mean)/std по окну), упакованная построчно. Выход: вероятность 0..1.
 */

#pragma once

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
"#);

    s.push_str(&format!("#define WAKE_L1_MODEL_LEN        {}\n", total_len));
    s.push_str(&format!("#define WAKE_L1_MODEL_PARAMS     {}\n", total_params));
    s.push_str("#define WAKE_L1_MODEL_FRAMES     74\n");
    s.push_str("#define WAKE_L1_MEL_BINS         20\n");
    s.push_str("#define WAKE_L1_ARENA_FLOATS     4480\n");
    s.push_str("#define WAKE_L1_BUF_A_FLOATS     2960\n");
    s.push_str("#define WAKE_L1_BUF_B_FLOATS     1520\n\n");

    // Arrays
    s.push_str(&c_array_i8("g_l1_w0", &layers[0].w, 16));
    s.push_str(&format!("static const float g_l1_s0 = {:.9e}f;\n\n", layers[0].w_scale));
    s.push_str(&c_array_f32("g_l1_b0", &layers[0].b, 16));

    s.push_str(&c_array_i8("g_l1_w1_dw", &layers[1].w_dw, 16));
    s.push_str(&format!("static const float g_l1_s1_dw = {:.9e}f;\n\n", layers[1].w_dw_scale));
    s.push_str(&c_array_i8("g_l1_w1_pw", &layers[1].w_pw, 16));
    s.push_str(&format!("static const float g_l1_s1_pw = {:.9e}f;\n\n", layers[1].w_pw_scale));
    s.push_str(&c_array_f32("g_l1_b1", &layers[1].b, 16));

    s.push_str(&c_array_i8("g_l1_w2_dw", &layers[2].w_dw, 16));
    s.push_str(&format!("static const float g_l1_s2_dw = {:.9e}f;\n\n", layers[2].w_dw_scale));
    s.push_str(&c_array_i8("g_l1_w2_pw", &layers[2].w_pw, 16));
    s.push_str(&format!("static const float g_l1_s2_pw = {:.9e}f;\n\n", layers[2].w_pw_scale));
    s.push_str(&c_array_f32("g_l1_b2", &layers[2].b, 16));

    s.push_str(&c_array_i8("g_l1_w3", &layers[3].w, 16));
    s.push_str(&format!("static const float g_l1_s3 = {:.9e}f;\n\n", layers[3].w_scale));
    s.push_str(&c_array_f32("g_l1_b3", &layers[3].b, 16));

    s.push_str(&c_array_i8("g_l1_w4", &layers[4].w, 16));
    s.push_str(&format!("static const float g_l1_s4 = {:.9e}f;\n\n", layers[4].w_scale));
    s.push_str(&c_array_f32("g_l1_b4", &layers[4].b, 16));

    s.push_str(r#"static inline float wake_l1_infer(const float *spec_normalized, float *arena)
{
    float *dyn_arena = NULL;
    if (!arena) {
        dyn_arena = (float *)malloc(WAKE_L1_ARENA_FLOATS * sizeof(float));
        if (!dyn_arena) return 0.0f;
        arena = dyn_arena;
    }

    float *buf_a = arena;
    float *buf_b = arena + WAKE_L1_BUF_A_FLOATS;
    const float *in = spec_normalized;

    /* Блок 0: Conv2D 3x3, выход 37x10x8 + ReLU */
    for (int oc = 0; oc < 8; oc++) {
        for (int oh = 0; oh < 37; oh++) {
            for (int ow = 0; ow < 10; ow++) {
                float sum = g_l1_b0[oc];
                for (int kh = 0; kh < 3; kh++) {
                    int ih = oh * 2 - 1 + kh;
                    if (ih < 0 || ih >= 74) continue;
                    for (int kw = 0; kw < 3; kw++) {
                        int iw = ow * 2 - 1 + kw;
                        if (iw < 0 || iw >= 20) continue;
                        int wi = ((oc * 1 + 0) * 3 + kh) * 3 + kw;
                        sum += in[ih * 20 + iw] * (g_l1_w0[wi] * g_l1_s0);
                    }
                }
                buf_a[(oh * 10 + ow) * 8 + oc] = (sum > 0.0f) ? sum : 0.0f;
            }
        }
    }

    /* Блок 1: depthwise 3x3 + pointwise -> 16, выход 19x5x16 + ReLU */
    for (int oh = 0; oh < 19; oh++) {
        for (int ow = 0; ow < 5; ow++) {
            float dw_out[8];
            for (int c = 0; c < 8; c++) {
                float sum = 0.0f;
                for (int kh = 0; kh < 3; kh++) {
                    int ih = oh * 2 - 1 + kh;
                    if (ih < 0 || ih >= 37) continue;
                    for (int kw = 0; kw < 3; kw++) {
                        int iw = ow * 2 - 1 + kw;
                        if (iw < 0 || iw >= 10) continue;
                        int wi = (c * 3 + kh) * 3 + kw;
                        sum += buf_a[(ih * 10 + iw) * 8 + c] * (g_l1_w1_dw[wi] * g_l1_s1_dw);
                    }
                }
                dw_out[c] = sum;
            }
            for (int oc = 0; oc < 16; oc++) {
                float sum = g_l1_b1[oc];
                for (int ic = 0; ic < 8; ic++) {
                    sum += dw_out[ic] * (g_l1_w1_pw[oc * 8 + ic] * g_l1_s1_pw);
                }
                buf_b[(oh * 5 + ow) * 16 + oc] = (sum > 0.0f) ? sum : 0.0f;
            }
        }
    }

    /* Блок 2: depthwise 3x3 + pointwise -> 32, выход 10x3x32 + ReLU */
    for (int oh = 0; oh < 10; oh++) {
        for (int ow = 0; ow < 3; ow++) {
            float dw_out[16];
            for (int c = 0; c < 16; c++) {
                float sum = 0.0f;
                for (int kh = 0; kh < 3; kh++) {
                    int ih = oh * 2 - 1 + kh;
                    if (ih < 0 || ih >= 19) continue;
                    for (int kw = 0; kw < 3; kw++) {
                        int iw = ow * 2 - 1 + kw;
                        if (iw < 0 || iw >= 5) continue;
                        int wi = (c * 3 + kh) * 3 + kw;
                        sum += buf_b[(ih * 5 + iw) * 16 + c] * (g_l1_w2_dw[wi] * g_l1_s2_dw);
                    }
                }
                dw_out[c] = sum;
            }
            for (int oc = 0; oc < 32; oc++) {
                float sum = g_l1_b2[oc];
                for (int ic = 0; ic < 16; ic++) {
                    sum += dw_out[ic] * (g_l1_w2_pw[oc * 16 + ic] * g_l1_s2_pw);
                }
                buf_a[(oh * 3 + ow) * 32 + oc] = (sum > 0.0f) ? sum : 0.0f;
            }
        }
    }

    /* Temporal pooling: среднее по mel-бинам -> 10x32, затем adaptive avg pool 10 -> 4 */
    float *t_feats = buf_b;
    float *c_in = buf_b + (10 * 32);
    for (int t = 0; t < 10; t++) {
        for (int c = 0; c < 32; c++) {
            float msum = 0.0f;
            for (int w = 0; w < 3; w++) {
                msum += buf_a[(t * 3 + w) * 32 + c];
            }
            t_feats[t * 32 + c] = msum / (float)3;
        }
    }

    for (int c = 0; c < 32; c++) {
        c_in[c * 4 + 0] = (t_feats[0 * 32 + c] + t_feats[1 * 32 + c] + t_feats[2 * 32 + c]) / (float)3;
        c_in[c * 4 + 1] = (t_feats[2 * 32 + c] + t_feats[3 * 32 + c] + t_feats[4 * 32 + c]) / (float)3;
        c_in[c * 4 + 2] = (t_feats[5 * 32 + c] + t_feats[6 * 32 + c] + t_feats[7 * 32 + c]) / (float)3;
        c_in[c * 4 + 3] = (t_feats[7 * 32 + c] + t_feats[8 * 32 + c] + t_feats[9 * 32 + c]) / (float)3;
    }

    /* FC1: 128 -> 24 + ReLU */
    float fc1[24];
    for (int i = 0; i < 24; i++) {
        float sum = g_l1_b3[i];
        for (int j = 0; j < 128; j++) {
            sum += c_in[j] * (g_l1_w3[i * 128 + j] * g_l1_s3);
        }
        fc1[i] = (sum > 0.0f) ? sum : 0.0f;
    }

    /* FC2: 24 -> 1 + Sigmoid */
    float logit = g_l1_b4[0];
    for (int i = 0; i < 24; i++) {
        logit += fc1[i] * (g_l1_w4[i] * g_l1_s4);
    }

    if (dyn_arena) free(dyn_arena);
    return 1.0f / (1.0f + expf(-logit));
}
"#);

    s
}

pub fn generate_c_header_l2(model: &WakeNetL2, th_l1: f32, th_l2: f32) -> String {
    let l0 = export_conv_bn(&model.b0, 16);
    let l1 = export_dwpw_bn(&model.b1, 32);
    let l2 = export_dwpw_bn(&model.b2, 64);
    let l3 = export_linear(&model.fc1);
    let l4 = export_linear(&model.fc2);

    let layers = vec![l0, l1, l2, l3, l4];
    let total_len: usize = layers.iter().map(|l| {
        if l.kind == "conv" || l.kind == "linear" {
            l.w.len()
        } else {
            l.w_dw.len() + l.w_pw.len()
        }
    }).sum();
    let total_params: usize = total_len + layers.iter().map(|l| l.b.len()).sum::<usize>();

    let mut s = String::new();
    s.push_str(r#"/*
 * wake_model_ey_gin_l2.h — L2-уровень детектора «эй гин» (T112/T163 Burn WGPU)
 *
 * Сгенерирован `wake-trainer` на Burn/WGPU, руками не правится: переобучение
 * перезаписывает файл целиком. Веса — INT8 (по тензору на слой), смещения и
 * активации — float32, BatchNorm слит в свёртки.
 *
 * Вход: нормированная спектрограмма (148 кадров x 40 mel-бинов,
 * (x-mean)/std по окну), упакованная построчно. Выход: вероятность 0..1.
 */

#pragma once

#include <math.h>
#include <stdint.h>
#include <stdlib.h>

#ifndef WAKE_DECISION_MODE_T_DEFINED
#define WAKE_DECISION_MODE_T_DEFINED
typedef enum {
    WAKE_DECISION_HARD_CASCADE   = 0,
    WAKE_DECISION_ADDITIVE       = 1,
    WAKE_DECISION_MULTIPLICATIVE = 2,
    WAKE_DECISION_MARGIN_RATIO   = 3,
    WAKE_DECISION_DYNAMIC        = 4,
    WAKE_DECISION_LOGIT_BONUS    = 5
} wake_decision_mode_t;
#endif

#ifndef WAKE_CONFIG_DECISION_MODE
#define WAKE_CONFIG_DECISION_MODE  WAKE_DECISION_HARD_CASCADE
#endif
"#);

    s.push_str(&format!("#ifndef WAKE_CONFIG_TH_L1_INVOKE\n#define WAKE_CONFIG_TH_L1_INVOKE   {:.2}f\n#endif\n", th_l1));
    s.push_str(&format!("#ifndef WAKE_CONFIG_TH_L1_NOM\n#define WAKE_CONFIG_TH_L1_NOM      {:.2}f\n#endif\n", th_l1));
    s.push_str("#ifndef WAKE_CONFIG_TH_L2_FLOOR\n#define WAKE_CONFIG_TH_L2_FLOOR    0.65f\n#endif\n");
    s.push_str(&format!("#ifndef WAKE_CONFIG_TH_L2_NOM\n#define WAKE_CONFIG_TH_L2_NOM      {:.2}f\n#endif\n", th_l2));
    s.push_str("#ifndef WAKE_CONFIG_COMBO_THRESHOLD\n#define WAKE_CONFIG_COMBO_THRESHOLD 0.00f\n#endif\n");
    s.push_str("#ifndef WAKE_CONFIG_DYNAMIC_K\n#define WAKE_CONFIG_DYNAMIC_K      2.0f\n#endif\n");
    s.push_str("#ifndef WAKE_CONFIG_DYNAMIC_MIN\n#define WAKE_CONFIG_DYNAMIC_MIN    0.70f\n#endif\n");
    s.push_str("#ifndef WAKE_CONFIG_DYNAMIC_MAX\n#define WAKE_CONFIG_DYNAMIC_MAX    0.92f\n#endif\n");
    s.push_str("#ifndef WAKE_CONFIG_TAU1\n#define WAKE_CONFIG_TAU1           0.00f\n#endif\n");
    s.push_str("#ifndef WAKE_CONFIG_TAU2\n#define WAKE_CONFIG_TAU2           1.39f\n#endif\n");
    s.push_str("#ifndef WAKE_CONFIG_ALPHA\n#define WAKE_CONFIG_ALPHA          0.40f\n#endif\n");

    s.push_str(&format!("#define WAKE_L2_MODEL_LEN        {}\n", total_len));
    s.push_str(&format!("#define WAKE_L2_MODEL_PARAMS     {}\n", total_params));
    s.push_str("#define WAKE_L2_MODEL_FRAMES     148\n");
    s.push_str("#define WAKE_L2_MEL_BINS         40\n");
    s.push_str("#define WAKE_L2_ARENA_FLOATS     35520\n");
    s.push_str("#define WAKE_L2_BUF_A_FLOATS     23680\n");
    s.push_str("#define WAKE_L2_BUF_B_FLOATS     11840\n\n");

    // Arrays
    s.push_str(&c_array_i8("g_l2_w0", &layers[0].w, 16));
    s.push_str(&format!("static const float g_l2_s0 = {:.9e}f;\n\n", layers[0].w_scale));
    s.push_str(&c_array_f32("g_l2_b0", &layers[0].b, 16));

    s.push_str(&c_array_i8("g_l2_w1_dw", &layers[1].w_dw, 16));
    s.push_str(&format!("static const float g_l2_s1_dw = {:.9e}f;\n\n", layers[1].w_dw_scale));
    s.push_str(&c_array_i8("g_l2_w1_pw", &layers[1].w_pw, 16));
    s.push_str(&format!("static const float g_l2_s1_pw = {:.9e}f;\n\n", layers[1].w_pw_scale));
    s.push_str(&c_array_f32("g_l2_b1", &layers[1].b, 16));

    s.push_str(&c_array_i8("g_l2_w2_dw", &layers[2].w_dw, 16));
    s.push_str(&format!("static const float g_l2_s2_dw = {:.9e}f;\n\n", layers[2].w_dw_scale));
    s.push_str(&c_array_i8("g_l2_w2_pw", &layers[2].w_pw, 16));
    s.push_str(&format!("static const float g_l2_s2_pw = {:.9e}f;\n\n", layers[2].w_pw_scale));
    s.push_str(&c_array_f32("g_l2_b2", &layers[2].b, 16));

    s.push_str(&c_array_i8("g_l2_w3", &layers[3].w, 16));
    s.push_str(&format!("static const float g_l2_s3 = {:.9e}f;\n\n", layers[3].w_scale));
    s.push_str(&c_array_f32("g_l2_b3", &layers[3].b, 16));

    s.push_str(&c_array_i8("g_l2_w4", &layers[4].w, 16));
    s.push_str(&format!("static const float g_l2_s4 = {:.9e}f;\n\n", layers[4].w_scale));
    s.push_str(&c_array_f32("g_l2_b4", &layers[4].b, 16));

    s.push_str(r#"static inline float wake_l2_infer(const float *spec_normalized, float *arena)
{
    float *dyn_arena = NULL;
    if (!arena) {
        dyn_arena = (float *)malloc(WAKE_L2_ARENA_FLOATS * sizeof(float));
        if (!dyn_arena) return 0.0f;
        arena = dyn_arena;
    }

    float *buf_a = arena;
    float *buf_b = arena + WAKE_L2_BUF_A_FLOATS;
    const float *in = spec_normalized;

    /* Блок 0: Conv2D 5x5, выход 74x20x16 + ReLU */
    for (int oc = 0; oc < 16; oc++) {
        for (int oh = 0; oh < 74; oh++) {
            for (int ow = 0; ow < 20; ow++) {
                float sum = g_l2_b0[oc];
                for (int kh = 0; kh < 5; kh++) {
                    int ih = oh * 2 - 2 + kh;
                    if (ih < 0 || ih >= 148) continue;
                    for (int kw = 0; kw < 5; kw++) {
                        int iw = ow * 2 - 2 + kw;
                        if (iw < 0 || iw >= 40) continue;
                        int wi = ((oc * 1 + 0) * 5 + kh) * 5 + kw;
                        sum += in[ih * 40 + iw] * (g_l2_w0[wi] * g_l2_s0);
                    }
                }
                buf_a[(oh * 20 + ow) * 16 + oc] = (sum > 0.0f) ? sum : 0.0f;
            }
        }
    }

    /* Блок 1: depthwise 3x3 + pointwise -> 32, выход 37x10x32 + ReLU */
    for (int oh = 0; oh < 37; oh++) {
        for (int ow = 0; ow < 10; ow++) {
            float dw_out[16];
            for (int c = 0; c < 16; c++) {
                float sum = 0.0f;
                for (int kh = 0; kh < 3; kh++) {
                    int ih = oh * 2 - 1 + kh;
                    if (ih < 0 || ih >= 74) continue;
                    for (int kw = 0; kw < 3; kw++) {
                        int iw = ow * 2 - 1 + kw;
                        if (iw < 0 || iw >= 20) continue;
                        int wi = (c * 3 + kh) * 3 + kw;
                        sum += buf_a[(ih * 20 + iw) * 16 + c] * (g_l2_w1_dw[wi] * g_l2_s1_dw);
                    }
                }
                dw_out[c] = sum;
            }
            for (int oc = 0; oc < 32; oc++) {
                float sum = g_l2_b1[oc];
                for (int ic = 0; ic < 16; ic++) {
                    sum += dw_out[ic] * (g_l2_w1_pw[oc * 16 + ic] * g_l2_s1_pw);
                }
                buf_b[(oh * 10 + ow) * 32 + oc] = (sum > 0.0f) ? sum : 0.0f;
            }
        }
    }

    /* Блок 2: depthwise 3x3 + pointwise -> 64, выход 19x5x64 + ReLU */
    for (int oh = 0; oh < 19; oh++) {
        for (int ow = 0; ow < 5; ow++) {
            float dw_out[32];
            for (int c = 0; c < 32; c++) {
                float sum = 0.0f;
                for (int kh = 0; kh < 3; kh++) {
                    int ih = oh * 2 - 1 + kh;
                    if (ih < 0 || ih >= 37) continue;
                    for (int kw = 0; kw < 3; kw++) {
                        int iw = ow * 2 - 1 + kw;
                        if (iw < 0 || iw >= 10) continue;
                        int wi = (c * 3 + kh) * 3 + kw;
                        sum += buf_b[(ih * 10 + iw) * 32 + c] * (g_l2_w2_dw[wi] * g_l2_s2_dw);
                    }
                }
                dw_out[c] = sum;
            }
            for (int oc = 0; oc < 64; oc++) {
                float sum = g_l2_b2[oc];
                for (int ic = 0; ic < 32; ic++) {
                    sum += dw_out[ic] * (g_l2_w2_pw[oc * 32 + ic] * g_l2_s2_pw);
                }
                buf_a[(oh * 5 + ow) * 64 + oc] = (sum > 0.0f) ? sum : 0.0f;
            }
        }
    }

    /* Temporal pooling: среднее по mel-бинам -> 19x64, затем adaptive avg pool 19 -> 4 */
    float *t_feats = buf_b;
    float *c_in = buf_b + (19 * 64);
    for (int t = 0; t < 19; t++) {
        for (int c = 0; c < 64; c++) {
            float msum = 0.0f;
            for (int w = 0; w < 5; w++) {
                msum += buf_a[(t * 5 + w) * 64 + c];
            }
            t_feats[t * 64 + c] = msum / (float)5;
        }
    }

    for (int c = 0; c < 64; c++) {
        c_in[c * 4 + 0] = (t_feats[0 * 64 + c] + t_feats[1 * 64 + c] + t_feats[2 * 64 + c] + t_feats[3 * 64 + c] + t_feats[4 * 64 + c]) / (float)5;
        c_in[c * 4 + 1] = (t_feats[4 * 64 + c] + t_feats[5 * 64 + c] + t_feats[6 * 64 + c] + t_feats[7 * 64 + c] + t_feats[8 * 64 + c] + t_feats[9 * 64 + c]) / (float)6;
        c_in[c * 4 + 2] = (t_feats[9 * 64 + c] + t_feats[10 * 64 + c] + t_feats[11 * 64 + c] + t_feats[12 * 64 + c] + t_feats[13 * 64 + c] + t_feats[14 * 64 + c]) / (float)6;
        c_in[c * 4 + 3] = (t_feats[14 * 64 + c] + t_feats[15 * 64 + c] + t_feats[16 * 64 + c] + t_feats[17 * 64 + c] + t_feats[18 * 64 + c]) / (float)5;
    }

    /* FC1: 256 -> 32 + ReLU */
    float fc1[32];
    for (int i = 0; i < 32; i++) {
        float sum = g_l2_b3[i];
        for (int j = 0; j < 256; j++) {
            sum += c_in[j] * (g_l2_w3[i * 256 + j] * g_l2_s3);
        }
        fc1[i] = (sum > 0.0f) ? sum : 0.0f;
    }

    /* FC2: 32 -> 1 + Sigmoid */
    float logit = g_l2_b4[0];
    for (int i = 0; i < 32; i++) {
        logit += fc1[i] * (g_l2_w4[i] * g_l2_s4);
    }

    if (dyn_arena) free(dyn_arena);
    return 1.0f / (1.0f + expf(-logit));
}
"#);

    s
}

pub fn save_headers<P: AsRef<Path>>(
    out_dir: P,
    l1_model: &WakeNetL1,
    l2_model: &WakeNetL2,
    th_l1: f32,
    th_l2: f32,
) -> std::io::Result<()> {
    std::fs::create_dir_all(&out_dir)?;

    let l1_h = generate_c_header_l1(l1_model);
    let mut f1 = File::create(out_dir.as_ref().join("wake_model_ey_gin_l1.h"))?;
    f1.write_all(l1_h.as_bytes())?;

    let l2_h = generate_c_header_l2(l2_model, th_l1, th_l2);
    let mut f2 = File::create(out_dir.as_ref().join("wake_model_ey_gin_l2.h"))?;
    f2.write_all(l2_h.as_bytes())?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Device, DeviceKind};

    #[test]
    fn test_quantize_int8() {
        let weights = vec![0.0f32, 1.27f32, -1.27f32];
        let (q, scale) = quantize_int8(&weights);
        assert_eq!(q[0], 0);
        assert_eq!(q[1], 127);
        assert_eq!(q[2], -127);
        assert!((scale - 0.01).abs() < 1e-6);
    }

    #[test]
    fn test_header_generation_shapes() {
        let device = Device::wgpu(DeviceKind::DiscreteGpu(0));
        let l1 = WakeNetL1::new(&device);
        let l2 = WakeNetL2::new(&device);

        let h1 = generate_c_header_l1(&l1);
        assert!(h1.contains("#define WAKE_L1_MODEL_LEN        4024"));
        assert!(h1.contains("static inline float wake_l1_infer"));

        let h2 = generate_c_header_l2(&l2, 0.40, 0.98);
        assert!(h2.contains("#define WAKE_L2_MODEL_LEN        11616"));
        assert!(h2.contains("static inline float wake_l2_infer"));
        println!("C header generation validated: L1 len=4024, L2 len=11616!");
    }
}
