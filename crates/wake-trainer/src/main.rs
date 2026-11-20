use clap::Parser;
use std::path::PathBuf;
use wake_trainer::pipeline::{run_pipeline, run_theories_sweep, PipelineConfig};
use burn::tensor::{Device, DeviceKind, Tensor};

#[derive(Parser, Debug)]
#[command(name = "wake-trainer", about = "Kouki Wake-Word GPU Trainer on Burn (WGPU/Vulkan)")]
struct Cli {
    /// Проверить доступность GPU и инициализацию WGPU/Vulkan
    #[arg(long)]
    check_device: bool,

    /// Запустить бенчмарк пула теорий (свип по гипотезам) и выбрать лучшую
    #[arg(long)]
    sweep_theories: bool,

    /// Путь к файлу извлеченных признаков (features.safetensors)
    #[arg(long, default_value = "artifacts/cache/features.safetensors")]
    features: PathBuf,

    /// Каталог для выгрузки Си-заголовков и метрик
    #[arg(long, default_value = "artifacts/models")]
    out_dir: PathBuf,

    /// Число эпох обучения L1
    #[arg(long, default_value_t = 20)]
    epochs_l1: usize,

    /// Число эпох обучения L2
    #[arg(long, default_value_t = 20)]
    epochs_l2: usize,

    /// Размер батча
    #[arg(long, default_value_t = 64)]
    batch_size: usize,

    /// Скорость обучения L1
    #[arg(long, default_value_t = 1.5e-3)]
    lr_l1: f64,

    /// Скорость обучения L2
    #[arg(long, default_value_t = 1.0e-3)]
    lr_l2: f64,

    /// Порог L1 для отбора hard negatives
    #[arg(long, default_value_t = 0.35)]
    th_hard: f32,

    /// Кратность повтора окон с ошибками (0 или 1 — выключено)
    #[arg(long, default_value_t = 3)]
    err_repeat: usize,

    /// Порог ошибки L2 для перевзвеса
    #[arg(long, default_value_t = 0.50)]
    err_th_l2: f32,

    /// Базовый сид генератора случайных чисел
    #[arg(long, default_value_t = 42)]
    seed: u64,

    /// Лимит пула негативов для L2 (по умолчанию 1.5 * N_pos)
    #[arg(long)]
    neg_cap_l2: Option<usize>,

    /// Множитель динамического переноса легких негативов (0.0 = строгая изоляция)
    #[arg(long, default_value_t = 0.0)]
    k_easy: f32,

    /// Степень уверенности для динамического переноса (1.0 = линейный, 2.0 = квадратичный)
    #[arg(long, default_value_t = 1.0)]
    gamma_easy: f32,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    if cli.check_device {
        println!("=== Kouki Wake-Trainer: Проверка WGPU / Vulkan ===");
        let device = Device::wgpu(DeviceKind::DiscreteGpu(0));
        println!("Инициализировано устройство Burn WGPU: {:?}", device);

        let a: Tensor<2> = Tensor::from_floats([[1.0, 2.0], [3.0, 4.0]], &device);
        let b: Tensor<2> = Tensor::from_floats([[5.0, 6.0], [7.0, 8.0]], &device);
        let c = a.matmul(b);

        let data = c.into_data();
        println!("Результат матричного умножения на GPU:\n{:?}", data);
        println!("GPU AMD Radeon RX 5600 XT через WGPU/Vulkan успешно протестирован и готов!");
        return Ok(());
    }

    println!("=== Kouki Wake-Trainer v0.1.0 (Burn wgpu / Vulkan) ===");
    let device = Device::wgpu(DeviceKind::DiscreteGpu(0)).autodiff();
    println!("Выбран вычислительный дискретный GPU: {:?}", device);

    if cli.sweep_theories {
        run_theories_sweep(&cli.features, &cli.out_dir, &device)?;
        return Ok(());
    }

    let cfg = PipelineConfig {
        epochs_l1: cli.epochs_l1,
        epochs_l2: cli.epochs_l2,
        batch_size: cli.batch_size,
        lr_l1: cli.lr_l1,
        lr_l2: cli.lr_l2,
        pos_weight_l1: 2.5,
        pos_weight_l2: 2.0,
        th_hard: cli.th_hard,
        err_repeat: cli.err_repeat,
        err_th_l2: cli.err_th_l2,
        seed: cli.seed,
        neg_cap_l2: cli.neg_cap_l2,
        k_easy: cli.k_easy,
        gamma_easy: cli.gamma_easy,
    };

    run_pipeline(&cli.features, &cli.out_dir, &cfg, &device)?;

    Ok(())
}
