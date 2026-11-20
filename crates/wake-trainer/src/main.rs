use clap::Parser;
use burn::tensor::{Device, DeviceKind, Tensor};

#[derive(Parser, Debug)]
#[command(name = "wake-trainer", about = "Kouki Wake-Word GPU Trainer on Burn (WGPU)")]
struct Cli {
    /// Проверить доступность GPU и инициализацию WGPU/Vulkan
    #[arg(long)]
    check_device: bool,
}

fn main() {
    let cli = Cli::parse();

    if cli.check_device {
        println!("=== Kouki Wake-Trainer: Проверка WGPU / Vulkan ===");

        // Инициализируем дискретный GPU (AMD Radeon RX 5600 XT) через WGPU
        let device = Device::wgpu(DeviceKind::DiscreteGpu(0));
        println!("Инициализировано устройство Burn WGPU: {:?}", device);

        // Создаем тензоры на GPU и выполняем матричное умножение
        let a: Tensor<2> = Tensor::from_floats([[1.0, 2.0], [3.0, 4.0]], &device);
        let b: Tensor<2> = Tensor::from_floats([[5.0, 6.0], [7.0, 8.0]], &device);
        let c = a.matmul(b);

        let data = c.into_data();
        println!("Результат матричного умножения на GPU:");
        println!("{:?}", data);

        println!("GPU AMD Radeon RX 5600 XT через WGPU/Vulkan успешно протестирован и готов!");
        return;
    }

    println!("wake-trainer v0.1.0: используйте --check-device для проверки окружения");
}
