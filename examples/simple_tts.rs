use anyhow::Result;
use std::time::Instant;
use voxcpm_rs::VoxCPMGenerator;

fn main() -> Result<()> {
    println!("=== VoxCPM Simple TTS Example ===\n");

    // Set model path - adjust to your location
    let model_path =
        std::env::var("VOXCPM_MODEL_PATH").unwrap_or_else(|_| "./VoxCPM-0.5B/".to_string());

    println!("Loading model from: {}", model_path);
    let start = Instant::now();

    let mut generator = VoxCPMGenerator::new(&model_path, None, None)?;

    println!("✓ Model loaded in {:?}\n", start.elapsed());

    // Test texts
    let texts = vec![
        "太阳当空照，花儿对我笑，小鸟说早早早",
        "人工智能技术正在深刻改变我们的生活方式",
        "今天天气真好，适合出去散步",
    ];

    for (i, text) in texts.iter().enumerate() {
        println!("Generating audio {} of {}", i + 1, texts.len());
        println!("Text: {}", text);

        let start = Instant::now();
        let audio = generator.generate_simple(text.to_string())?;
        let duration = start.elapsed();

        let output_file = format!("output_{}.wav", i + 1);
        generator.save_wav(&audio, &output_file)?;

        println!("✓ Saved to: {} (took {:?})\n", output_file, duration);
    }

    println!("=== All examples completed!  ===");
    Ok(())
}
