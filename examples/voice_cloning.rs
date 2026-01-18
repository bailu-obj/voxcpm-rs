use anyhow::Result;
use voxcpm_rs::VoxCPMGenerator;

fn main() -> Result<()> {
    println!("=== VoxCPM Voice Cloning Example ===\n");

    let model_path =
        std::env::var("VOXCPM_MODEL_PATH").unwrap_or_else(|_| "./VoxCPM-0.5B/".to_string());

    let mut generator = VoxCPMGenerator::new(&model_path, None, None)?;
    println!("✓ Model loaded\n");

    // Reference audio and text
    let reference_audio = "reference_voice.wav";
    let reference_text = "这是参考音频的文本内容";
    let target_text = "我想用这个声音说话";

    println!("Reference text: {}", reference_text);
    println!("Target text: {}", target_text);
    println!("Cloning voice...\n");

    let audio = generator.inference(
        target_text.to_string(),
        Some(reference_text.to_string()),
        Some(reference_audio.to_string()),
        5,   // min_len
        200, // max_len
        15,  // inference_timesteps
        2.5, // cfg_value
        6.0, // retry_threshold
    )?;

    generator.save_wav(&audio, "cloned_output.wav")?;
    println!("✓ Voice cloned successfully!");
    println!("✓ Saved to: cloned_output. wav");

    Ok(())
}
