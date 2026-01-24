use anyhow::Result;
use candle_core::Tensor;
use std::time::Instant;
use voxcpm_rs::VoxCPMGenerator;

fn main() -> Result<()> {
    println!("=== VoxCPM Streaming TTS Example ===\n");

    let model_path =
        std::env::var("VOXCPM_MODEL_PATH").unwrap_or_else(|_| "./VoxCPM-0.5B/".to_string());

    println!("Loading model from: {}", model_path);
    let start = Instant::now();

    let mut generator = VoxCPMGenerator::new(&model_path, None, None)?;

    println!("✓ Model loaded in {:?}\n", start.elapsed());

    let text = "人工智能技术正在深刻改变我们的生活方式，让未来充满无限可能。";
    println!("Text: {}", text);

    println!("Starting streaming generation...");
    let start = Instant::now();

    let stream = generator.generate_stream_simple(text.to_string())?;

    let mut chunks = Vec::new();
    let mut first_chunk_time = None;

    for (i, chunk_result) in stream.enumerate() {
        if first_chunk_time.is_none() {
            first_chunk_time = Some(start.elapsed());
            println!("✓ First chunk produced in {:?}", first_chunk_time.unwrap());
        }

        match chunk_result {
            Ok(chunk) => {
                println!("  Chunk {}: {} samples", i + 1, chunk.dim(1).unwrap_or(0));
                chunks.push(chunk);
            }
            Err(e) => {
                eprintln!("Error in stream: {}", e);
                break;
            }
        }
    }

    let total_duration = start.elapsed();
    println!("\n✓ Generation completed in {:?}", total_duration);

    if !chunks.is_empty() {
        // Concatenate all chunks to save the full audio
        let full_audio = Tensor::cat(&chunks, 1)?;
        let output_file = "streaming_output.wav";
        generator.save_wav(&full_audio, output_file)?;
        println!(
            "✓ Saved full audio ({} chunks) to: {}",
            chunks.len(),
            output_file
        );
    }

    Ok(())
}
