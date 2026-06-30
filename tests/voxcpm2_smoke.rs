//! Manual GPU/model smoke + performance test for VoxCPM2 weights.
//!
//! Run when `OpenBMB/VoxCPM2` is available locally:
//! `VOXCPM_PROFILE=1 cargo test -p voxcpm-rs --test voxcpm2_smoke -- --ignored --nocapture`

use std::time::Instant;
use voxcpm_rs::profile::BenchmarkMetrics;
use voxcpm_rs::{VoxCPMGenerationConfig, VoxCPMGenerator};

#[test]
#[ignore = "requires GPU and downloaded VoxCPM2 weights"]
fn voxcpm2_load_and_short_generate() -> anyhow::Result<()> {
    let model_path = std::env::var("VOXCPM2_MODEL_PATH")
        .unwrap_or_else(|_| "models/VoxCPM2".to_string());
    let load_start = Instant::now();
    let mut generator = VoxCPMGenerator::new(&model_path, None, None)?;
    let load_secs = load_start.elapsed().as_secs_f64();
    assert_eq!(generator.model_name(), "VoxCPM2");
    assert_eq!(generator.sample_rate(), 48_000);

    let text = "VoxCPM2 vendor smoke test.".to_string();
    let gen_start = Instant::now();
    let audio = generator.generate_with_config(text, VoxCPMGenerationConfig::simple())?;
    let wall_secs = gen_start.elapsed().as_secs_f64();
    let samples = match audio.dims().len() {
        1 => audio.dim(0)?,
        2 => audio.dim(1)?,
        _ => 0,
    };
    let audio_secs = samples as f64 / generator.sample_rate() as f64;
    let rtf = if audio_secs > 0.0 {
        wall_secs / audio_secs
    } else {
        0.0
    };

    BenchmarkMetrics {
        load_secs,
        prompt_cache_secs: 0.0,
        prefill_secs: 0.0,
        ttfa_secs: wall_secs,
        wall_secs,
        audio_secs,
        rtf,
        latent_count: 0,
        pcm_chunks: 1,
    }
    .print("smoke_batch");

    Ok(())
}

#[test]
#[ignore = "requires GPU and downloaded VoxCPM2 weights"]
fn voxcpm2_stream_short_generate() -> anyhow::Result<()> {
    let model_path = std::env::var("VOXCPM2_MODEL_PATH")
        .unwrap_or_else(|_| "models/VoxCPM2".to_string());
    let mut generator = VoxCPMGenerator::new(&model_path, None, None)?;
    let text = "VoxCPM2 streaming smoke.".to_string();
    let start = Instant::now();
    let mut first_chunk = None;
    let mut chunks = 0usize;
    for item in generator.generate_stream_with_config(text, VoxCPMGenerationConfig::simple())? {
        let _ = item?;
        chunks += 1;
        if first_chunk.is_none() {
            first_chunk = Some(start.elapsed().as_secs_f64());
        }
    }
    assert!(chunks > 0, "expected at least one latent chunk");
    eprintln!(
        "VOXCPM_BENCH smoke_stream ttfa={:.3}s chunks={}",
        first_chunk.unwrap_or(0.0),
        chunks
    );
    Ok(())
}
