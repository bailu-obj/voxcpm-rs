//! Manual GPU/model smoke + performance test for VoxCPM2 weights.
//!
//! Run when `OpenBMB/VoxCPM2` is available locally:
//! `VOXCPM_PROFILE=1 cargo test -p voxcpm-rs --test voxcpm2_smoke -- --ignored --nocapture`

use std::time::Instant;
use voxcpm_rs::profile::{pcm_correlation, BenchmarkMetrics};
use voxcpm_rs::{
    VoxCPMGenerationConfig, VoxCPMGenerator, VoxCPMGeneratorOptions, VoxCPMStopReason,
};

#[test]
#[ignore = "requires GPU and downloaded VoxCPM2 weights"]
fn voxcpm2_load_and_short_generate() -> anyhow::Result<()> {
    let model_path =
        std::env::var("VOXCPM2_MODEL_PATH").unwrap_or_else(|_| "models/VoxCPM2".to_string());
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
        cfm_secs: 0.0,
        stop_secs: 0.0,
        lm_advance_secs: 0.0,
        vae_decode_secs: 0.0,
        fp_correlation: None,
    }
    .print("smoke_batch");

    if let Some(diag) = generator.last_diagnostics() {
        eprintln!(
            "VOXCPM_QUALITY smoke stop={} latents={} tokens={} max_len={}",
            diag.stop_reason.as_str(),
            diag.latent_count,
            diag.target_text_tokens,
            diag.effective_max_len
        );
    }

    Ok(())
}

#[test]
#[ignore = "requires GPU and downloaded VoxCPM2 weights"]
fn voxcpm2_stream_short_generate() -> anyhow::Result<()> {
    let model_path =
        std::env::var("VOXCPM2_MODEL_PATH").unwrap_or_else(|_| "models/VoxCPM2".to_string());
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

/// Quality fixture: conversational Chinese with opening/closing interjections.
/// Prefers stop-head completion and stream/batch edge parity under a fixed seed.
#[test]
#[ignore = "requires GPU and downloaded VoxCPM2 weights"]
fn voxcpm2_paimon_chat_line_stop_and_stream_parity() -> anyhow::Result<()> {
    let model_path =
        std::env::var("VOXCPM2_MODEL_PATH").unwrap_or_else(|_| "models/VoxCPM2".to_string());
    let text = "诶嘿！那就好，那就好！刚才派蒙还担心旅行者是不是因为太累了，所以睡着了呢。嘿嘿。"
        .to_string();
    // Uses voice_clone defaults (Bailu sync: min_len=2, stop=1, batches 4/8).
    let cfg = VoxCPMGenerationConfig::voice_clone();

    let mut options = VoxCPMGeneratorOptions::default();
    options.seed = Some(42);

    let mut batch_gen = VoxCPMGenerator::new_with_options(&model_path, &options)?;
    let batch = batch_gen.generate_with_config(text.clone(), cfg)?;
    let batch_pcm = batch_gen.to_pcm(&batch)?;
    let diag = batch_gen
        .last_diagnostics()
        .expect("batch generation should record diagnostics");
    assert_eq!(
        diag.stop_reason,
        VoxCPMStopReason::StopHead,
        "conversational line should stop via stop head, not max_len (latents={}, max={})",
        diag.latent_count,
        diag.effective_max_len
    );
    assert!(
        diag.latent_count < diag.effective_max_len,
        "expected headroom under effective_max_len"
    );

    let mut stream_gen = VoxCPMGenerator::new_with_options(&model_path, &options)?;
    let mut stream_pcm = Vec::new();
    for chunk in stream_gen.generate_pcm_stream_with_config(text, cfg)? {
        stream_pcm.extend(chunk?);
    }
    let corr = pcm_correlation(&batch_pcm, &stream_pcm);
    eprintln!(
        "VOXCPM_QUALITY paimon_chat_line stop={} latents={} stream_batch_corr={corr:.4}",
        diag.stop_reason.as_str(),
        diag.latent_count
    );
    assert!(
        corr >= 0.90,
        "stream vs batch correlation {corr:.4} below 0.90"
    );
    Ok(())
}
