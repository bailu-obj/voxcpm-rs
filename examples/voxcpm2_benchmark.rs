//! Manual GPU benchmark for VoxCPM2 (ignored; run with `--ignored`).
//!
//! ```bash
//! VOXCPM_PROFILE=1 cargo run -p voxcpm-rs --example voxcpm2_benchmark --release -- --ignored
//! cargo run -p voxcpm-rs --example voxcpm2_benchmark --release -- \
//!   --model models/VoxCPM2 --stream
//! ```

use anyhow::Result;
use candle_core::{Device, Tensor};
use clap::Parser;
use std::path::PathBuf;
use std::time::Instant;
use voxcpm_rs::profile::BenchmarkMetrics;
use voxcpm_rs::{VoxCPMGenerationConfig, VoxCPMGenerator};

#[derive(Parser, Debug)]
#[command(name = "voxcpm2_benchmark")]
struct Args {
    #[arg(long, default_value = "models/VoxCPM2")]
    model: PathBuf,
    #[arg(long)]
    ref_wav: Option<PathBuf>,
    #[arg(long, requires = "ref_wav")]
    ref_text: Option<String>,
    #[arg(long, default_value = "VoxCPM2 short benchmark utterance.")]
    text: String,
    #[arg(long, default_value = "false")]
    stream: bool,
    #[arg(long, default_value = "8")]
    inference_timesteps: usize,
    #[arg(long, default_value = "8")]
    stream_decode_latent_batch: usize,
    #[arg(long, default_value = "4")]
    stop_check_interval: usize,
    /// Use the Metal RTF preset (same as voice_clone defaults after optimization).
    #[arg(long, default_value = "false")]
    metal_rtf: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let load_start = Instant::now();
    let mut generator = VoxCPMGenerator::new(args.model.to_str().unwrap(), None, None)?;
    let load_secs = load_start.elapsed().as_secs_f64();

    let mut prompt_cache_secs = 0.0;
    if let (Some(wav), Some(text)) = (&args.ref_wav, &args.ref_text) {
        let t0 = Instant::now();
        generator.build_prompt_cache(text.clone(), wav.to_string_lossy().to_string())?;
        prompt_cache_secs = t0.elapsed().as_secs_f64();
    }

    let mut config = if args.metal_rtf {
        VoxCPMGenerationConfig::metal_rtf()
    } else if args.ref_wav.is_some() {
        VoxCPMGenerationConfig::voice_clone()
    } else {
        VoxCPMGenerationConfig::simple()
    };
    config.inference_timesteps = args.inference_timesteps;
    config.stream_decode_latent_batch = args.stream_decode_latent_batch;
    config.stop_check_interval = args.stop_check_interval;

    let gen_start = Instant::now();
    let (audio, ttfa_secs, pcm_chunks, latent_count) = if args.stream {
        run_stream(&mut generator, &args.text, config)?
    } else {
        let tensor = generator.generate_with_config(args.text.clone(), config)?;
        let tensor_cpu = tensor.to_device(&Device::Cpu)?;
        let dur = audio_duration_secs(&tensor_cpu, generator.sample_rate())?;
        (tensor_cpu, dur, 1, 0)
    };
    let wall_secs = gen_start.elapsed().as_secs_f64();
    let audio_secs = audio_duration_secs(&audio, generator.sample_rate())?;
    let rtf = if audio_secs > 0.0 {
        wall_secs / audio_secs
    } else {
        0.0
    };

    BenchmarkMetrics {
        load_secs,
        prompt_cache_secs,
        prefill_secs: 0.0,
        ttfa_secs,
        wall_secs,
        audio_secs,
        rtf,
        latent_count,
        pcm_chunks,
    }
    .print(if args.stream { "stream" } else { "batch" });

    Ok(())
}

fn run_stream(
    generator: &mut VoxCPMGenerator,
    text: &str,
    config: VoxCPMGenerationConfig,
) -> Result<(Tensor, f64, usize, usize)> {
    let start = Instant::now();
    let mut first_chunk: Option<f64> = None;
    let mut chunks = Vec::new();
    let stream = generator.generate_stream_with_config(text.to_string(), config)?;
    let mut latent_count = 0usize;
    for chunk in stream {
        latent_count += 1;
        let chunk = chunk?;
        if first_chunk.is_none() {
            first_chunk = Some(start.elapsed().as_secs_f64());
        }
        chunks.push(chunk);
    }
    let tensor = if chunks.is_empty() {
        Tensor::zeros((1, 1), candle_core::DType::F32, &Device::Cpu)?
    } else {
        Tensor::cat(&chunks, 1)?
    };
    Ok((
        tensor,
        first_chunk.unwrap_or(wall_secs_fallback(start)),
        chunks.len(),
        latent_count,
    ))
}

fn wall_secs_fallback(start: Instant) -> f64 {
    start.elapsed().as_secs_f64()
}

fn audio_duration_secs(audio: &Tensor, sample_rate: usize) -> Result<f64> {
    let n = match audio.dims().len() {
        1 => audio.dim(0)?,
        2 => audio.dim(1)?,
        _ => anyhow::bail!("expected audio rank 1 or 2"),
    };
    Ok(n as f64 / sample_rate as f64)
}
