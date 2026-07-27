//! VoxCPM benchmark CLI: load, optional prompt cache, stream/non-stream generation, RTF/TTFA.

use anyhow::{bail, Result};
use clap::Parser;
use std::path::PathBuf;
use std::time::Instant;
use voxcpm_rs::{
    audio_quality_ok, bench_profile_enabled, bottleneck_hint, compare_fp_enabled,
    compare_fp_min_correlation, pcm_correlation, reset_stage_profile, take_stage_profile,
    BenchmarkMetrics, QuantStats, StageProfile, VoxCPMGenerationConfig, VoxCPMGenerator,
    VoxCPMGeneratorOptions, VoxCPMQuantConfig, VoxCPMWeightQuant, COMPARE_FP_DEFAULT_SEED,
};

#[derive(Parser, Debug)]
#[command(name = "voxcpm2-benchmark", about = "VoxCPM RTF/TTFA benchmark")]
struct Args {
    #[arg(long, default_value = "models/VoxCPM-0.5B")]
    model: PathBuf,

    #[arg(long, default_value = "非流式语音合成耗时测试")]
    text: String,

    #[arg(long)]
    ref_wav: Option<PathBuf>,

    #[arg(long, requires = "ref_wav")]
    ref_text: Option<String>,

    #[arg(long, default_value = "false")]
    stream: bool,

    /// Recommended: q8_0. K-quants (q4_k/q5_k/q6_k) are experimental.
    #[arg(long, default_value = "none")]
    quant: String,

    /// Use quality_first quant skip patterns (attention/bridges stay FP).
    #[arg(long, default_value = "false")]
    quant_quality_first: bool,

    #[arg(long, default_value = "auto")]
    dtype: String,

    #[arg(long, default_value = "auto")]
    vae_dtype: String,

    #[arg(long)]
    device_id: Option<usize>,

    /// Fixed RNG seed (overrides `VOXCPM_SEED` when set).
    #[arg(long)]
    seed: Option<u64>,

    /// Generation preset: `voice_clone` (default with ref), `simple`, `low_latency`, `metal_rtf`.
    #[arg(long)]
    preset: Option<String>,

    /// CFM Euler denoising steps per latent frame.
    #[arg(long)]
    inference_timesteps: Option<usize>,

    /// Fraction of Euler steps that use 2× CFG batch (default 0.5).
    #[arg(long)]
    cfg_full_fraction: Option<f64>,

    /// Compare output PCM against FP (`none`) baseline; fail if correlation too low.
    #[arg(long, default_value = "false")]
    compare_fp: bool,

    /// Compare output PCM against a previously saved 16-bit mono WAV (config A/B quality).
    #[arg(long)]
    compare_ref_wav: Option<PathBuf>,

    /// Write measured PCM to this WAV path (last measured run).
    #[arg(long)]
    save_wav: Option<PathBuf>,

    /// Capture per-stage timings (cfm/stop/lm/vae) in benchmark JSON.
    #[arg(long, default_value = "true")]
    profile: bool,

    /// Override FP correlation threshold (default: mode-specific, q8_0=0.95).
    #[arg(long)]
    quality_threshold: Option<f64>,

    /// Warmup runs before measured benchmark (discarded).
    #[arg(long, default_value = "0")]
    warmup: usize,

    /// Measured runs; report median wall/RTF when >1.
    #[arg(long, default_value = "1")]
    runs: usize,

    /// Latents per VAE decode after the first chunk (streaming only).
    #[arg(long)]
    stream_decode_latent_batch: Option<usize>,

    /// First streaming VAE decode batch in latents (streaming only); 0 = vendor auto.
    #[arg(long)]
    stream_decode_initial_latent_batch: Option<usize>,

    /// Run stop-head every N latents after min_len (streaming only).
    #[arg(long)]
    stop_check_interval: Option<usize>,
}

fn main() -> Result<()> {
    let args = Args::parse();
    if args.profile {
        std::env::set_var("VOXCPM_BENCH_PROFILE", "1");
    }
    let quant_weight: VoxCPMWeightQuant = args.quant.parse().map_err(anyhow::Error::msg)?;
    if quant_weight.is_experimental() {
        eprintln!(
            "VOXCPM_QUANT_WARN {} is experimental; q8_0 is recommended for production",
            quant_weight.as_str()
        );
    }
    let compare_fp = args.compare_fp || compare_fp_enabled();
    let compare_seed = if compare_fp && quant_weight.is_enabled() {
        Some(COMPARE_FP_DEFAULT_SEED)
    } else {
        args.seed
    };
    let quality_threshold = args.quality_threshold.unwrap_or_else(|| {
        if quant_weight.is_enabled() {
            compare_fp_min_correlation(quant_weight)
        } else {
            // Config A/B (timesteps/batch) vs a saved reference WAV.
            0.90
        }
    });

    let options = build_options(&args, quant_weight, compare_seed);
    let generation_config = build_generation_config(&args)?;
    let ref_wav_pcm = args
        .compare_ref_wav
        .as_ref()
        .map(|p| load_pcm_i16(p))
        .transpose()?;

    let fp_baseline = if compare_fp && quant_weight.is_enabled() {
        Some(run_generation(
            &args,
            build_options(&args, VoxCPMWeightQuant::None, compare_seed),
            generation_config,
        )?)
    } else {
        None
    };

    for _ in 0..args.warmup {
        let _ = run_measured(
            &args,
            &options,
            generation_config,
            None,
            ref_wav_pcm.as_deref(),
        )?;
    }

    let mut walls = Vec::new();
    let mut rtfs = Vec::new();
    let mut last: Option<(BenchmarkMetrics, QuantStats, Option<StageProfile>, Vec<i16>, u32)> =
        None;

    for _ in 0..args.runs.max(1) {
        let result = run_measured(
            &args,
            &options,
            generation_config,
            fp_baseline.as_ref(),
            ref_wav_pcm.as_deref(),
        )?;
        walls.push(result.0.wall_secs);
        rtfs.push(result.0.rtf);
        last = Some(result);
    }

    let (mut metrics, quant_stats, stage, samples, sample_rate) = last.expect("at least one run");
    if walls.len() > 1 {
        walls.sort_by(|a, b| a.partial_cmp(b).unwrap());
        rtfs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mid = walls.len() / 2;
        metrics.wall_secs = walls[mid];
        metrics.rtf = rtfs[mid];
    }

    if let Some(path) = &args.save_wav {
        save_pcm_wav(path, &samples, sample_rate)?;
        eprintln!("VOXCPM_SAVE_WAV {}", path.display());
    }

    quant_stats.print_summary();
    quant_stats.print_json_line();

    let label = if args.stream { "stream" } else { "batch" };
    metrics.print(label);
    metrics.print_json_line(label, Some(&quant_stats));
    metrics.print_cli_summary();

    if let Some(stage) = stage.as_ref() {
        let hint = bottleneck_hint(stage, Some(&quant_stats));
        eprintln!("VOXCPM_BOTTLENECK_HINT {hint}");
    }

    // Print metrics before quality gate so sweeps still capture RTF on soft fails.
    if let Some(corr) = metrics.fp_correlation {
        eprintln!("VOXCPM_COMPARE_FP correlation={corr:.4} threshold={quality_threshold:.2}");
        if corr < quality_threshold {
            bail!(
                "quality compare failed: correlation {corr:.4} < {quality_threshold:.2} (quant={})",
                quant_weight.as_str()
            );
        }
    }

    Ok(())
}

fn run_measured(
    args: &Args,
    options: &VoxCPMGeneratorOptions,
    generation_config: VoxCPMGenerationConfig,
    fp_baseline: Option<&(f64, f64, usize, Vec<i16>)>,
    ref_wav_pcm: Option<&[i16]>,
) -> Result<(
    BenchmarkMetrics,
    QuantStats,
    Option<StageProfile>,
    Vec<i16>,
    u32,
)> {
    if bench_profile_enabled() {
        reset_stage_profile();
    }

    let load_start = Instant::now();
    let mut generator = VoxCPMGenerator::new_with_options(args.model.to_str().unwrap(), options)?;
    let load_secs = load_start.elapsed().as_secs_f64();
    let quant_stats = generator.quant_stats();
    let sample_rate = generator.sample_rate() as u32;

    let mut prompt_cache_secs = 0.0;
    if let (Some(wav), Some(text)) = (&args.ref_wav, &args.ref_text) {
        let t0 = Instant::now();
        generator.build_prompt_cache(text.clone(), wav.to_string_lossy().to_string())?;
        prompt_cache_secs = t0.elapsed().as_secs_f64();
    }

    let wall_start = Instant::now();
    let (audio_secs, ttfa_secs, pcm_chunks, samples) = if args.stream {
        run_stream(&mut generator, &args.text, generation_config)?
    } else {
        run_batch(&mut generator, &args.text, generation_config)?
    };
    let wall_secs = wall_start.elapsed().as_secs_f64();

    if !audio_quality_ok(&samples) {
        bail!("audio quality check failed: empty, near-silent, or flat PCM");
    }

    let fp_correlation = fp_baseline
        .map(|(_, _, _, ref_pcm)| pcm_correlation(ref_pcm, &samples))
        .or_else(|| ref_wav_pcm.map(|ref_pcm| pcm_correlation(ref_pcm, &samples)));
    let stage = take_stage_profile();

    let metrics = BenchmarkMetrics::from_stage(
        load_secs,
        prompt_cache_secs,
        ttfa_secs,
        wall_secs,
        audio_secs,
        pcm_chunks,
        stage.clone(),
        fp_correlation,
    );

    Ok((metrics, quant_stats, stage, samples, sample_rate))
}

fn build_options(
    args: &Args,
    quant_weight: VoxCPMWeightQuant,
    seed: Option<u64>,
) -> VoxCPMGeneratorOptions {
    let mut options = VoxCPMGeneratorOptions::default();
    options.device_id = args.device_id;
    options.dtype = voxcpm_rs::utils::device::parse_dtype_option(Some(&args.dtype));
    options.vae_dtype = voxcpm_rs::utils::device::parse_dtype_option(Some(&args.vae_dtype));
    options.quant = if args.quant_quality_first && quant_weight.is_enabled() {
        VoxCPMQuantConfig::quality_first(quant_weight)
    } else {
        VoxCPMQuantConfig::with_weight(quant_weight)
    };
    options.seed = seed.or(args.seed);
    options
}

fn build_generation_config(args: &Args) -> Result<VoxCPMGenerationConfig> {
    let mut config = match args.preset.as_deref() {
        Some("metal_rtf") => VoxCPMGenerationConfig::metal_rtf(),
        Some("low_latency") => VoxCPMGenerationConfig::low_latency(),
        Some("simple") => VoxCPMGenerationConfig::simple(),
        Some("voice_clone") => VoxCPMGenerationConfig::voice_clone(),
        Some(other) => bail!(
            "unknown preset '{other}' (expected voice_clone|simple|low_latency|metal_rtf)"
        ),
        None if args.ref_wav.is_some() && args.ref_text.is_some() => {
            VoxCPMGenerationConfig::voice_clone()
        }
        None => VoxCPMGenerationConfig::simple(),
    };
    if let Some(n) = args.inference_timesteps {
        if n == 0 {
            bail!("--inference-timesteps must be > 0");
        }
        config.inference_timesteps = n;
    }
    if let Some(f) = args.cfg_full_fraction {
        if !(0.0..=1.0).contains(&f) {
            bail!("--cfg-full-fraction must be in [0, 1]");
        }
        config.cfg_full_fraction = f;
    }
    if let Some(n) = args.stream_decode_latent_batch {
        config.stream_decode_latent_batch = n;
    }
    if let Some(n) = args.stream_decode_initial_latent_batch {
        config.stream_decode_initial_latent_batch = n;
    }
    if let Some(n) = args.stop_check_interval {
        config.stop_check_interval = n;
    }
    Ok(config)
}

fn load_pcm_i16(path: &PathBuf) -> Result<Vec<i16>> {
    let mut reader = hound::WavReader::open(path)?;
    let spec = reader.spec();
    if spec.channels != 1 || spec.sample_format != hound::SampleFormat::Int {
        bail!(
            "compare-ref-wav must be 16-bit mono int WAV (got channels={}, format={:?})",
            spec.channels,
            spec.sample_format
        );
    }
    let samples: Result<Vec<i16>, _> = reader.samples::<i16>().collect();
    Ok(samples?)
}

fn save_pcm_wav(path: &PathBuf, samples: &[i16], sample_rate: u32) -> Result<()> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(path, spec)?;
    for &s in samples {
        writer.write_sample(s)?;
    }
    writer.finalize()?;
    Ok(())
}

fn run_generation(
    args: &Args,
    options: VoxCPMGeneratorOptions,
    config: VoxCPMGenerationConfig,
) -> Result<(f64, f64, usize, Vec<i16>)> {
    let mut generator = VoxCPMGenerator::new_with_options(args.model.to_str().unwrap(), &options)?;
    if let (Some(wav), Some(text)) = (&args.ref_wav, &args.ref_text) {
        generator.build_prompt_cache(text.clone(), wav.to_string_lossy().to_string())?;
    }
    if args.stream {
        run_stream(&mut generator, &args.text, config)
    } else {
        run_batch(&mut generator, &args.text, config)
    }
}

fn run_batch(
    generator: &mut VoxCPMGenerator,
    text: &str,
    config: VoxCPMGenerationConfig,
) -> Result<(f64, f64, usize, Vec<i16>)> {
    let tensor = generator.generate_with_config(text.to_string(), config)?;
    let samples = generator.to_pcm(&tensor)?;
    let audio_secs = samples.len() as f64 / generator.sample_rate() as f64;
    Ok((audio_secs, 0.0, 1, samples))
}

fn run_stream(
    generator: &mut VoxCPMGenerator,
    text: &str,
    config: VoxCPMGenerationConfig,
) -> Result<(f64, f64, usize, Vec<i16>)> {
    let stream = generator.generate_pcm_stream_with_config(text.to_string(), config)?;
    let mut samples = Vec::new();
    let mut ttfa_secs = 0.0;
    let start = Instant::now();
    let mut chunks = 0usize;
    for chunk in stream {
        let chunk = chunk?;
        if chunks == 0 {
            ttfa_secs = start.elapsed().as_secs_f64();
        }
        chunks += 1;
        samples.extend(chunk);
    }
    let audio_secs = samples.len() as f64 / generator.sample_rate() as f64;
    Ok((audio_secs, ttfa_secs, chunks, samples))
}
