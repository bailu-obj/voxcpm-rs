//! # VoxCPM:  Standalone Text-to-Speech Library
//!
//! A pure Rust implementation of the VoxCPM TTS model.

pub mod audio_vae;
pub mod common;
pub mod config;
pub mod generate;
pub mod kv_cache;
pub mod linear;
pub mod minicpm4;
pub mod models;
pub mod position_embed;
pub mod profile;
pub mod quant;
pub mod tokenizer;
pub mod utils;

pub use config::{AudioVaeConfig, VoxCPMConfig};
pub use generate::{
    COMPARE_FP_DEFAULT_SEED, DEFAULT_STREAM_DECODE_INITIAL_LATENT_BATCH,
    DEFAULT_STREAM_DECODE_LATENT_BATCH, VoxCPMGenerationConfig, VoxCPMGenerationDiagnostics,
    VoxCPMGenerator, VoxCPMGeneratorOptions, VoxCPMStopReason, VoxCPMStreamContext,
};
pub use profile::{
    BenchmarkMetrics, COMPARE_FP_MIN_CORRELATION, InferenceStepProfile, StageProfile,
    audio_quality_ok, bench_profile_enabled, bottleneck_hint, compare_fp_enabled,
    compare_fp_min_correlation, pcm_correlation, reset_stage_profile, take_stage_profile,
};
pub use quant::{
    QuantBuildCtx, QuantStats, VoxCPMQuantConfig, VoxCPMWeightQuant, quant_audit_enabled,
};

pub const VERSION: &str = env!("CARGO_PKG_VERSION");
