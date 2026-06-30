//! Optional generation profiling (`VOXCPM_PROFILE`, `VOXCPM_PROFILE_STREAM`).

use std::time::Duration;

/// Whether detailed per-stage profiling is enabled.
#[must_use]
pub fn profile_enabled() -> bool {
    std::env::var("VOXCPM_PROFILE").is_ok() || std::env::var("VOXCPM_PROFILE_STREAM").is_ok()
}

/// Whether stream iterator timing (inference vs VAE) is enabled.
#[must_use]
pub fn profile_stream_enabled() -> bool {
    std::env::var("VOXCPM_PROFILE_STREAM").is_ok() || profile_enabled()
}

/// Accumulated per-latent inference timings (CFM, stop, LM advance).
#[derive(Debug, Default, Clone)]
pub struct InferenceStepProfile {
    pub cfm: Duration,
    pub stop: Duration,
    pub lm_advance: Duration,
    pub latent_count: usize,
}

impl InferenceStepProfile {
    pub fn record_cfm(&mut self, d: Duration) {
        self.cfm += d;
        self.latent_count += 1;
    }

    pub fn print_summary(&self) {
        if self.latent_count == 0 {
            return;
        }
        eprintln!(
            "VOXCPM_PROFILE inference_steps={} cfm={:.3}s stop={:.3}s lm_advance={:.3}s avg_cfm_ms={:.2}",
            self.latent_count,
            self.cfm.as_secs_f64(),
            self.stop.as_secs_f64(),
            self.lm_advance.as_secs_f64(),
            self.cfm.as_secs_f64() * 1000.0 / self.latent_count as f64,
        );
    }
}

/// End-to-end benchmark metrics for CLI / smoke tests.
#[derive(Debug, Clone, Copy)]
pub struct BenchmarkMetrics {
    pub load_secs: f64,
    pub prompt_cache_secs: f64,
    pub prefill_secs: f64,
    pub ttfa_secs: f64,
    pub wall_secs: f64,
    pub audio_secs: f64,
    pub rtf: f64,
    pub latent_count: usize,
    pub pcm_chunks: usize,
}

impl BenchmarkMetrics {
    pub fn print(&self, label: &str) {
        eprintln!(
            "VOXCPM_BENCH {label} load={:.3}s prompt_cache={:.3}s prefill={:.3}s ttfa={:.3}s wall={:.3}s audio={:.3}s rtf={:.3} latents={} chunks={}",
            self.load_secs,
            self.prompt_cache_secs,
            self.prefill_secs,
            self.ttfa_secs,
            self.wall_secs,
            self.audio_secs,
            self.rtf,
            self.latent_count,
            self.pcm_chunks,
        );
    }
}
