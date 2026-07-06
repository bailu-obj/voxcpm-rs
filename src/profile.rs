//! Optional generation profiling (`VOXCPM_PROFILE`, `VOXCPM_PROFILE_STREAM`).

use std::sync::Mutex;
use std::time::Duration;

use crate::quant::{QuantStats, VoxCPMWeightQuant};

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

/// Capture stage timings for benchmark JSON (`VOXCPM_BENCH_PROFILE=1` or `--profile`).
#[must_use]
pub fn bench_profile_enabled() -> bool {
    match std::env::var("VOXCPM_BENCH_PROFILE") {
        Ok(v) => !v.is_empty() && v != "0",
        Err(_) => false,
    }
}

#[must_use]
pub fn stage_capture_enabled() -> bool {
    profile_enabled() || bench_profile_enabled()
}

static STAGE_PROFILE: Mutex<Option<StageProfile>> = Mutex::new(None);

/// Reset captured stage profile (call before a timed generation).
pub fn reset_stage_profile() {
    if let Ok(mut guard) = STAGE_PROFILE.lock() {
        *guard = None;
    }
}

/// Take the last captured stage profile.
#[must_use]
pub fn take_stage_profile() -> Option<StageProfile> {
    STAGE_PROFILE.lock().ok()?.take()
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

/// End-to-end stage breakdown for benchmark reporting.
#[derive(Debug, Default, Clone)]
pub struct StageProfile {
    pub prefill_secs: f64,
    pub cfm_secs: f64,
    pub stop_secs: f64,
    pub lm_advance_secs: f64,
    pub vae_decode_secs: f64,
    pub latent_count: usize,
}

impl StageProfile {
    #[must_use]
    pub fn total_inference_secs(&self) -> f64 {
        self.cfm_secs + self.stop_secs + self.lm_advance_secs
    }

    #[must_use]
    pub fn total_secs(&self) -> f64 {
        self.prefill_secs + self.total_inference_secs() + self.vae_decode_secs
    }

    pub fn store_inference(profile: &InferenceStepProfile, prefill: Option<Duration>) {
        if let Ok(mut guard) = STAGE_PROFILE.lock() {
            let vae_decode_secs = guard.as_ref().map(|s| s.vae_decode_secs).unwrap_or(0.0);
            *guard = Some(StageProfile {
                prefill_secs: prefill.map(|d| d.as_secs_f64()).unwrap_or(0.0),
                cfm_secs: profile.cfm.as_secs_f64(),
                stop_secs: profile.stop.as_secs_f64(),
                lm_advance_secs: profile.lm_advance.as_secs_f64(),
                vae_decode_secs,
                latent_count: profile.latent_count,
            });
        }
    }

    pub fn add_vae_decode(d: Duration) {
        if let Ok(mut guard) = STAGE_PROFILE.lock() {
            let mut stage = guard.take().unwrap_or_default();
            stage.vae_decode_secs += d.as_secs_f64();
            *guard = Some(stage);
        }
    }

    pub fn store_stream(
        prefill: Duration,
        inference: Duration,
        vae: Duration,
        latent_count: usize,
    ) {
        if let Ok(mut guard) = STAGE_PROFILE.lock() {
            *guard = Some(StageProfile {
                prefill_secs: prefill.as_secs_f64(),
                cfm_secs: inference.as_secs_f64(),
                vae_decode_secs: vae.as_secs_f64(),
                latent_count,
                ..StageProfile::default()
            });
        }
    }
}

/// End-to-end benchmark metrics for CLI / smoke tests.
#[derive(Debug, Clone)]
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
    pub cfm_secs: f64,
    pub stop_secs: f64,
    pub lm_advance_secs: f64,
    pub vae_decode_secs: f64,
    pub fp_correlation: Option<f64>,
}

impl BenchmarkMetrics {
    pub fn from_stage(
        load_secs: f64,
        prompt_cache_secs: f64,
        ttfa_secs: f64,
        wall_secs: f64,
        audio_secs: f64,
        pcm_chunks: usize,
        stage: Option<StageProfile>,
        fp_correlation: Option<f64>,
    ) -> Self {
        let rtf = if audio_secs > 0.0 {
            wall_secs / audio_secs
        } else {
            0.0
        };
        let (prefill_secs, cfm_secs, stop_secs, lm_advance_secs, vae_decode_secs, latent_count) =
            if let Some(s) = stage {
                (
                    s.prefill_secs,
                    s.cfm_secs,
                    s.stop_secs,
                    s.lm_advance_secs,
                    s.vae_decode_secs,
                    s.latent_count,
                )
            } else {
                (0.0, 0.0, 0.0, 0.0, 0.0, 0)
            };
        Self {
            load_secs,
            prompt_cache_secs,
            prefill_secs,
            ttfa_secs,
            wall_secs,
            audio_secs,
            rtf,
            latent_count,
            pcm_chunks,
            cfm_secs,
            stop_secs,
            lm_advance_secs,
            vae_decode_secs,
            fp_correlation,
        }
    }

    pub fn print(&self, label: &str) {
        eprintln!(
            "VOXCPM_BENCH {label} load={:.3}s prompt_cache={:.3}s prefill={:.3}s ttfa={:.3}s wall={:.3}s audio={:.3}s rtf={:.3} latents={} chunks={} cfm={:.3}s stop={:.3}s lm={:.3}s vae={:.3}s",
            self.load_secs,
            self.prompt_cache_secs,
            self.prefill_secs,
            self.ttfa_secs,
            self.wall_secs,
            self.audio_secs,
            self.rtf,
            self.latent_count,
            self.pcm_chunks,
            self.cfm_secs,
            self.stop_secs,
            self.lm_advance_secs,
            self.vae_decode_secs,
        );
        if let Some(corr) = self.fp_correlation {
            eprintln!("VOXCPM_BENCH fp_correlation={corr:.4}");
        }
    }

    pub fn print_json_line(&self, label: &str, quant: Option<&QuantStats>) {
        let quant_json = quant.map(|q| {
            format!(
                "\"quant_requested\":\"{}\",\"quantized\":{},\"skipped\":{},\"fallback_fp\":{},\"fallback_q8\":{},\"quant_load_secs\":{:.6},\"dense_bytes\":{},\"quantized_bytes\":{},\"mixed_k_quant\":{}",
                q.requested.as_str(),
                q.quantized,
                q.skipped,
                q.fallback_fp,
                q.fallback_q8,
                q.quant_load_secs,
                q.dense_bytes,
                q.quantized_bytes,
                q.is_mixed_k_quant(),
            )
        });
        let fp_corr = self
            .fp_correlation
            .map(|c| format!("{c:.6}"))
            .unwrap_or_else(|| "null".to_string());
        let quant_part = quant_json.map(|q| format!(",{q}")).unwrap_or_default();
        println!(
            "VOXCPM_BENCH_JSON {{\"mode\":\"{label}\",\"load_secs\":{:.6},\"prompt_cache_secs\":{:.6},\"prefill_secs\":{:.6},\"ttfa_secs\":{:.6},\"wall_secs\":{:.6},\"audio_secs\":{:.6},\"rtf\":{:.6},\"latent_count\":{},\"pcm_chunks\":{},\"cfm_secs\":{:.6},\"stop_secs\":{:.6},\"lm_advance_secs\":{:.6},\"vae_decode_secs\":{:.6},\"fp_correlation\":{fp_corr}{quant_part}}}",
            self.load_secs,
            self.prompt_cache_secs,
            self.prefill_secs,
            self.ttfa_secs,
            self.wall_secs,
            self.audio_secs,
            self.rtf,
            self.latent_count,
            self.pcm_chunks,
            self.cfm_secs,
            self.stop_secs,
            self.lm_advance_secs,
            self.vae_decode_secs,
        );
    }

    /// Machine-readable summary line for benchmark scripts.
    pub fn print_cli_summary(&self) {
        println!("Average time cost: {:.2} seconds", self.wall_secs);
        println!("Average audio time: {:.2} seconds", self.audio_secs);
        if self.audio_secs > 0.0 {
            println!("Average rtf: {:.2}", self.rtf);
        } else {
            println!("Average rtf: n/a");
        }
        if self.ttfa_secs > 0.0 {
            println!("Average ttfa: {:.2} seconds", self.ttfa_secs);
        }
    }
}

/// Minimum acceptable PCM correlation vs FP for a quant mode.
#[must_use]
pub fn compare_fp_min_correlation(quant: VoxCPMWeightQuant) -> f64 {
    match quant {
        VoxCPMWeightQuant::None => 0.0,
        VoxCPMWeightQuant::Q8_0 => 0.95,
        VoxCPMWeightQuant::Q6K => 0.90,
        VoxCPMWeightQuant::Q4K | VoxCPMWeightQuant::Q5K => 0.85,
    }
}

/// Legacy floor used only when no quant mode is specified.
pub const COMPARE_FP_MIN_CORRELATION: f64 = 0.35;

/// Suggest the next optimization target from stage + quant stats.
#[must_use]
pub fn bottleneck_hint(stage: &StageProfile, quant: Option<&QuantStats>) -> String {
    if let Some(q) = quant {
        if q.is_mixed_k_quant() {
            return format!(
                "mixed_k_quant: requested {} with {} q8_0 fallbacks — use q8_0 or fix block alignment",
                q.requested.as_str(),
                q.fallback_q8
            );
        }
        if q.quant_load_secs > 2.0 && q.quantized > 0 {
            return format!(
                "load_quant: {:.1}s quantize-at-load dominates cold start — consider pre-quantized weights or cache",
                q.quant_load_secs
            );
        }
    }
    let total = stage.total_secs();
    if total < 1e-6 {
        return "insufficient_profile: enable --profile or VOXCPM_BENCH_PROFILE=1".to_string();
    }
    let vae_pct = stage.vae_decode_secs / total;
    let cfm_pct = stage.cfm_secs / total;
    let lm_pct = stage.lm_advance_secs / total;
    if vae_pct > 0.30 {
        return format!(
            "vae_decode: {:.0}% of stage time — LM/DiT quant alone won't move RTF much; tune VAE batch/dtype",
            vae_pct * 100.0
        );
    }
    if cfm_pct > 0.45 {
        return format!(
            "cfm_dit: {:.0}% of stage time — focus QMatMul/casts in DiT MLP (inference_timesteps={})",
            cfm_pct * 100.0,
            stage.latent_count
        );
    }
    if lm_pct > 0.35 {
        return format!(
            "lm_advance: {:.0}% of stage time — check quant dtype churn (F16<->F32) in QLinear",
            lm_pct * 100.0
        );
    }
    "balanced: no single stage >45%; measure dtype=f32 vs auto for quant".to_string()
}

/// Basic PCM sanity check: non-empty, not near-silent, and enough signal variance.
#[must_use]
pub fn audio_quality_ok(samples: &[i16]) -> bool {
    if samples.is_empty() {
        return false;
    }
    let peak = samples.iter().map(|s| s.abs()).max().unwrap_or(0);
    if peak <= 64 {
        return false;
    }
    let n = samples.len() as f64;
    let sum_sq: f64 = samples.iter().map(|&s| (s as f64).powi(2)).sum::<f64>();
    let rms = (sum_sq / n).sqrt();
    if rms < 100.0 {
        return false;
    }
    let mean: f64 = samples.iter().map(|&s| s as f64).sum::<f64>() / n;
    let var: f64 = samples
        .iter()
        .map(|&s| ((s as f64) - mean).powi(2))
        .sum::<f64>()
        / n;
    var > 1e4
}

/// Whether FP-vs-quant comparison is requested (`VOXCPM_COMPARE_FP=1`).
#[must_use]
pub fn compare_fp_enabled() -> bool {
    match std::env::var("VOXCPM_COMPARE_FP") {
        Ok(v) => !v.is_empty() && v != "0",
        Err(_) => false,
    }
}

/// Pearson correlation between two PCM buffers (aligned to min length).
#[must_use]
pub fn pcm_correlation(a: &[i16], b: &[i16]) -> f64 {
    let n = a.len().min(b.len());
    if n == 0 {
        return 0.0;
    }
    let n_f = n as f64;
    let mean_a: f64 = a.iter().take(n).map(|&s| s as f64).sum::<f64>() / n_f;
    let mean_b: f64 = b.iter().take(n).map(|&s| s as f64).sum::<f64>() / n_f;
    let mut cov = 0.0;
    let mut var_a = 0.0;
    let mut var_b = 0.0;
    for i in 0..n {
        let da = a[i] as f64 - mean_a;
        let db = b[i] as f64 - mean_b;
        cov += da * db;
        var_a += da * da;
        var_b += db * db;
    }
    let denom = (var_a * var_b).sqrt();
    if denom < 1e-12 {
        0.0
    } else {
        cov / denom
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn audio_quality_ok_detects_silence() {
        assert!(!audio_quality_ok(&[]));
        assert!(!audio_quality_ok(&[0, 0, 1, 0]));
        assert!(audio_quality_ok(&[0, 1000, -800, 0]));
    }

    #[test]
    fn pcm_correlation_identical() {
        let s = vec![100i16, -200, 300, -50];
        assert!((pcm_correlation(&s, &s) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn compare_fp_thresholds_by_mode() {
        assert!((compare_fp_min_correlation(VoxCPMWeightQuant::Q8_0) - 0.95).abs() < 1e-6);
        assert!(compare_fp_min_correlation(VoxCPMWeightQuant::Q4K) < 0.90);
    }

    #[test]
    fn bottleneck_hint_vae_dominant() {
        let stage = StageProfile {
            cfm_secs: 0.5,
            lm_advance_secs: 0.3,
            vae_decode_secs: 2.0,
            ..Default::default()
        };
        let hint = bottleneck_hint(&stage, None);
        assert!(hint.contains("vae_decode"));
    }
}
