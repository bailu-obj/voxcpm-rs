//! Weight quantization configuration for VoxCPM linear layers.

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;
use std::str::FromStr;
use std::time::Duration;

/// Supported in-situ weight quantization modes (Candle GGUF dtypes).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum VoxCPMWeightQuant {
    #[default]
    None,
    Q8_0,
    Q4K,
    Q5K,
    Q6K,
}

impl VoxCPMWeightQuant {
    #[must_use]
    pub fn is_enabled(self) -> bool {
        !matches!(self, Self::None)
    }

    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Q8_0 => "q8_0",
            Self::Q4K => "q4_k",
            Self::Q5K => "q5_k",
            Self::Q6K => "q6_k",
        }
    }

    /// Production-recommended preset (validated speech quality on Metal).
    #[must_use]
    pub fn is_recommended(self) -> bool {
        matches!(self, Self::Q8_0)
    }

    /// K-quant modes: functional but mixed fallback and weaker quality gates.
    #[must_use]
    pub fn is_experimental(self) -> bool {
        matches!(self, Self::Q4K | Self::Q5K | Self::Q6K)
    }

    #[must_use]
    pub fn is_k_quant(self) -> bool {
        matches!(self, Self::Q4K | Self::Q5K | Self::Q6K)
    }
}

impl FromStr for VoxCPMWeightQuant {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_lowercase().as_str() {
            "" | "none" | "off" | "false" => Ok(Self::None),
            "q8" | "q80" | "q8_0" => Ok(Self::Q8_0),
            "q4k" | "q4_k" => Ok(Self::Q4K),
            "q5k" | "q5_k" => Ok(Self::Q5K),
            "q6k" | "q6_k" => Ok(Self::Q6K),
            other => Err(format!(
                "unknown VoxCPM quant mode '{other}'; expected none|q8_0|q4_k|q5_k|q6_k"
            )),
        }
    }
}

/// Quantization policy for model construction.
#[derive(Debug, Clone)]
pub struct VoxCPMQuantConfig {
    pub weight: VoxCPMWeightQuant,
    /// Extra module-path substrings that should stay unquantized.
    pub skip_patterns: Vec<String>,
}

impl Default for VoxCPMQuantConfig {
    fn default() -> Self {
        Self {
            weight: VoxCPMWeightQuant::None,
            skip_patterns: Vec::new(),
        }
    }
}

/// Whether layer quant audit logging is enabled (`VOXCPM_QUANT_AUDIT=1`).
#[must_use]
pub fn quant_audit_enabled() -> bool {
    match std::env::var("VOXCPM_QUANT_AUDIT") {
        Ok(v) => !v.is_empty() && v != "0",
        Err(_) => false,
    }
}

/// Log a quant build decision for one module path.
pub fn quant_audit_log(module_path: &str, mode: &str) {
    eprintln!("VOXCPM_QUANT_AUDIT {module_path} {mode}");
}

impl VoxCPMQuantConfig {
    #[must_use]
    pub fn disabled() -> Self {
        Self::default()
    }

    #[must_use]
    pub fn with_weight(weight: VoxCPMWeightQuant) -> Self {
        Self {
            weight,
            ..Self::default()
        }
    }

    /// Quality-first preset: quantize core LM/DiT MLP while skipping bridges and encoder.
    #[must_use]
    pub fn quality_first(weight: VoxCPMWeightQuant) -> Self {
        Self {
            weight,
            skip_patterns: vec![
                // Separate Q/K/V and gate/up projections (not fused in this codebase).
                "self_attn.q_proj".to_string(),
                "self_attn.k_proj".to_string(),
                "self_attn.v_proj".to_string(),
                "mlp.gate_proj".to_string(),
                "mlp.up_proj".to_string(),
                "feat_encoder".to_string(),
                "enc_to_lm_proj".to_string(),
                "lm_to_dit_proj".to_string(),
                "res_to_dit_proj".to_string(),
            ],
        }
    }

    #[must_use]
    pub fn is_enabled(&self) -> bool {
        self.weight.is_enabled()
    }

    /// Whether a fully-qualified module path should remain in full precision.
    #[must_use]
    pub fn should_skip_module(&self, module_path: &str) -> bool {
        const DEFAULT_SKIP: &[&str] = &["embed_tokens", "stop_head", "stop_proj", "fsq_layer"];
        for pat in DEFAULT_SKIP {
            if module_path.contains(pat) {
                return true;
            }
        }
        self.skip_patterns
            .iter()
            .any(|pat| module_path.contains(pat))
    }
}

/// Load-time aggregation of quant decisions and estimated weight footprint.
#[derive(Debug, Clone, Default)]
pub struct QuantStats {
    pub requested: VoxCPMWeightQuant,
    pub quantized: u32,
    pub skipped: u32,
    pub fallback_fp: u32,
    /// Requested K-quant but applied Q8_0 due to block-size alignment.
    pub fallback_q8: u32,
    pub quant_load_secs: f64,
    pub dense_bytes: u64,
    pub quantized_bytes: u64,
    pub by_actual_dtype: HashMap<String, u32>,
}

impl QuantStats {
    #[must_use]
    pub fn new(requested: VoxCPMWeightQuant) -> Self {
        Self {
            requested,
            ..Self::default()
        }
    }

    pub fn record_skipped(&mut self) {
        self.skipped += 1;
    }

    pub fn record_fallback_fp(&mut self, dense_bytes: u64) {
        self.fallback_fp += 1;
        self.dense_bytes += dense_bytes;
    }

    pub fn record_quantized(
        &mut self,
        requested: VoxCPMWeightQuant,
        actual_dtype: &str,
        dense_bytes: u64,
        quantized_bytes: u64,
        fallback_to_q8: bool,
    ) {
        self.quantized += 1;
        self.dense_bytes += dense_bytes;
        self.quantized_bytes += quantized_bytes;
        *self
            .by_actual_dtype
            .entry(actual_dtype.to_string())
            .or_default() += 1;
        if fallback_to_q8 && requested.is_k_quant() {
            self.fallback_q8 += 1;
        }
    }

    pub fn add_quant_load(&mut self, d: Duration) {
        self.quant_load_secs += d.as_secs_f64();
    }

    #[must_use]
    pub fn is_mixed_k_quant(&self) -> bool {
        self.requested.is_k_quant() && self.fallback_q8 > 0
    }

    #[must_use]
    pub fn compression_ratio(&self) -> f64 {
        if self.dense_bytes == 0 {
            0.0
        } else {
            self.quantized_bytes as f64 / self.dense_bytes as f64
        }
    }

    pub fn print_summary(&self) {
        eprintln!(
            "VOXCPM_QUANT_STATS requested={} quantized={} skipped={} fallback_fp={} fallback_q8={} quant_load={:.3}s dense_mb={:.2} quant_mb={:.2} ratio={:.3} mixed_k={}",
            self.requested.as_str(),
            self.quantized,
            self.skipped,
            self.fallback_fp,
            self.fallback_q8,
            self.quant_load_secs,
            self.dense_bytes as f64 / (1024.0 * 1024.0),
            self.quantized_bytes as f64 / (1024.0 * 1024.0),
            self.compression_ratio(),
            self.is_mixed_k_quant(),
        );
        if !self.by_actual_dtype.is_empty() {
            eprintln!("VOXCPM_QUANT_STATS dtypes={:?}", self.by_actual_dtype);
        }
        if self.is_mixed_k_quant() {
            eprintln!(
                "VOXCPM_QUANT_WARN requested {} but {} layers fell back to q8_0 (block-size alignment)",
                self.requested.as_str(),
                self.fallback_q8
            );
        }
    }

    pub fn print_json_line(&self) {
        let dtypes: Vec<_> = self
            .by_actual_dtype
            .iter()
            .map(|(k, v)| format!("\"{k}\":{v}"))
            .collect();
        println!(
            "VOXCPM_QUANT_JSON {{\"requested\":\"{}\",\"quantized\":{},\"skipped\":{},\"fallback_fp\":{},\"fallback_q8\":{},\"quant_load_secs\":{:.6},\"dense_bytes\":{},\"quantized_bytes\":{},\"compression_ratio\":{:.4},\"mixed_k_quant\":{},\"dtypes\":{{{}}}}}",
            self.requested.as_str(),
            self.quantized,
            self.skipped,
            self.fallback_fp,
            self.fallback_q8,
            self.quant_load_secs,
            self.dense_bytes,
            self.quantized_bytes,
            self.compression_ratio(),
            self.is_mixed_k_quant(),
            dtypes.join(",")
        );
    }
}

/// Tracks module path while building layers for quant skip decisions.
#[derive(Debug, Clone)]
pub struct QuantBuildCtx {
    pub config: VoxCPMQuantConfig,
    pub module_path: String,
    stats: Rc<RefCell<QuantStats>>,
}

impl QuantBuildCtx {
    #[must_use]
    pub fn root(config: VoxCPMQuantConfig) -> Self {
        let stats = Rc::new(RefCell::new(QuantStats::new(config.weight)));
        Self {
            config,
            module_path: String::new(),
            stats,
        }
    }

    #[must_use]
    pub fn pp(&self, name: &str) -> Self {
        let module_path = if self.module_path.is_empty() {
            name.to_string()
        } else {
            format!("{}.{}", self.module_path, name)
        };
        Self {
            config: self.config.clone(),
            module_path,
            stats: Rc::clone(&self.stats),
        }
    }

    #[must_use]
    pub fn should_quantize(&self) -> bool {
        self.config.is_enabled() && !self.config.should_skip_module(&self.module_path)
    }

    pub fn stats(&self) -> QuantStats {
        self.stats.borrow().clone()
    }

    pub fn record_skipped(&self) {
        self.stats.borrow_mut().record_skipped();
    }

    pub fn record_fallback_fp(&self, dense_bytes: u64) {
        self.stats.borrow_mut().record_fallback_fp(dense_bytes);
    }

    pub fn record_quantized(
        &self,
        actual_dtype: &str,
        dense_bytes: u64,
        quantized_bytes: u64,
        fallback_to_q8: bool,
    ) {
        let requested = self.config.weight;
        self.stats.borrow_mut().record_quantized(
            requested,
            actual_dtype,
            dense_bytes,
            quantized_bytes,
            fallback_to_q8,
        );
    }

    pub fn add_quant_load(&self, d: Duration) {
        self.stats.borrow_mut().add_quant_load(d);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_quant_modes() {
        assert_eq!(
            "q4_k".parse::<VoxCPMWeightQuant>().unwrap(),
            VoxCPMWeightQuant::Q4K
        );
        assert_eq!(
            "none".parse::<VoxCPMWeightQuant>().unwrap(),
            VoxCPMWeightQuant::None
        );
    }

    #[test]
    fn default_skip_patterns() {
        let cfg = VoxCPMQuantConfig::with_weight(VoxCPMWeightQuant::Q4K);
        assert!(!cfg.should_skip_module("base_lm.layers.0.self_attn.o_proj"));
        assert!(cfg.should_skip_module("stop_head"));
        assert!(cfg.should_skip_module("fsq_layer.in_proj"));
        assert!(!cfg.should_skip_module("base_lm.layers.0.mlp.down_proj"));
    }

    #[test]
    fn quality_first_skips_attention_and_bridges() {
        let cfg = VoxCPMQuantConfig::quality_first(VoxCPMWeightQuant::Q8_0);
        assert!(cfg.should_skip_module("base_lm.layers.0.self_attn.q_proj"));
        assert!(cfg.should_skip_module("base_lm.layers.0.mlp.gate_proj"));
        assert!(cfg.should_skip_module("feat_encoder.in_proj"));
        assert!(!cfg.should_skip_module("base_lm.layers.0.mlp.down_proj"));
    }

    #[test]
    fn quant_stats_mixed_k_quant() {
        let mut stats = QuantStats::new(VoxCPMWeightQuant::Q4K);
        stats.record_quantized(VoxCPMWeightQuant::Q4K, "Q4K", 4096, 2304, false);
        stats.record_quantized(VoxCPMWeightQuant::Q4K, "Q8_0", 4096, 4096, true);
        assert!(stats.is_mixed_k_quant());
        assert_eq!(stats.fallback_q8, 1);
    }
}
