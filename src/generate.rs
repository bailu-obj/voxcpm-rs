use anyhow::{Ok, Result};
use candle_core::{pickle::read_all_with_key, DType, Device, DeviceLocation, Tensor};
use candle_nn::VarBuilder;
use std::collections::HashMap;

use crate::{
    audio_vae::AudioVAE,
    config::{AudioVaeConfig, VoxCPMConfig},
    models::VoxCPMModel,
    quant::VoxCPMQuantConfig,
    tokenizer::SingleChineseTokenizer,
    utils::device::get_device,
};

const DEFAULT_INFERENCE_TIMESTEPS: usize = 10;
pub const DEFAULT_STREAM_DECODE_LATENT_BATCH: usize = 12;
/// Bailu quality default: first streaming VAE decode batch, matched to the
/// steady-state batch (larger first decode, ~320 ms more TTFA than batch 4).
pub const DEFAULT_STREAM_DECODE_INITIAL_LATENT_BATCH: usize = 12;
/// Quality default: check stop head every latent (matches OpenBMB VoxCPM2).
const DEFAULT_STOP_CHECK_INTERVAL: usize = 1;
const DEFAULT_MIN_LEN: usize = 2;
const DEFAULT_MAX_LEN: usize = 500;
const DEFAULT_RETRY_BADCASE_RATIO: f64 = 6.0;

/// Why latent generation ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VoxCPMStopReason {
    /// Learned stop head predicted end-of-speech.
    StopHead,
    /// Hit `effective_max_len` without a stop-head trigger.
    MaxLen,
}

impl VoxCPMStopReason {
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::StopHead => "stop_head",
            Self::MaxLen => "max_len",
        }
    }
}

/// Diagnostics from the most recent generation on a [`VoxCPMGenerator`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VoxCPMGenerationDiagnostics {
    pub stop_reason: VoxCPMStopReason,
    pub latent_count: usize,
    pub target_text_tokens: usize,
    pub effective_max_len: usize,
}

/// Runtime options for model load (device, quantization).
///
/// Compute dtype is fixed to **F32**: quantized `QMatMul` accumulates in F32 anyway,
/// and dense F32 avoids per-layer cast kernels (measured faster on Apple Silicon).
#[derive(Debug, Clone, Default)]
pub struct VoxCPMGeneratorOptions {
    pub device: Option<Device>,
    pub device_id: Option<usize>,
    pub quant: VoxCPMQuantConfig,
    /// Fixed RNG seed for reproducible generation (compare / tests).
    pub seed: Option<u64>,
}

impl VoxCPMGeneratorOptions {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
}

#[derive(Debug, Clone, Copy)]
pub struct VoxCPMGenerationConfig {
    pub min_len: usize,
    pub max_len: usize,
    pub inference_timesteps: usize,
    pub cfg_value: f64,
    /// Fraction of Euler steps that run the 2× CFG batch (rest are positive-only).
    /// Default `1.0` matches official OpenBMB VoxCPM2 (CFG on every denoising step).
    pub cfg_full_fraction: f64,
    pub retry_badcase: bool,
    pub retry_badcase_ratio_threshold: f64,
    pub stream_decode_latent_batch: usize,
    /// First streaming VAE decode waits for this many latents (0 = auto: max(16, 2× batch)).
    /// May be smaller than `stream_decode_latent_batch` for lower TTFA.
    pub stream_decode_initial_latent_batch: usize,
    /// Run stop-head every N latents after `min_len` (1 = every step).
    pub stop_check_interval: usize,
}

impl Default for VoxCPMGenerationConfig {
    fn default() -> Self {
        Self::voice_clone()
    }
}

impl VoxCPMGenerationConfig {
    pub fn simple() -> Self {
        Self {
            min_len: DEFAULT_MIN_LEN,
            max_len: 100,
            inference_timesteps: DEFAULT_INFERENCE_TIMESTEPS,
            cfg_value: 2.0,
            cfg_full_fraction: 1.0,
            retry_badcase: true,
            retry_badcase_ratio_threshold: DEFAULT_RETRY_BADCASE_RATIO,
            stream_decode_latent_batch: DEFAULT_STREAM_DECODE_LATENT_BATCH,
            stream_decode_initial_latent_batch: DEFAULT_STREAM_DECODE_INITIAL_LATENT_BATCH,
            stop_check_interval: DEFAULT_STOP_CHECK_INTERVAL,
        }
    }

    /// Default production / Bailu balanced preset (also [`Default`]).
    ///
    /// Matches OpenBMB VoxCPM2 stop schedule plus Bailu streaming batches:
    /// `min_len=2`, `stop_check_interval=1`, ratio `6.0`, latent VAE batches `4` then `12`.
    pub fn voice_clone() -> Self {
        Self {
            min_len: DEFAULT_MIN_LEN,
            max_len: DEFAULT_MAX_LEN,
            inference_timesteps: DEFAULT_INFERENCE_TIMESTEPS,
            cfg_value: 2.0,
            cfg_full_fraction: 1.0,
            retry_badcase: true,
            retry_badcase_ratio_threshold: DEFAULT_RETRY_BADCASE_RATIO,
            stream_decode_latent_batch: DEFAULT_STREAM_DECODE_LATENT_BATCH,
            stream_decode_initial_latent_batch: DEFAULT_STREAM_DECODE_INITIAL_LATENT_BATCH,
            stop_check_interval: DEFAULT_STOP_CHECK_INTERVAL,
        }
    }

    /// Lower TTFA: fewer Euler steps, smaller VAE batches, less frequent stop checks.
    pub fn low_latency() -> Self {
        Self {
            min_len: DEFAULT_MIN_LEN,
            max_len: DEFAULT_MAX_LEN,
            inference_timesteps: 8,
            cfg_value: 2.0,
            cfg_full_fraction: 1.0,
            retry_badcase: true,
            retry_badcase_ratio_threshold: DEFAULT_RETRY_BADCASE_RATIO,
            stream_decode_latent_batch: 2,
            stream_decode_initial_latent_batch: 2,
            stop_check_interval: 2,
        }
    }

    /// Metal GPU RTF preset: fewer Euler steps, less frequent stop syncs, larger VAE batches.
    pub fn metal_rtf() -> Self {
        Self {
            min_len: DEFAULT_MIN_LEN,
            max_len: DEFAULT_MAX_LEN,
            inference_timesteps: 8,
            cfg_value: 2.0,
            cfg_full_fraction: 1.0,
            retry_badcase: true,
            retry_badcase_ratio_threshold: DEFAULT_RETRY_BADCASE_RATIO,
            stream_decode_latent_batch: DEFAULT_STREAM_DECODE_LATENT_BATCH,
            stream_decode_initial_latent_batch: DEFAULT_STREAM_DECODE_INITIAL_LATENT_BATCH,
            // Coarser stop polling trades tail precision for fewer host syncs.
            stop_check_interval: 4,
        }
    }

    pub(crate) fn effective_max_len(self, target_text_len: usize) -> usize {
        if self.retry_badcase {
            self.max_len
                .min((target_text_len as f64 * self.retry_badcase_ratio_threshold + 10.0) as usize)
        } else {
            self.max_len
        }
    }

    pub(crate) fn stream_decode_latent_batch(self) -> usize {
        self.stream_decode_latent_batch.max(1)
    }

    pub(crate) fn stream_decode_initial_latent_batch(self) -> usize {
        let batch = self.stream_decode_latent_batch();
        if self.stream_decode_initial_latent_batch > 0 {
            // Allow initial < steady-state batch so TTFA and RTF can be tuned independently.
            self.stream_decode_initial_latent_batch.max(1)
        } else {
            batch.saturating_mul(2).max(16)
        }
    }
}

/// Main generator for VoxCPM text-to-speech
pub struct VoxCPMGenerator {
    voxcpm: VoxCPMModel,
    prompt_cache: Option<HashMap<String, Tensor>>,
    sample_rate: usize,
    model_name: String,
    generation_seed: Option<u64>,
}

/// Resolve main-model load dtype (no user-facing option).
///
/// Quant path: **F32 activations** — `QMatMul` accumulates in F32, so half-precision
/// activations only add per-layer cast churn (measured ~3% RTF slower on Apple Silicon).
/// Dense path (`quant=none`): keep the checkpoint dtype (BF16 for current VoxCPM
/// checkpoints); F32-declaring configs downcast to F16 on GPU (weight-stream-bound at
/// decode batch sizes).
fn resolve_model_dtype(quant_enabled: bool, cfg_dtype: &str, device: &Device) -> DType {
    if quant_enabled {
        return DType::F32;
    }
    match cfg_dtype.trim().to_lowercase().as_str() {
        "bfloat16" | "bf16" => DType::BF16,
        "float16" | "half" | "f16" => DType::F16,
        _ => match device.location() {
            DeviceLocation::Cpu => DType::F32,
            _ => DType::F16,
        },
    }
}

impl VoxCPMGenerator {
    /// Initialize VoxCPM model from path (backward-compatible defaults).
    pub fn new(path: &str, device: Option<&Device>) -> Result<Self> {
        let mut options = VoxCPMGeneratorOptions::default();
        options.device = device.cloned();
        Self::new_with_options(path, &options)
    }

    /// Initialize VoxCPM model with full runtime options.
    pub fn new_with_options(path: &str, options: &VoxCPMGeneratorOptions) -> Result<Self> {
        let device = get_device(options.device.as_ref(), options.device_id);
        let config_path = path.to_string() + "/config.json";
        let config: VoxCPMConfig = serde_json::from_slice(&std::fs::read(config_path)?)?;

        // Load VAE weights: prefer mmap safetensors (e.g. `*vae*.safetensors`), else PyTorch `.pth`.
        let vae_safetensors = find_vae_safetensors(path)?;
        let vb_vae = if !vae_safetensors.is_empty() {
            let vb = unsafe {
                VarBuilder::from_mmaped_safetensors(&vae_safetensors, DType::F32, &device)?
            };
            vb
        } else {
            let model_list = find_type_files(path, "pth")?;
            let mut dict_to_hashmap = HashMap::new();
            for m in model_list {
                let dict = read_all_with_key(m, Some("state_dict"))?;
                for (k, v) in dict {
                    dict_to_hashmap.insert(k, v);
                }
            }
            VarBuilder::from_tensors(dict_to_hashmap, DType::F32, &device)
        };
        let audio_config = match config.audio_vae_config.clone() {
            Some(config) => config,
            None => AudioVaeConfig {
                encoder_dim: 128,
                encoder_rates: vec![2, 5, 8, 8],
                latent_dim: 64,
                decoder_dim: 1536,
                decoder_rates: vec![8, 8, 5, 2],
                sample_rate: 16000,
                out_sample_rate: None,
                sr_bin_boundaries: None,
            },
        };

        let out_sample_rate = audio_config
            .out_sample_rate
            .unwrap_or(audio_config.sample_rate);

        let model_name = if config.is_voxcpm2() {
            "VoxCPM2".to_string()
        } else if audio_config.sample_rate == 16000 {
            "VoxCPM".to_string()
        } else {
            "VoxCPM1.5".to_string()
        };

        let cond_type = if config.is_voxcpm2() {
            Some("scale_bias".to_string())
        } else {
            None
        };

        let audio_vae = AudioVAE::new(
            vb_vae,
            audio_config.encoder_dim,
            audio_config.encoder_rates.clone(),
            Some(audio_config.latent_dim),
            audio_config.decoder_dim,
            audio_config.decoder_rates.clone(),
            audio_config.sample_rate,
            out_sample_rate,
            audio_config.sr_bin_boundaries.clone(),
            cond_type,
        )?;

        let m_dtype =
            resolve_model_dtype(options.quant.is_enabled(), config.dtype.as_str(), &device);

        // Load main model weights (. bin or .safetensors)
        let model_list = find_type_files(path, "bin")?;
        let vb_voxcpm = if model_list.is_empty() {
            let model_list = find_type_files(path, "safetensors")?;
            let main_st: Vec<String> = model_list
                .into_iter()
                .filter(|p| !p.to_lowercase().contains("vae"))
                .collect();
            unsafe { VarBuilder::from_mmaped_safetensors(&main_st, m_dtype, &device)? }
        } else {
            let mut dict_to_hashmap = HashMap::new();
            for m in model_list {
                let dict = read_all_with_key(m, Some("state_dict"))?;
                for (k, v) in dict {
                    dict_to_hashmap.insert(k, v);
                }
            }
            VarBuilder::from_tensors(dict_to_hashmap, m_dtype, &device)
        };

        let tokenizer = SingleChineseTokenizer::new(path)?;
        let voxcpm = VoxCPMModel::new(
            vb_voxcpm,
            config,
            tokenizer,
            audio_vae,
            options.quant.clone(),
        )?;

        Ok(Self {
            voxcpm,
            prompt_cache: None,
            sample_rate: out_sample_rate,
            model_name,
            generation_seed: options.seed.or_else(|| parse_seed_from_env()),
        })
    }

    /// Diagnostics from the most recently completed batch or fully consumed stream.
    #[must_use]
    pub fn last_diagnostics(&self) -> Option<VoxCPMGenerationDiagnostics> {
        self.voxcpm.last_diagnostics()
    }

    fn apply_generation_seed(&self) -> Result<()> {
        if let Some(seed) = self.generation_seed {
            match self.voxcpm.device().location() {
                DeviceLocation::Metal { .. } | DeviceLocation::Cuda { .. } => {
                    self.voxcpm.device().set_seed(seed)?;
                }
                DeviceLocation::Cpu => {}
            }
        }
        Ok(())
    }

    /// Build prompt cache for batch processing
    pub fn build_prompt_cache(
        &mut self,
        prompt_text: String,
        prompt_wav_path: String,
    ) -> Result<()> {
        let cache = self
            .voxcpm
            .build_prompt_cache(prompt_text, prompt_wav_path)?;
        self.prompt_cache = Some(cache);
        Ok(())
    }

    /// Generate using cached prompt
    pub fn generate_with_config(
        &mut self,
        target_text: String,
        config: VoxCPMGenerationConfig,
    ) -> Result<Tensor> {
        self.apply_generation_seed()?;
        match self.prompt_cache.as_ref() {
            Some(cache) => self
                .voxcpm
                .generate_with_prompt_cache(target_text, cache, config),
            None => self.voxcpm.generate(target_text, None, None, config),
        }
    }

    /// Simple generation with default parameters
    pub fn generate_simple(&mut self, target_text: String) -> Result<Tensor> {
        self.generate_with_config(target_text, VoxCPMGenerationConfig::simple())
    }

    /// Simple streaming generation with default parameters
    pub fn generate_stream_simple(
        &mut self,
        target_text: String,
    ) -> Result<Box<dyn Iterator<Item = Result<Tensor>> + '_>> {
        let iter =
            self.generate_stream_with_config(target_text, VoxCPMGenerationConfig::simple())?;
        Ok(Box::new(iter))
    }

    pub fn generate_stream_with_config(
        &mut self,
        target_text: String,
        config: VoxCPMGenerationConfig,
    ) -> Result<Box<dyn Iterator<Item = Result<Tensor>> + '_>> {
        self.apply_generation_seed()?;
        match self.prompt_cache.as_ref() {
            Some(cache) => Ok(Box::new(self.voxcpm.generate_stream_with_prompt_cache(
                target_text,
                cache,
                config,
            )?) as Box<dyn Iterator<Item = Result<Tensor>>>),
            None => {
                let iter = self
                    .voxcpm
                    .generate_stream(target_text, None, None, config)?;
                Ok(Box::new(iter) as Box<dyn Iterator<Item = Result<Tensor>>>)
            }
        }
    }

    /// Streaming generation returning WAV bytes (Vec<u8>)
    pub fn generate_wav_stream_simple(
        &mut self,
        target_text: String,
    ) -> Result<Box<dyn Iterator<Item = Result<Vec<u8>>> + '_>> {
        let sample_rate = self.sample_rate as u32;
        let stream = self.generate_stream_simple(target_text)?;
        let iter = stream.map(move |res| match res {
            std::result::Result::Ok(tensor) => crate::utils::audio::to_wav(&tensor, sample_rate),
            Err(e) => Err(e),
        });
        Ok(Box::new(iter))
    }

    /// Streaming generation using prompt cache returning WAV bytes (Vec<u8>)
    pub fn generate_wav_stream_with_config(
        &mut self,
        target_text: String,
        config: VoxCPMGenerationConfig,
    ) -> Result<Box<dyn Iterator<Item = Result<Vec<u8>>> + '_>> {
        let sample_rate = self.sample_rate as u32;
        let stream = self.generate_stream_with_config(target_text, config)?;
        let iter = stream.map(move |res| match res {
            std::result::Result::Ok(tensor) => crate::utils::audio::to_wav(&tensor, sample_rate),
            Err(e) => Err(e),
        });
        Ok(Box::new(iter))
    }

    /// Streaming generation using prompt cache returning PCM samples (Vec<i16>)
    pub fn generate_pcm_stream_with_config(
        &mut self,
        target_text: String,
        config: VoxCPMGenerationConfig,
    ) -> Result<Box<dyn Iterator<Item = Result<Vec<i16>>> + '_>> {
        let stream = self.generate_stream_with_config(target_text, config)?;
        Ok(Box::new(PcmStreamIter {
            inner: stream,
            normalizer: PcmNormalizerSlot::Owned(crate::utils::audio::StreamPcmNormalizer::new()),
        }))
    }

    /// Streaming PCM generation that shares a caller-owned
    /// [`StreamPcmNormalizer`](crate::utils::audio::StreamPcmNormalizer), so PCM gain stays
    /// continuous across successive streams (e.g. text segments of one reply).
    pub fn generate_pcm_stream_with_normalizer<'a>(
        &'a mut self,
        target_text: String,
        config: VoxCPMGenerationConfig,
        normalizer: &'a mut crate::utils::audio::StreamPcmNormalizer,
    ) -> Result<Box<dyn Iterator<Item = Result<Vec<i16>>> + 'a>> {
        let stream = self.generate_stream_with_config(target_text, config)?;
        Ok(Box::new(PcmStreamIter {
            inner: stream,
            normalizer: PcmNormalizerSlot::Shared(normalizer),
        }))
    }

    /// Get sample rate
    pub fn sample_rate(&self) -> usize {
        self.sample_rate
    }

    /// Get model name
    pub fn model_name(&self) -> &str {
        &self.model_name
    }

    /// Save audio tensor to WAV file
    pub fn save_wav(&self, audio: &Tensor, path: &str) -> Result<()> {
        crate::utils::audio::save_wav(audio, path, self.sample_rate as u32)
    }

    pub fn to_wav(&self, audio: &Tensor) -> Result<Vec<u8>> {
        crate::utils::audio::to_wav(audio, self.sample_rate as u32)
    }

    /// Convert audio tensor to PCM samples (i16)
    pub fn to_pcm(&self, audio: &Tensor) -> Result<Vec<i16>> {
        crate::utils::audio::to_pcm(audio)
    }

    /// Load-time quantization statistics (layer coverage, bytes, fallbacks).
    #[must_use]
    pub fn quant_stats(&self) -> crate::quant::QuantStats {
        self.voxcpm.quant_stats()
    }
}

struct PcmStreamIter<'a> {
    inner: Box<dyn Iterator<Item = Result<Tensor>> + 'a>,
    normalizer: PcmNormalizerSlot<'a>,
}

/// Per-stream PCM normalizer: fresh per stream by default, or caller-shared so gain stays
/// continuous across streams of one logical utterance.
enum PcmNormalizerSlot<'a> {
    Owned(crate::utils::audio::StreamPcmNormalizer),
    Shared(&'a mut crate::utils::audio::StreamPcmNormalizer),
}

impl PcmNormalizerSlot<'_> {
    fn get_mut(&mut self) -> &mut crate::utils::audio::StreamPcmNormalizer {
        match self {
            Self::Owned(normalizer) => normalizer,
            Self::Shared(normalizer) => normalizer,
        }
    }
}

impl Iterator for PcmStreamIter<'_> {
    type Item = Result<Vec<i16>>;

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next().map(|res| {
            res.and_then(|tensor| {
                crate::utils::audio::to_pcm_stream_chunk_with_normalizer(
                    &tensor,
                    self.normalizer.get_mut(),
                )
            })
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn voice_clone_stop_schedule_matches_upstream_quality() {
        let cfg = VoxCPMGenerationConfig::voice_clone();
        assert_eq!(cfg.min_len, 2);
        assert_eq!(cfg.stop_check_interval, 1);
        assert!((cfg.retry_badcase_ratio_threshold - 6.0).abs() < 1e-9);
        assert_eq!(cfg.stream_decode_initial_latent_batch, 12);
        assert_eq!(cfg.stream_decode_latent_batch, 12);
        assert_eq!(cfg.inference_timesteps, 10);
        assert!((cfg.cfg_full_fraction - 1.0).abs() < 1e-9);
    }

    #[test]
    fn effective_max_len_scales_with_tokens_until_cap() {
        let cfg = VoxCPMGenerationConfig::voice_clone();
        assert_eq!(cfg.effective_max_len(10), 70); // 6*10+10
        assert_eq!(cfg.effective_max_len(100), 500); // capped
        assert_eq!(cfg.effective_max_len(200), 500);
    }

    #[test]
    fn stop_reason_labels() {
        assert_eq!(VoxCPMStopReason::StopHead.as_str(), "stop_head");
        assert_eq!(VoxCPMStopReason::MaxLen.as_str(), "max_len");
    }

    #[test]
    fn shared_normalizer_keeps_gain_across_streams() -> Result<()> {
        use candle_core::{Device, Tensor};

        fn chunk(value: f32) -> Result<Tensor> {
            Ok(Tensor::from_slice(&[value, -value], 2, &Device::Cpu)?.unsqueeze(0)?)
        }

        let mut normalizer = crate::utils::audio::StreamPcmNormalizer::new();
        // First "stream" peaks at 2.0 → scale 32767/2.
        let first: Vec<Vec<i16>> = PcmStreamIter {
            inner: Box::new(vec![chunk(2.0)].into_iter()),
            normalizer: PcmNormalizerSlot::Shared(&mut normalizer),
        }
        .collect::<Result<_>>()?;
        assert_eq!(first, vec![vec![32767, -32767]]);
        assert!((normalizer.running_peak() - 2.0).abs() < 1e-6);

        // Second "stream" peaks at 0.5 but keeps the first stream's gain (no reset).
        let second: Vec<Vec<i16>> = PcmStreamIter {
            inner: Box::new(vec![chunk(0.5)].into_iter()),
            normalizer: PcmNormalizerSlot::Shared(&mut normalizer),
        }
        .collect::<Result<_>>()?;
        assert_eq!(second, vec![vec![8192, -8192]]);

        // Owned slot (default path) still resets per stream.
        let third: Vec<Vec<i16>> = PcmStreamIter {
            inner: Box::new(vec![chunk(0.5)].into_iter()),
            normalizer: PcmNormalizerSlot::Owned(crate::utils::audio::StreamPcmNormalizer::new()),
        }
        .collect::<Result<_>>()?;
        assert_eq!(third, vec![vec![16384, -16384]]);
        Ok(())
    }
}

/// Find files with specific extension in directory
fn find_type_files(path: &str, file_type: &str) -> Result<Vec<String>> {
    let mut files = Vec::new();
    for entry in std::fs::read_dir(path)? {
        let entry = entry?;
        let path = entry.path();
        if let Some(ext) = path.extension() {
            if ext == file_type {
                files.push(path.to_str().unwrap().to_string());
            }
        }
    }
    Ok(files)
}

/// Safetensors files whose path contains `vae` (case-insensitive), for mmap VAE load.
fn find_vae_safetensors(path: &str) -> Result<Vec<String>> {
    Ok(find_type_files(path, "safetensors")?
        .into_iter()
        .filter(|p| p.to_lowercase().contains("vae"))
        .collect())
}

fn parse_seed_from_env() -> Option<u64> {
    std::env::var("VOXCPM_SEED")
        .ok()
        .and_then(|s| s.parse().ok())
}

/// Default seed for FP-vs-quant comparison runs.
pub const COMPARE_FP_DEFAULT_SEED: u64 = 42;
