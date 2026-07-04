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
    utils::device::{get_compute_dtype, get_device, get_quant_compute_dtype, get_vae_compute_dtype},
};

const DEFAULT_INFERENCE_TIMESTEPS: usize = 8;
pub const DEFAULT_STREAM_DECODE_LATENT_BATCH: usize = 8;
const DEFAULT_STOP_CHECK_INTERVAL: usize = 4;

/// Runtime options for model load (device, dtype, quantization).
#[derive(Debug, Clone, Default)]
pub struct VoxCPMGeneratorOptions {
    pub device: Option<Device>,
    pub device_id: Option<usize>,
    pub dtype: Option<DType>,
    pub vae_dtype: Option<DType>,
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
    pub retry_badcase: bool,
    pub retry_badcase_ratio_threshold: f64,
    pub stream_decode_latent_batch: usize,
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
            min_len: 2,
            max_len: 100,
            inference_timesteps: DEFAULT_INFERENCE_TIMESTEPS,
            cfg_value: 2.0,
            retry_badcase: true,
            retry_badcase_ratio_threshold: 6.0,
            stream_decode_latent_batch: DEFAULT_STREAM_DECODE_LATENT_BATCH,
            stop_check_interval: DEFAULT_STOP_CHECK_INTERVAL,
        }
    }

    pub fn voice_clone() -> Self {
        Self {
            min_len: 5,
            max_len: 500,
            inference_timesteps: DEFAULT_INFERENCE_TIMESTEPS,
            cfg_value: 2.0,
            retry_badcase: true,
            retry_badcase_ratio_threshold: 3.0,
            stream_decode_latent_batch: DEFAULT_STREAM_DECODE_LATENT_BATCH,
            stop_check_interval: DEFAULT_STOP_CHECK_INTERVAL,
        }
    }

    /// Lower TTFA: fewer Euler steps, smaller VAE batches, less frequent stop checks.
    pub fn low_latency() -> Self {
        Self {
            min_len: 5,
            max_len: 500,
            inference_timesteps: 8,
            cfg_value: 2.0,
            retry_badcase: true,
            retry_badcase_ratio_threshold: 3.0,
            stream_decode_latent_batch: 2,
            stop_check_interval: 2,
        }
    }

    /// Adaptive Euler steps from target text length (short prompts use fewer steps).
    pub fn adaptive_for_text_len(target_text_len: usize) -> Self {
        let mut cfg = Self::voice_clone();
        cfg.inference_timesteps = match target_text_len {
            0..=20 => 6,
            21..=60 => 8,
            _ => 8,
        };
        cfg
    }

    /// Metal GPU RTF preset: fewer Euler steps, less frequent stop syncs, larger VAE batches.
    pub fn metal_rtf() -> Self {
        Self {
            min_len: 5,
            max_len: 500,
            inference_timesteps: 8,
            cfg_value: 2.0,
            retry_badcase: true,
            retry_badcase_ratio_threshold: 3.0,
            stream_decode_latent_batch: 8,
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
}

/// Main generator for VoxCPM text-to-speech
pub struct VoxCPMGenerator {
    voxcpm: VoxCPMModel,
    prompt_cache: Option<HashMap<String, Tensor>>,
    sample_rate: usize,
    model_name: String,
    generation_seed: Option<u64>,
}

impl VoxCPMGenerator {
    /// Initialize VoxCPM model from path (backward-compatible defaults).
    pub fn new(path: &str, device: Option<&Device>, dtype: Option<DType>) -> Result<Self> {
        let mut options = VoxCPMGeneratorOptions::default();
        options.device = device.cloned();
        options.dtype = dtype;
        Self::new_with_options(path, &options)
    }

    /// Initialize VoxCPM model with full runtime options.
    pub fn new_with_options(path: &str, options: &VoxCPMGeneratorOptions) -> Result<Self> {
        let device = get_device(options.device.as_ref(), options.device_id);
        let dtype = options.dtype;
        let vae_dtype_override = options.vae_dtype;
        let config_path = path.to_string() + "/config.json";
        let config: VoxCPMConfig = serde_json::from_slice(&std::fs::read(config_path)?)?;

        // Load VAE weights: prefer mmap safetensors (e.g. `*vae*.safetensors`), else PyTorch `.pth`.
        let vae_safetensors = find_vae_safetensors(path)?;
        let (vb_vae, _vae_dtype) = if !vae_safetensors.is_empty() {
            let vae_dtype = get_vae_compute_dtype(vae_dtype_override, DType::F32, &device);
            let vb = unsafe { VarBuilder::from_mmaped_safetensors(&vae_safetensors, vae_dtype, &device)? };
            (vb, vae_dtype)
        } else {
            let model_list = find_type_files(path, "pth")?;
            let mut dict_to_hashmap = HashMap::new();
            let mut vae_dtype = DType::F32;

            for m in model_list {
                let dict = read_all_with_key(m, Some("state_dict"))?;
                vae_dtype = dict[0].1.dtype();
                for (k, v) in dict {
                    dict_to_hashmap.insert(k, v);
                }
            }

            let vae_dtype = get_vae_compute_dtype(vae_dtype_override, vae_dtype, &device);
            (
                VarBuilder::from_tensors(dict_to_hashmap, vae_dtype, &device),
                vae_dtype,
            )
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

        let cfg_dtype = config.dtype.as_str();
        let m_dtype = if options.quant.is_enabled() {
            get_quant_compute_dtype(dtype, cfg_dtype, &device)
        } else {
            get_compute_dtype(dtype, cfg_dtype, &device)
        };

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
            generation_seed: options
                .seed
                .or_else(|| parse_seed_from_env()),
        })
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
        let iter = stream.map(move |res| match res {
            std::result::Result::Ok(tensor) => crate::utils::audio::to_pcm_stream_chunk(&tensor),
            Err(e) => Err(e),
        });
        Ok(Box::new(iter))
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
