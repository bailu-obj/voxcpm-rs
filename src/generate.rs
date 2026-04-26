use anyhow::{Ok, Result};
use candle_core::{pickle::read_all_with_key, DType, Device, Tensor};
use candle_nn::VarBuilder;
use std::collections::HashMap;

use crate::{
    audio_vae::AudioVAE,
    config::{AudioVaeConfig, VoxCPMConfig},
    models::VoxCPMModel,
    tokenizer::SingleChineseTokenizer,
    utils::device::{get_compute_dtype, get_device, get_vae_compute_dtype},
};

const DEFAULT_INFERENCE_TIMESTEPS: usize = 10;
pub const DEFAULT_STREAM_DECODE_LATENT_BATCH: usize = 4;

#[derive(Debug, Clone, Copy)]
pub struct VoxCPMGenerationConfig {
    pub min_len: usize,
    pub max_len: usize,
    pub inference_timesteps: usize,
    pub cfg_value: f64,
    pub retry_badcase: bool,
    pub retry_badcase_ratio_threshold: f64,
    pub stream_decode_latent_batch: usize,
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
}

impl VoxCPMGenerator {
    /// Initialize VoxCPM model from path
    ///
    /// # Arguments
    /// * `path` - Path to model directory
    /// * `device` - Optional device (None for auto-detect)
    /// * `dtype` - Optional data type (None prefers F16 on GPU when the config is F32)
    pub fn new(path: &str, device: Option<&Device>, dtype: Option<DType>) -> Result<Self> {
        let device = &get_device(device);
        let config_path = path.to_string() + "/config.json";
        let config: VoxCPMConfig = serde_json::from_slice(&std::fs::read(config_path)?)?;

        // Load VAE weights (PyTorch . pth files)
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

        let vae_dtype = get_vae_compute_dtype(dtype, vae_dtype, device);
        let vb_vae = VarBuilder::from_tensors(dict_to_hashmap, vae_dtype, device);
        let audio_config = match config.audio_vae_config.clone() {
            Some(config) => config,
            None => AudioVaeConfig {
                encoder_dim: 128,
                encoder_rates: vec![2, 5, 8, 8],
                latent_dim: 64,
                decoder_dim: 1536,
                decoder_rates: vec![8, 8, 5, 2],
                sample_rate: 16000,
            },
        };

        let model_name = if audio_config.sample_rate == 16000 {
            "VoxCPM".to_string()
        } else {
            "VoxCPM1.5".to_string()
        };

        let audio_vae = AudioVAE::new(
            vb_vae,
            audio_config.encoder_dim,
            audio_config.encoder_rates.clone(),
            Some(audio_config.latent_dim),
            audio_config.decoder_dim,
            audio_config.decoder_rates.clone(),
            audio_config.sample_rate,
        )?;

        let cfg_dtype = config.dtype.as_str();
        let m_dtype = get_compute_dtype(dtype, cfg_dtype, device);

        // Load main model weights (. bin or .safetensors)
        let model_list = find_type_files(path, "bin")?;
        let vb_voxcpm = if model_list.is_empty() {
            let model_list = find_type_files(path, "safetensors")?;
            unsafe { VarBuilder::from_mmaped_safetensors(&model_list, m_dtype, device)? }
        } else {
            dict_to_hashmap = HashMap::new();
            for m in model_list {
                let dict = read_all_with_key(m, Some("state_dict"))?;
                for (k, v) in dict {
                    dict_to_hashmap.insert(k, v);
                }
            }
            VarBuilder::from_tensors(dict_to_hashmap, m_dtype, device)
        };

        let tokenizer = SingleChineseTokenizer::new(path)?;
        let voxcpm = VoxCPMModel::new(vb_voxcpm, config, tokenizer, audio_vae)?;

        Ok(Self {
            voxcpm,
            prompt_cache: None,
            sample_rate: audio_config.sample_rate,
            model_name,
        })
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
