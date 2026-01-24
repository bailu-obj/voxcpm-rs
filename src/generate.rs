use anyhow::{Ok, Result};
use candle_core::{pickle::read_all_with_key, DType, Device, Tensor};
use candle_nn::VarBuilder;
use std::collections::HashMap;

use crate::{
    audio_vae::AudioVAE,
    config::{AudioVaeConfig, VoxCPMConfig},
    models::VoxCPMModel,
    tokenizer::SingleChineseTokenizer,
    utils::device::{get_device, get_dtype},
};

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
    /// * `dtype` - Optional data type (None for config default)
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
        let m_dtype = get_dtype(dtype, cfg_dtype);

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
    pub fn generate_use_prompt_cache(
        &mut self,
        target_text: String,
        min_len: usize,
        max_len: usize,
        inference_timesteps: usize,
        cfg_value: f64,
        retry_badcase: bool,
        retry_badcase_ratio_threshold: f64,
    ) -> Result<Tensor> {
        match self.prompt_cache.take() {
            Some(cache) => {
                let audio = self.voxcpm.generate_with_prompt_cache(
                    target_text,
                    cache.clone(),
                    min_len,
                    max_len,
                    inference_timesteps,
                    cfg_value,
                    retry_badcase,
                    retry_badcase_ratio_threshold,
                )?;
                self.prompt_cache = Some(cache);
                Ok(audio)
            }
            None => self.generate_simple(target_text),
        }
    }

    /// Simple generation with default parameters
    pub fn generate_simple(&mut self, target_text: String) -> Result<Tensor> {
        self.inference(target_text, None, None, 2, 100, 10, 2.0, 6.0)
    }

    /// Simple streaming generation with default parameters
    pub fn generate_stream_simple(
        &mut self,
        target_text: String,
    ) -> Result<Box<dyn Iterator<Item = Result<Tensor>> + '_>> {
        let iter = self.inference_stream(target_text, None, None, 2, 100, 10, 2.0, 6.0)?;
        Ok(Box::new(iter))
    }

    pub fn generate_stream_use_prompt_cache(
        &mut self,
        target_text: String,
        min_len: usize,
        max_len: usize,
        inference_timesteps: usize,
        cfg_value: f64,
        retry_badcase: bool,
        retry_badcase_ratio_threshold: f64,
    ) -> Result<Box<dyn Iterator<Item = Result<Tensor>> + '_>> {
        match self.prompt_cache.take() {
            Some(cache) => {
                let iter = self.voxcpm.generate_stream_with_prompt_cache(
                    target_text,
                    cache.clone(),
                    min_len,
                    max_len,
                    inference_timesteps,
                    cfg_value,
                    retry_badcase,
                    retry_badcase_ratio_threshold,
                )?;
                self.prompt_cache = Some(cache);
                Ok(Box::new(iter) as Box<dyn Iterator<Item = Result<Tensor>>>)
            }
            None => {
                let iter = self.generate_stream_simple(target_text)?;
                Ok(Box::new(iter) as Box<dyn Iterator<Item = Result<Tensor>>>)
            }
        }
    }

    /// Full inference with all parameters
    ///
    /// # Arguments
    /// * `target_text` - Text to synthesize
    /// * `prompt_text` - Optional reference text for voice cloning
    /// * `prompt_wav_path` - Optional reference audio path
    /// * `min_len` - Minimum generation length
    /// * `max_len` - Maximum generation length
    /// * `inference_timesteps` - Number of diffusion steps (higher = better quality)
    /// * `cfg_value` - Classifier-free guidance value
    /// * `retry_badcase_ratio_threshold` - Quality threshold
    pub fn inference(
        &mut self,
        target_text: String,
        prompt_text: Option<String>,
        prompt_wav_path: Option<String>,
        min_len: usize,
        max_len: usize,
        inference_timesteps: usize,
        cfg_value: f64,
        retry_badcase_ratio_threshold: f64,
    ) -> Result<Tensor> {
        self.voxcpm.generate(
            target_text,
            prompt_text,
            prompt_wav_path,
            min_len,
            max_len,
            inference_timesteps,
            cfg_value,
            retry_badcase_ratio_threshold,
        )
    }

    /// Full streaming inference with all parameters
    pub fn inference_stream(
        &mut self,
        target_text: String,
        prompt_text: Option<String>,
        prompt_wav_path: Option<String>,
        min_len: usize,
        max_len: usize,
        inference_timesteps: usize,
        cfg_value: f64,
        retry_badcase_ratio_threshold: f64,
    ) -> Result<Box<dyn Iterator<Item = Result<Tensor>> + '_>> {
        let iter = self.voxcpm.generate_stream(
            target_text,
            prompt_text,
            prompt_wav_path,
            min_len,
            max_len,
            inference_timesteps,
            cfg_value,
            retry_badcase_ratio_threshold,
        )?;
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
