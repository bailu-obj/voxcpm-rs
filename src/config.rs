use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub struct VoxRopeScalingConfig {
    pub r#type: String,
    pub long_factor: Vec<f32>,
    pub short_factor: Vec<f32>,
    pub original_max_position_embeddings: usize,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub struct VoxMiniCPM4Config {
    pub bos_token_id: u32,
    pub eos_token_id: u32,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub max_position_embeddings: usize,
    pub num_attention_heads: usize,
    pub num_hidden_layers: usize,
    pub num_key_value_heads: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f32,
    pub rope_scaling: VoxRopeScalingConfig,
    pub vocab_size: usize,
    pub scale_emb: f32,
    pub dim_model_base: usize,
    pub scale_depth: f32,
    pub use_mup: bool,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub struct VoxCPMEncoderConfig {
    pub hidden_dim: usize,
    pub ffn_dim: usize,
    pub num_heads: usize,
    pub num_layers: usize,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub struct CfmConfig {
    pub sigma_min: f32,
    pub solver: String,
    pub t_scheduler: String,
    pub inference_cfg_rate: f32,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub struct VoxCPMDitConfig {
    pub hidden_dim: usize,
    pub ffn_dim: usize,
    pub num_heads: usize,
    pub num_layers: usize,
    pub cfm_config: CfmConfig,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub struct AudioVaeConfig {
    pub encoder_dim: usize,
    pub encoder_rates: Vec<usize>,
    pub latent_dim: usize,
    pub decoder_dim: usize,
    pub decoder_rates: Vec<usize>,
    pub sample_rate: usize,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub struct VoxCPMConfig {
    pub lm_config: VoxMiniCPM4Config,
    pub patch_size: usize,
    pub feat_dim: usize,
    pub scalar_quantization_latent_dim: usize,
    pub scalar_quantization_scale: usize,
    pub residual_lm_num_layers: usize,
    pub encoder_config: VoxCPMEncoderConfig,
    pub dit_config: VoxCPMDitConfig,
    pub audio_vae_config: Option<AudioVaeConfig>,
    pub max_length: usize,
    pub dtype: String,
}
