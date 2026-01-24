use anyhow::Result;
use candle_core::{DType, Device, IndexOp, Tensor, D};
use candle_nn::{linear, linear_no_bias, Linear, Module, VarBuilder};
use candle_transformers::models::deepseek2::SplitOp;
use std::{cmp::max, collections::HashMap};

use crate::{
    audio_vae::AudioVAE,
    config::{CfmConfig, VoxCPMConfig, VoxMiniCPM4Config},
    minicpm4::MiniCPMModel,
    tokenizer::SingleChineseTokenizer,
    utils::audio::load_audio_with_resample,
    utils::tensor::linspace,
};

pub struct ScalarQuantizationLayer {
    scale: usize,
    in_proj: Linear,
    out_proj: Linear,
}

impl ScalarQuantizationLayer {
    pub fn new(
        vb: VarBuilder,
        in_dim: usize,
        out_dim: usize,
        laten_dim: usize,
        scale: usize,
    ) -> Result<Self> {
        let in_proj = linear(in_dim, laten_dim, vb.pp("in_proj"))?;
        let out_proj = linear(laten_dim, out_dim, vb.pp("out_proj"))?;
        Ok(Self {
            scale,
            in_proj,
            out_proj,
        })
    }
    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = self.in_proj.forward(xs)?;
        let xs = xs.tanh()?;
        let xs = xs
            .affine(self.scale as f64, 0.0)?
            .round()?
            .affine(1.0 / self.scale as f64, 0.0)?;
        let xs = self.out_proj.forward(&xs)?;
        Ok(xs)
    }
}

pub struct SinusoidalPosEmb {
    dim: usize,
}

impl SinusoidalPosEmb {
    pub fn new(dim: usize) -> Result<Self> {
        assert_eq!(dim % 2, 0, "SinusoidalPosEmb requires dim to be even");
        Ok(Self { dim })
    }
    pub fn forward(&self, x: &Tensor, scale: usize) -> Result<Tensor> {
        let x = if x.rank() < 1 {
            x.unsqueeze(0)?
        } else {
            x.clone()
        };
        let half_dim = self.dim / 2;
        let dif = 10000.0_f64.ln() / (half_dim - 1) as f64;
        let emb = Tensor::arange(0.0, half_dim as f32, x.device())?
            .affine(-dif, 0.0)?
            .exp()?
            .to_dtype(x.dtype())?;

        let emb = x
            .unsqueeze(1)?
            .contiguous()?
            .affine(scale as f64, 0.0)?
            .matmul(&emb.unsqueeze(0)?.contiguous()?)?;
        let emb = Tensor::cat(&[emb.sin()?, emb.cos()?], D::Minus1)?;
        Ok(emb)
    }
}

pub struct TimestepEmbedding {
    linear_1: Linear,
    linear_2: Linear,
}

impl TimestepEmbedding {
    pub fn new(
        vb: VarBuilder,
        in_channels: usize,
        time_embed_dim: usize,
        out_dim: Option<usize>,
    ) -> Result<Self> {
        let linear_1 = linear(in_channels, time_embed_dim, vb.pp("linear_1"))?;
        let time_embed_dim_out = if let Some(out_dim) = out_dim {
            out_dim
        } else {
            time_embed_dim
        };
        let linear_2 = linear(time_embed_dim, time_embed_dim_out, vb.pp("linear_2"))?;
        Ok(Self { linear_1, linear_2 })
    }

    pub fn forward(&self, sample: &Tensor) -> Result<Tensor> {
        let sample = self.linear_1.forward(sample)?.silu()?;
        let sample = self.linear_2.forward(&sample)?;
        Ok(sample)
    }
}

pub struct VoxCPMLocDiT {
    in_proj: Linear,
    cond_proj: Linear,
    out_proj: Linear,
    time_embeddings: SinusoidalPosEmb,
    time_mlp: TimestepEmbedding,
    delta_time_mlp: TimestepEmbedding,
    decoder: MiniCPMModel,
}

impl VoxCPMLocDiT {
    pub fn new(vb: VarBuilder, config: VoxMiniCPM4Config, in_channels: usize) -> Result<Self> {
        let in_proj = linear(in_channels, config.hidden_size, vb.pp("in_proj"))?;
        let cond_proj = linear(in_channels, config.hidden_size, vb.pp("cond_proj"))?;
        let out_proj = linear(config.hidden_size, in_channels, vb.pp("out_proj"))?;
        let time_embeddings = SinusoidalPosEmb::new(config.hidden_size)?;
        let time_mlp = TimestepEmbedding::new(
            vb.pp("time_mlp"),
            config.hidden_size,
            config.hidden_size,
            None,
        )?;
        let delta_time_mlp = TimestepEmbedding::new(
            vb.pp("delta_time_mlp"),
            config.hidden_size,
            config.hidden_size,
            None,
        )?;
        assert_eq!(config.vocab_size, 0, "vocab_size must be 0 for local DiT");
        let decoder = MiniCPMModel::new(vb.pp("decoder"), config.clone())?;
        Ok(Self {
            in_proj,
            cond_proj,
            out_proj,
            time_embeddings,
            time_mlp,
            delta_time_mlp,
            decoder,
        })
    }

    pub fn forward(
        &mut self,
        x: &Tensor,
        mu: &Tensor,
        t: &Tensor,
        cond: &Tensor,
        dt: &Tensor,
    ) -> Result<Tensor> {
        let x = self.in_proj.forward(&x.transpose(1, 2)?.contiguous()?)?;
        let cond = self
            .cond_proj
            .forward(&cond.transpose(1, 2)?.contiguous()?)?;
        let prefix = cond.dim(1)?;
        let t = self.time_embeddings.forward(t, 1000)?.to_dtype(x.dtype())?;
        let t = self.time_mlp.forward(&t)?;
        let dt = self
            .time_embeddings
            .forward(dt, 1000)?
            .to_dtype(x.dtype())?;
        let dt = self.delta_time_mlp.forward(&dt)?;
        let t = t.add(&dt)?;

        let x = Tensor::cat(&[mu.add(&t)?.unsqueeze(1)?, cond, x], 1)?;
        let hidden = self.decoder.forward(&x, 0, false)?;
        let select_len = hidden.dims()[1] - (prefix + 1);
        let hidden = hidden.narrow(1, prefix + 1, select_len)?;
        let hidden = self.out_proj.forward(&hidden)?;
        let hidden = hidden.transpose(1, 2)?.contiguous()?;
        Ok(hidden)
    }
}

pub struct UnifiedCFM {
    in_channels: usize,
    mean_mode: bool,
    estimator: VoxCPMLocDiT,
}

impl UnifiedCFM {
    pub fn new(
        in_channels: usize,
        _cfm_params: CfmConfig,
        estimator: VoxCPMLocDiT,
        mean_mode: bool,
    ) -> Result<Self> {
        Ok(Self {
            in_channels,
            mean_mode,
            estimator,
        })
    }

    pub fn forward(
        &mut self,
        mu: &Tensor,
        n_timesteps: usize,
        patch_size: usize,
        cond: &Tensor,
        temperature: f64,
        cfg_value: f64,
        sway_sampling_coef: f64,
        use_cfg_zero_star: bool,
    ) -> Result<Tensor> {
        let (b, _) = mu.dims2()?;
        let t = patch_size;
        let dtype = mu.dtype();
        let z = Tensor::randn(0.0f32, 1.0, (b, self.in_channels, t), mu.device())?
            .to_dtype(dtype)?
            .affine(temperature, 0.0)?;
        let t_span = linspace(1.0, 0.0, n_timesteps + 1, mu.device())?.to_dtype(dtype)?;
        let t_span = t_span
            .affine(std::f64::consts::PI / 2.0, 0.0)?
            .cos()?
            .affine(1.0, -1.0)?
            .add(&t_span)?
            .affine(sway_sampling_coef, 0.0)?
            .add(&t_span)?;
        let x = self.solve_euler(&z, &t_span, mu, cond, cfg_value, use_cfg_zero_star)?;
        Ok(x)
    }

    pub fn optimized_scale(
        &self,
        positive_flat: &Tensor,
        negative_flat: &Tensor,
    ) -> Result<Tensor> {
        let dot_product = positive_flat.mul(negative_flat)?.sum_keepdim(1)?;
        let squared_norm = negative_flat.powf(2.0)?.sum_keepdim(1)?.affine(1.0, 1e-8)?;
        let st_star = dot_product.div(&squared_norm)?;
        Ok(st_star)
    }

    pub fn solve_euler(
        &mut self,
        x: &Tensor,
        t_span: &Tensor,
        mu: &Tensor,
        cond: &Tensor,
        cfg_value: f64,
        use_cfg_zero_star: bool,
    ) -> Result<Tensor> {
        let mut t = t_span.i(0)?;
        let mut dt = t.sub(&t_span.i(1)?)?;
        let mut sol = Vec::new();
        let t_span_len = t_span.dim(0)?;
        let zero_init_steps = max(1, (t_span_len as f32 * 0.04) as usize);
        let mut dphi_dt;
        let mut x = x.to_owned();
        for step in 1..t_span_len {
            if use_cfg_zero_star && step <= zero_init_steps {
                // dphi_dt = Tensor::zeros(1, t_span.dtype(), t_span.device())?;
                dphi_dt = x.zeros_like()?;
            } else {
                let b = x.dim(0)?;
                // let x_in = Tensor::zeros((2*b, self.in_channels, x.dim(2)?), x.dtype(), x.device())?;
                let x_in = Tensor::cat(&[x.clone(), x.clone()], 0)?;
                let mu_in = Tensor::zeros((b, mu.dim(1)?), x.dtype(), x.device())?;
                let mu_in = Tensor::cat(&[mu.clone(), mu_in], 0)?;
                let t_in = t.broadcast_as(2 * b)?;
                let dt_in = if self.mean_mode {
                    dt.broadcast_as(2 * b)?
                } else {
                    Tensor::zeros(2 * b, x.dtype(), x.device())?
                };
                let cond_in = Tensor::cat(&[cond, cond], 0)?;
                dphi_dt = self
                    .estimator
                    .forward(&x_in, &mu_in, &t_in, &cond_in, &dt_in)?;
                let split = dphi_dt.split(&[b, b], 0)?;
                dphi_dt = split[0].to_owned();
                let cfg_dphi_dt = split[1].to_owned();
                let mut st_star = Tensor::ones(1, x.dtype(), x.device())?;
                if use_cfg_zero_star {
                    let positive_flat = dphi_dt.reshape((b, ()))?;
                    let negative_flat = cfg_dphi_dt.reshape((b, ()))?;
                    st_star = self.optimized_scale(&positive_flat, &negative_flat)?;
                    let mut vec_shape = vec![b];
                    let vec_shape1 = vec![1; dphi_dt.rank() - 1];
                    vec_shape.extend_from_slice(&vec_shape1);
                    st_star = st_star.reshape(vec_shape)?;
                }
                let cfg = cfg_dphi_dt.broadcast_mul(&st_star)?;
                dphi_dt = cfg.add(&dphi_dt.sub(&cfg)?.affine(cfg_value, 0.0)?)?;
                // step步的预测噪声
            }
            x = x.broadcast_sub(&dphi_dt.broadcast_mul(&dt)?)?; // 逐步去噪
            t = t.sub(&dt)?;
            sol.push(x.to_owned());
            if step < t_span_len - 1 {
                dt = t.sub(&t_span.i(step + 1)?)?;
            }
        }
        Ok(sol.into_iter().last().unwrap())
    }
}

pub struct VoxCPMLocEnc {
    special_token: Tensor,
    in_proj: Linear,
    encoder: MiniCPMModel,
    hidden_size: usize,
}

impl VoxCPMLocEnc {
    pub fn new(vb: VarBuilder, config: VoxMiniCPM4Config, input_dim: usize) -> Result<Self> {
        let special_token = vb.get((1, 1, 1, config.hidden_size), "special_token")?;
        let in_proj = linear(input_dim, config.hidden_size, vb.pp("in_proj"))?;
        assert_eq!(
            config.vocab_size, 0,
            "vocab_size must be 0 for local encoder"
        );
        let hidden_size = config.hidden_size;
        let encoder = MiniCPMModel::new(vb.pp("encoder"), config)?;
        Ok(Self {
            special_token,
            in_proj,
            encoder,
            hidden_size,
        })
    }

    pub fn forward(&mut self, x: &Tensor) -> Result<Tensor> {
        let (b, t, _, _) = x.dims4()?;
        let x = self.in_proj.forward(x)?;
        let special_tokens = self.special_token.expand((b, t, 1, self.hidden_size))?;
        let x = Tensor::cat(&[&special_tokens, &x], 2)?;
        let (b, t, p, c) = x.dims4()?;
        let x = x.reshape((b * t, p, c))?;
        let outputs = self.encoder.forward(&x, 0, false)?;
        let cls_output = outputs.i((.., 0, ..))?;
        let cls_output = cls_output.reshape((b, t, c))?;
        Ok(cls_output)
    }
}

pub struct VoxCPMModel {
    config: VoxCPMConfig,
    patch_size: usize,
    audio_start_token: usize,
    chunk_size: usize,
    sample_rate: usize,
    tokenizer: SingleChineseTokenizer,
    audio_vae: AudioVAE,
    base_lm: MiniCPMModel,
    residual_lm: MiniCPMModel,
    feat_encoder: VoxCPMLocEnc,
    feat_decoder: UnifiedCFM,
    fsq_layer: ScalarQuantizationLayer,
    enc_to_lm_proj: Linear,
    lm_to_dit_proj: Linear,
    res_to_dit_proj: Linear,
    stop_proj: Linear,
    stop_head: Linear,
    device: Device,
    dtype: DType,
}

impl VoxCPMModel {
    pub fn new(
        vb: VarBuilder,
        config: VoxCPMConfig,
        tokenizer: SingleChineseTokenizer,
        audio_vae: AudioVAE,
    ) -> Result<Self> {
        let base_lm = MiniCPMModel::new(vb.pp("base_lm"), config.lm_config.clone())?;
        let audio_start_token = 101usize;
        let mut residual_lm_config = config.lm_config.clone();
        residual_lm_config.num_hidden_layers = config.residual_lm_num_layers;
        residual_lm_config.vocab_size = 0;
        let residual_lm = MiniCPMModel::new(vb.pp("residual_lm"), residual_lm_config)?;
        let mut encoder_config = config.lm_config.clone();
        encoder_config.hidden_size = config.encoder_config.hidden_dim;
        encoder_config.intermediate_size = config.encoder_config.ffn_dim;
        encoder_config.num_attention_heads = config.encoder_config.num_heads;
        encoder_config.num_hidden_layers = config.encoder_config.num_layers;
        encoder_config.vocab_size = 0;
        let feat_encoder =
            VoxCPMLocEnc::new(vb.pp("feat_encoder"), encoder_config, config.feat_dim)?;

        let mut decoder_config = config.lm_config.clone();
        decoder_config.hidden_size = config.dit_config.hidden_dim;
        decoder_config.intermediate_size = config.dit_config.ffn_dim;
        decoder_config.num_attention_heads = config.dit_config.num_heads;
        decoder_config.num_hidden_layers = config.dit_config.num_layers;
        decoder_config.vocab_size = 0;
        let estimator = VoxCPMLocDiT::new(
            vb.pp("feat_decoder.estimator"),
            decoder_config,
            config.feat_dim,
        )?;
        let feat_decoder = UnifiedCFM::new(
            config.feat_dim,
            config.dit_config.cfm_config.clone(),
            estimator,
            false,
        )?;
        let fsq_layer = ScalarQuantizationLayer::new(
            vb.pp("fsq_layer"),
            config.lm_config.hidden_size,
            config.lm_config.hidden_size,
            config.scalar_quantization_latent_dim,
            config.scalar_quantization_scale,
        )?;
        let enc_to_lm_proj = linear(
            config.encoder_config.hidden_dim,
            config.lm_config.hidden_size,
            vb.pp("enc_to_lm_proj"),
        )?;
        let lm_to_dit_proj = linear(
            config.lm_config.hidden_size,
            config.dit_config.hidden_dim,
            vb.pp("lm_to_dit_proj"),
        )?;
        let res_to_dit_proj = linear(
            config.lm_config.hidden_size,
            config.dit_config.hidden_dim,
            vb.pp("res_to_dit_proj"),
        )?;

        let stop_proj = linear(
            config.lm_config.hidden_size,
            config.lm_config.hidden_size,
            vb.pp("stop_proj"),
        )?;
        let stop_head = linear_no_bias(config.lm_config.hidden_size, 2, vb.pp("stop_head"))?;

        let patch_size = config.patch_size;
        Ok(Self {
            config,
            patch_size,
            audio_start_token,
            // audio_end_token,
            chunk_size: audio_vae.chunk_size,
            sample_rate: audio_vae.sample_rate,
            tokenizer,
            audio_vae,
            base_lm,
            residual_lm,
            feat_encoder,
            feat_decoder,
            fsq_layer,
            enc_to_lm_proj,
            lm_to_dit_proj,
            res_to_dit_proj,
            stop_proj,
            stop_head,
            device: vb.device().clone(),
            dtype: vb.dtype(),
        })
    }

    pub fn generate(
        &mut self,
        target_text: String,
        prompt_text: Option<String>,
        prompt_wav_path: Option<String>,
        min_len: usize,
        max_len: usize,
        inference_timesteps: usize,
        cfg_value: f64,
        // retry_badcase: bool,
        retry_badcase_ratio_threshold: f64,
    ) -> Result<Tensor> {
        let (text_token, text_mask, audio_feat, audio_mask) = match prompt_wav_path {
            None => {
                let text_token = self.tokenizer.encode(target_text.clone())?;
                let text_token = Tensor::from_slice(&text_token, text_token.len(), &self.device)?;
                let audio_start = Tensor::new(vec![self.audio_start_token as u32], &self.device)?;
                let text_token = Tensor::cat(&[text_token, audio_start], D::Minus1)?;
                let text_length = text_token.dim(0)?;
                let audio_feat = Tensor::zeros(
                    (text_length, self.patch_size, self.audio_vae.latent_dim),
                    DType::F32,
                    &self.device,
                )?;
                let text_mask = Tensor::ones(text_length, self.dtype, &self.device)?;
                let audio_mask = Tensor::zeros(text_length, self.dtype, &self.device)?;
                (text_token, text_mask, audio_feat, audio_mask)
            }
            Some(path) => {
                let text = prompt_text.unwrap_or("".to_string()) + &target_text;
                let text_token = self.tokenizer.encode(text)?;
                let text_token = Tensor::from_slice(&text_token, text_token.len(), &self.device)?;
                let audio_start = Tensor::new(vec![self.audio_start_token as u32], &self.device)?;
                let text_token = Tensor::cat(&[text_token, audio_start], D::Minus1)?;
                let text_length = text_token.dim(0)?;
                let mut audio =
                    load_audio_with_resample(&path, &self.device.clone(), Some(self.sample_rate))?;
                let patch_len = self.patch_size * self.chunk_size;
                if audio.dim(1)? % patch_len != 0 {
                    audio = audio.pad_with_zeros(
                        D::Minus1,
                        // 0,
                        // patch_len - audio.dim(1)? % patch_len,
                        patch_len - audio.dim(1)? % patch_len,
                        0,
                    )?;
                }
                let audio_feat = self.audio_vae.encode(&audio, Some(self.sample_rate))?;
                let audio_feat = audio_feat
                    .reshape((self.audio_vae.latent_dim, (), self.patch_size))?
                    .permute((1, 2, 0))?;
                // let dim0 = audio_feat.dim(0)? - 1;
                // let audio_feat = audio_feat.i(..dim0)?;
                let audio_length = audio_feat.dim(0)?;
                let text_pad_token = Tensor::zeros(audio_length, DType::U32, &self.device)?;
                let text_token = Tensor::cat(&[text_token, text_pad_token], D::Minus1)?;
                let audio_pad_feat = Tensor::zeros(
                    (text_length, self.patch_size, self.audio_vae.latent_dim),
                    audio_feat.dtype(),
                    &self.device,
                )?;
                let audio_feat = Tensor::cat(&[audio_pad_feat, audio_feat], 0)?;
                let text_mask = Tensor::cat(
                    &[
                        Tensor::ones(text_length, self.dtype, &self.device)?,
                        Tensor::zeros(audio_length, self.dtype, &self.device)?,
                    ],
                    D::Minus1,
                )?;
                let audio_mask = Tensor::cat(
                    &[
                        Tensor::zeros(text_length, self.dtype, &self.device)?,
                        Tensor::ones(audio_length, self.dtype, &self.device)?,
                    ],
                    D::Minus1,
                )?;
                (text_token, text_mask, audio_feat, audio_mask)
            }
        };
        let target_text_length = self.tokenizer.encode(target_text)?.len();
        let max_len = max_len
            .min((target_text_length as f64 * retry_badcase_ratio_threshold + 10.0) as usize);
        let decode_audio = self._generate(
            &text_token,
            &text_mask,
            &audio_feat,
            &audio_mask,
            min_len,
            max_len,
            inference_timesteps,
            cfg_value,
        )?;
        Ok(decode_audio)
    }

    fn _generate(
        &mut self,
        text_token: &Tensor,
        text_mask: &Tensor,
        audio_feat: &Tensor,
        audio_mask: &Tensor,
        min_len: usize,
        max_len: usize,
        inference_timesteps: usize,
        cfg_value: f64,
    ) -> Result<Tensor> {
        let text_token = text_token.unsqueeze(0)?;
        let text_mask = text_mask.unsqueeze(0)?;
        let audio_feat = audio_feat.unsqueeze(0)?.to_dtype(self.dtype)?;
        let audio_mask = audio_mask.unsqueeze(0)?;

        let latent_pred = self.inference(
            &text_token,
            &text_mask,
            &audio_feat,
            &audio_mask,
            min_len,
            max_len,
            inference_timesteps,
            cfg_value,
        )?;
        let decode_audio = self
            .audio_vae
            .decode(&latent_pred.to_dtype(DType::F32)?)?
            .squeeze(1)?;
        let audio_len = decode_audio.dim(D::Minus1)?;
        let decode_audio = decode_audio.narrow(D::Minus1, 640, audio_len - 1280)?;
        Ok(decode_audio)
    }

    fn _generate_stream(
        &mut self,
        text_token: &Tensor,
        text_mask: &Tensor,
        audio_feat: &Tensor,
        audio_mask: &Tensor,
        min_len: usize,
        max_len: usize,
        inference_timesteps: usize,
        cfg_value: f64,
    ) -> Result<VoxCPMGenerateStream<'_>> {
        let text_token = text_token.unsqueeze(0)?;
        let text_mask = text_mask.unsqueeze(0)?;
        let audio_feat = audio_feat.unsqueeze(0)?.to_dtype(self.dtype)?;
        let audio_mask = audio_mask.unsqueeze(0)?;
        let inf_stream = self.inference_stream(
            &text_token,
            &text_mask,
            &audio_feat,
            &audio_mask,
            min_len,
            max_len,
            inference_timesteps,
            cfg_value,
        )?;

        Ok(VoxCPMGenerateStream {
            inf_stream,
            vae_state: None,
            all_latents: Vec::new(),
            pending_chunk: None,
            finished: false,
        })
    }

    fn inference(
        &mut self,
        text: &Tensor,
        text_mask: &Tensor,
        feat: &Tensor,
        feat_mask: &Tensor,
        min_len: usize,
        max_len: usize,
        inference_timesteps: usize,
        cfg_value: f64,
    ) -> Result<Tensor> {
        let mut pred_feat_seq = Vec::new();
        self.inference_stream(
            text,
            text_mask,
            feat,
            feat_mask,
            min_len,
            max_len,
            inference_timesteps,
            cfg_value,
        )?
        .for_each(|chunk| {
            if let Ok(feat) = chunk {
                pred_feat_seq.push(feat);
            }
        });

        let pred_seq = Tensor::cat(&pred_feat_seq, 1)?; // (b, t, p, d)
        let (b, _, _, d) = pred_seq.dims4()?;
        let feat_pred = pred_seq
            .permute((0, 3, 1, 2))?
            .reshape((b, d, ()))?
            .contiguous()?;
        self.base_lm.clear_kv_cache();
        self.residual_lm.clear_kv_cache();
        Ok(feat_pred)
    }

    pub fn inference_stream(
        &mut self,
        text: &Tensor,
        text_mask: &Tensor,
        feat: &Tensor,
        feat_mask: &Tensor,
        min_len: usize,
        max_len: usize,
        inference_timesteps: usize,
        cfg_value: f64,
    ) -> Result<VoxCPMInferenceStream<'_>> {
        let (_, t, _, _) = feat.dims4()?;
        let feat_embed = self.feat_encoder.forward(feat)?; // [b, t, h_feat]
        let feat_embed = self.enc_to_lm_proj.forward(&feat_embed)?;
        let scale_emb = if self.config.lm_config.use_mup {
            self.config.lm_config.scale_emb
        } else {
            1.0
        };

        let text_embed = self
            .base_lm
            .embed_tokens
            .as_ref()
            .unwrap()
            .forward(text)?
            .affine(scale_emb as f64, 0.0)?;
        let combined_embed = text_mask
            .unsqueeze(D::Minus1)?
            .broadcast_mul(&text_embed)?
            .add(&feat_mask.unsqueeze(D::Minus1)?.broadcast_mul(&feat_embed)?)?;
        let prefix_feat_cond = feat.i((.., t - 1, ..))?;

        let position_id = 0;
        let seq_len = t;
        let enc_outputs = self
            .base_lm
            .forward_with_cache(&combined_embed, position_id)?;
        let enc_outputs = self
            .fsq_layer
            .forward(&enc_outputs)?
            .broadcast_mul(&feat_mask.unsqueeze(D::Minus1)?)?
            .add(&enc_outputs.broadcast_mul(&text_mask.unsqueeze(D::Minus1)?)?)?;

        let lm_hidden = enc_outputs.i((.., t - 1, ..))?;

        let input_embeds =
            enc_outputs.add(&feat_mask.unsqueeze(D::Minus1)?.broadcast_mul(&feat_embed)?)?;
        let residual_enc_outputs = self
            .residual_lm
            .forward_with_cache(&input_embeds, position_id)?;
        let residual_hidden = residual_enc_outputs.i((.., t - 1, ..))?;

        Ok(VoxCPMInferenceStream {
            model: self,
            lm_hidden,
            residual_hidden,
            prefix_feat_cond,
            position_id,
            seq_len,
            i: 0,
            max_len,
            min_len,
            inference_timesteps,
            cfg_value,
            finished: false,
        })
    }

    pub fn generate_stream(
        &mut self,
        target_text: String,
        prompt_text: Option<String>,
        prompt_wav_path: Option<String>,
        min_len: usize,
        max_len: usize,
        inference_timesteps: usize,
        cfg_value: f64,
        retry_badcase_ratio_threshold: f64,
    ) -> Result<VoxCPMGenerateStream<'_>> {
        let text_token = self.tokenizer.encode(target_text.clone())?;
        let text_token = Tensor::from_slice(&text_token, text_token.len(), &self.device)?;
        let audio_start = Tensor::new(vec![self.audio_start_token as u32], &self.device)?;
        let text_token = Tensor::cat(&[text_token, audio_start], D::Minus1)?;
        let text_length = text_token.dim(0)?;

        let (text_token, text_mask, audio_feat, audio_mask) = match (prompt_text, prompt_wav_path) {
            (None, _) | (_, None) => {
                let audio_feat = Tensor::zeros(
                    (text_length, self.patch_size, self.audio_vae.latent_dim),
                    self.dtype,
                    &self.device,
                )?;
                let text_mask = Tensor::ones(text_length, self.dtype, &self.device)?;
                let audio_mask = Tensor::zeros(text_length, self.dtype, &self.device)?;
                (text_token, text_mask, audio_feat, audio_mask)
            }
            (Some(prompt_text), Some(prompt_wav_path)) => {
                let prompt_token = self.tokenizer.encode(prompt_text)?;
                let prompt_token =
                    Tensor::from_slice(&prompt_token, prompt_token.len(), &self.device)?;
                let text_token = Tensor::cat(&[prompt_token, text_token], 0)?;
                let text_length = text_token.dim(0)?;

                let mut audio = crate::utils::audio::load_audio_with_resample(
                    &prompt_wav_path,
                    &self.device.clone(),
                    Some(self.sample_rate),
                )?;
                let patch_len = self.patch_size * self.chunk_size;
                if audio.dim(1)? % patch_len != 0 {
                    audio = audio.pad_with_zeros(
                        D::Minus1,
                        patch_len - audio.dim(1)? % patch_len,
                        0,
                    )?;
                }
                let audio_feat = self.audio_vae.encode(&audio, Some(self.sample_rate))?;
                let audio_feat = audio_feat
                    .reshape((self.audio_vae.latent_dim, (), self.patch_size))?
                    .permute((1, 2, 0))?;
                let audio_length = audio_feat.dim(0)?;
                let text_pad_token = Tensor::zeros(audio_length, DType::U32, &self.device)?;
                let text_token = Tensor::cat(&[text_token, text_pad_token], D::Minus1)?;
                let audio_pad_feat = Tensor::zeros(
                    (text_length, self.patch_size, self.audio_vae.latent_dim),
                    audio_feat.dtype(),
                    &self.device,
                )?;
                let audio_feat = Tensor::cat(&[audio_pad_feat, audio_feat], 0)?;
                let text_mask = Tensor::cat(
                    &[
                        Tensor::ones(text_length, self.dtype, &self.device)?,
                        Tensor::zeros(audio_length, self.dtype, &self.device)?,
                    ],
                    D::Minus1,
                )?;
                let audio_mask = Tensor::cat(
                    &[
                        Tensor::zeros(text_length, self.dtype, &self.device)?,
                        Tensor::ones(audio_length, self.dtype, &self.device)?,
                    ],
                    D::Minus1,
                )?;
                (text_token, text_mask, audio_feat, audio_mask)
            }
        };
        let target_text_length = self.tokenizer.encode(target_text)?.len();
        let max_len = max_len
            .min((target_text_length as f64 * retry_badcase_ratio_threshold + 10.0) as usize);

        self._generate_stream(
            &text_token,
            &text_mask,
            &audio_feat,
            &audio_mask,
            min_len,
            max_len,
            inference_timesteps,
            cfg_value,
        )
    }
}

pub struct VoxCPMInferenceStream<'a> {
    model: &'a mut VoxCPMModel,
    lm_hidden: Tensor,
    residual_hidden: Tensor,
    prefix_feat_cond: Tensor,
    position_id: usize,
    seq_len: usize,
    i: usize,
    max_len: usize,
    min_len: usize,
    inference_timesteps: usize,
    cfg_value: f64,
    finished: bool,
}

impl<'a> Iterator for VoxCPMInferenceStream<'a> {
    type Item = Result<Tensor>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.finished || self.i >= self.max_len {
            return None;
        }

        let dit_hidden_1 = match self.model.lm_to_dit_proj.forward(&self.lm_hidden) {
            Ok(t) => t,
            Err(e) => return Some(Err(e.into())),
        };
        let dit_hidden_2 = match self.model.res_to_dit_proj.forward(&self.residual_hidden) {
            Ok(t) => t,
            Err(e) => return Some(Err(e.into())),
        };
        let dit_hidden = match dit_hidden_1.add(&dit_hidden_2) {
            Ok(t) => t,
            Err(e) => return Some(Err(e.into())),
        };
        let cond = match self.prefix_feat_cond.transpose(1, 2) {
            Ok(t) => match t.contiguous() {
                Ok(t) => t,
                Err(e) => return Some(Err(e.into())),
            },
            Err(e) => return Some(Err(e.into())),
        };

        let pred_feat = match self.model.feat_decoder.forward(
            &dit_hidden,
            self.inference_timesteps,
            self.model.patch_size,
            &cond,
            1.0,
            self.cfg_value,
            1.0,
            true,
        ) {
            Ok(t) => match t.transpose(1, 2) {
                Ok(t) => t,
                Err(e) => return Some(Err(e.into())),
            },
            Err(e) => return Some(Err(e.into())),
        };

        let pred_feat_unsqueezed = match pred_feat.unsqueeze(1) {
            Ok(t) => t,
            Err(e) => return Some(Err(e.into())),
        };

        let curr_embed = match self.model.feat_encoder.forward(&pred_feat_unsqueezed) {
            Ok(t) => match self.model.enc_to_lm_proj.forward(&t) {
                Ok(t) => t,
                Err(e) => return Some(Err(e.into())),
            },
            Err(e) => return Some(Err(e.into())),
        };

        self.prefix_feat_cond = pred_feat;
        let stop_flag = match self.model.stop_proj.forward(&self.lm_hidden) {
            Ok(t) => match t.silu() {
                Ok(t) => match self.model.stop_head.forward(&t) {
                    Ok(t) => match t.argmax(D::Minus1) {
                        Ok(t) => match t.i(0) {
                            Ok(t) => match t.to_scalar::<u32>() {
                                Ok(v) => v,
                                Err(e) => return Some(Err(e.into())),
                            },
                            Err(e) => return Some(Err(e.into())),
                        },
                        Err(e) => return Some(Err(e.into())),
                    },
                    Err(e) => return Some(Err(e.into())),
                },
                Err(e) => return Some(Err(e.into())),
            },
            Err(e) => return Some(Err(e.into())),
        };

        if self.i > self.min_len && stop_flag == 1 {
            self.finished = true;
        }

        self.position_id += self.seq_len;
        self.seq_len = 1;

        let curr_embed_val = match curr_embed.i((.., 0, ..)) {
            Ok(t) => t,
            Err(e) => return Some(Err(e.into())),
        };

        self.lm_hidden = match self
            .model
            .base_lm
            .forward_with_cache(&curr_embed_val, self.position_id)
        {
            Ok(t) => match t.squeeze(1) {
                Ok(t) => match self.model.fsq_layer.forward(&t) {
                    Ok(t) => t,
                    Err(e) => return Some(Err(e.into())),
                },
                Err(e) => return Some(Err(e.into())),
            },
            Err(e) => return Some(Err(e.into())),
        };

        self.residual_hidden = match self.lm_hidden.add(&curr_embed_val) {
            Ok(t) => match self
                .model
                .residual_lm
                .forward_with_cache(&t, self.position_id)
            {
                Ok(t) => match t.squeeze(1) {
                    Ok(t) => t,
                    Err(e) => return Some(Err(e.into())),
                },
                Err(e) => return Some(Err(e.into())),
            },
            Err(e) => return Some(Err(e.into())),
        };

        self.i += 1;
        Some(Ok(pred_feat_unsqueezed))
    }
}

impl<'a> Drop for VoxCPMInferenceStream<'a> {
    fn drop(&mut self) {
        self.model.base_lm.clear_kv_cache();
        self.model.residual_lm.clear_kv_cache();
    }
}

pub struct VoxCPMGenerateStream<'a> {
    inf_stream: VoxCPMInferenceStream<'a>,
    vae_state: Option<crate::audio_vae::DecoderState>,
    all_latents: Vec<Tensor>,
    pending_chunk: Option<Tensor>,
    finished: bool,
}

impl<'a> Iterator for VoxCPMGenerateStream<'a> {
    type Item = Result<Tensor>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.finished {
            return None;
        }

        while let Some(next_latent) = self.inf_stream.next() {
            match next_latent {
                Ok(latent) => {
                    // latent: [B, 1, P, D]
                    let (b, _, p, d) = match latent.dims4() {
                        Ok(dims) => dims,
                        Err(e) => return Some(Err(e.into())),
                    };

                    // Reshape to [B, D, P] for VAE
                    let latent_vae = match latent
                        .permute((0, 3, 1, 2))
                        .and_then(|t| t.reshape((b, d, p)))
                    {
                        Ok(t) => t,
                        Err(e) => return Some(Err(e.into())),
                    };

                    if self.vae_state.is_none() {
                        let state = match self.inf_stream.model.audio_vae.init_decoder_state(
                            b,
                            &latent.device(),
                            DType::F32,
                        ) {
                            Ok(s) => s,
                            Err(e) => return Some(Err(e.into())),
                        };
                        self.vae_state = Some(state);
                    }

                    self.all_latents.push(latent);

                    if self.all_latents.len() == 1 {
                        // Latent 0 skipped
                    } else {
                        let audio_chunk = match self.inf_stream.model.audio_vae.decode_stream(
                            &latent_vae.to_dtype(DType::F32).unwrap(),
                            self.vae_state.as_mut().unwrap(),
                        ) {
                            Ok(t) => match t.squeeze(1) {
                                Ok(t) => t,
                                Err(e) => return Some(Err(e.into())),
                            },
                            Err(e) => return Some(Err(e.into())),
                        };

                        if let Some(to_yield) = self.pending_chunk.replace(audio_chunk) {
                            return Some(Ok(to_yield));
                        }
                    }
                }
                Err(e) => return Some(Err(e)),
            }
        }

        self.finished = true;
        None
    }
}

impl VoxCPMModel {
    pub fn build_prompt_cache(
        &mut self,
        prompt_text: String,
        prompt_wav_path: String,
    ) -> Result<HashMap<String, Tensor>> {
        let text_token = self.tokenizer.encode(prompt_text)?;
        let text_token = Tensor::from_slice(&text_token, text_token.len(), &self.device)?;
        let mut audio = load_audio_with_resample(
            &prompt_wav_path,
            &self.device.clone(),
            Some(self.sample_rate),
        )?;
        let patch_len = self.patch_size * self.chunk_size;
        if audio.dim(1)? % patch_len != 0 {
            audio = audio.pad_with_zeros(D::Minus1, 0, patch_len - audio.dim(1)? % patch_len)?;
        }
        let audio_feat = self.audio_vae.encode(&audio, Some(self.sample_rate))?;
        let audio_feat = audio_feat
            .reshape((self.audio_vae.latent_dim, (), self.patch_size))?
            .permute((1, 2, 0))?;
        let dim0 = audio_feat.dim(0)? - 1;
        let audio_feat = audio_feat.i(..dim0)?;
        let mut hashmap = HashMap::new();
        hashmap.insert("text_token".to_string(), text_token);
        hashmap.insert("audio_feat".to_string(), audio_feat);
        Ok(hashmap)
    }

    pub fn generate_with_prompt_cache(
        &mut self,
        target_text: String,
        prompt_cache: HashMap<String, Tensor>,
        min_len: usize,
        max_len: usize,
        inference_timesteps: usize,
        cfg_value: f64,
        retry_badcase: bool,
        retry_badcase_ratio_threshold: f64,
    ) -> Result<Tensor> {
        let target_text_token = self.tokenizer.encode(target_text.clone())?;
        let target_text_token =
            Tensor::from_slice(&target_text_token, target_text_token.len(), &self.device)?;
        let text_token = match prompt_cache.get("text_token") {
            Some(token) => Tensor::cat(&[token, &target_text_token], 0)?,
            None => target_text_token,
        };
        let audio_start = Tensor::new(vec![self.audio_start_token as u32], &self.device)?;
        let text_token = Tensor::cat(&[text_token, audio_start], D::Minus1)?;
        let text_length = text_token.dim(0)?;
        let (audio_length, audio_feat) = match prompt_cache.get("audio_feat") {
            Some(feat) => (feat.dim(0)?, Some(feat.clone())),
            None => (0, None),
        };
        let (text_token, text_mask, audio_feat, audio_mask) = if audio_length > 0 {
            let audio_feat = audio_feat.unwrap();
            let audio_length = audio_feat.dim(0)?;
            let text_pad_token = Tensor::zeros(audio_length, DType::U32, &self.device)?;
            let text_token = Tensor::cat(&[text_token, text_pad_token], D::Minus1)?;
            let audio_pad_feat = Tensor::zeros(
                (text_length, self.patch_size, self.audio_vae.latent_dim),
                audio_feat.dtype(),
                &self.device,
            )?;
            let audio_feat = Tensor::cat(&[audio_pad_feat, audio_feat], 0)?;
            let text_mask = Tensor::cat(
                &[
                    Tensor::ones(text_length, self.dtype, &self.device)?,
                    Tensor::zeros(audio_length, self.dtype, &self.device)?,
                ],
                D::Minus1,
            )?;
            let audio_mask = Tensor::cat(
                &[
                    Tensor::zeros(text_length, self.dtype, &self.device)?,
                    Tensor::ones(audio_length, self.dtype, &self.device)?,
                ],
                D::Minus1,
            )?;
            (text_token, text_mask, audio_feat, audio_mask)
        } else {
            let audio_feat = Tensor::zeros(
                (text_length, self.patch_size, self.audio_vae.latent_dim),
                DType::F32,
                &self.device,
            )?;
            let text_mask = Tensor::ones(text_length, self.dtype, &self.device)?;
            let audio_mask = Tensor::zeros(text_length, self.dtype, &self.device)?;
            (text_token, text_mask, audio_feat, audio_mask)
        };
        let target_text_length = self.tokenizer.encode(target_text)?.len();
        let max_len = if retry_badcase {
            (target_text_length as f64 * retry_badcase_ratio_threshold + 10.0) as usize
        } else {
            max_len
        };
        let decode_audio = self._generate(
            &text_token,
            &text_mask,
            &audio_feat,
            &audio_mask,
            min_len,
            max_len,
            inference_timesteps,
            cfg_value,
        )?;
        Ok(decode_audio)
    }
}
