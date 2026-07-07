use anyhow::Result;
use candle_core::{DType, Device, IndexOp, Tensor, D};
use candle_nn::{Module, VarBuilder};
use std::{
    cmp::max,
    collections::HashMap,
    time::{Duration, Instant},
};

use crate::{
    audio_vae::AudioVAE,
    config::{CfmConfig, VoxCPMConfig, VoxMiniCPM4Config},
    generate::VoxCPMGenerationConfig,
    linear::{linear_x, LinearX},
    minicpm4::MiniCPMModel,
    profile::{profile_stream_enabled, stage_capture_enabled, InferenceStepProfile, StageProfile},
    quant::{QuantBuildCtx, QuantStats, VoxCPMQuantConfig},
    tokenizer::SingleChineseTokenizer,
    utils::audio::load_audio_with_resample,
    utils::tensor::{linspace, scatter_ranges_dim0},
};

/// VoxCPM2 audio embedding placement derived during input preparation (no GPU mask readback).
#[derive(Debug, Clone)]
struct VoxCPM2InputLayout {
    has_audio: bool,
    audio_ranges: Vec<(usize, usize)>,
}

#[derive(Debug, Clone, Copy)]
struct GenerationParams {
    min_len: usize,
    max_len: usize,
    inference_timesteps: usize,
    cfg_value: f64,
    stream_decode_latent_batch: usize,
    stream_decode_initial_latent_batch: usize,
    stop_check_interval: usize,
}

impl GenerationParams {
    fn resolve(config: VoxCPMGenerationConfig, target_text_len: usize) -> Self {
        Self {
            min_len: config.min_len,
            max_len: config.effective_max_len(target_text_len),
            inference_timesteps: config.inference_timesteps.max(1),
            cfg_value: config.cfg_value,
            stream_decode_latent_batch: config.stream_decode_latent_batch(),
            stream_decode_initial_latent_batch: config.stream_decode_initial_latent_batch(),
            stop_check_interval: config.stop_check_interval.max(1),
        }
    }
}

pub struct ScalarQuantizationLayer {
    scale: usize,
    in_proj: LinearX,
    out_proj: LinearX,
}

impl ScalarQuantizationLayer {
    pub fn new(
        vb: VarBuilder,
        in_dim: usize,
        out_dim: usize,
        laten_dim: usize,
        scale: usize,
        qctx: &QuantBuildCtx,
    ) -> Result<Self> {
        let in_proj = linear_x(
            in_dim,
            laten_dim,
            vb.pp("in_proj"),
            &qctx.pp("in_proj"),
            true,
        )?;
        let out_proj = linear_x(
            laten_dim,
            out_dim,
            vb.pp("out_proj"),
            &qctx.pp("out_proj"),
            true,
        )?;
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
    half_dim: usize,
    /// `10000^(-2i/d)` pre-log exponent spacing: `-ln(10000) * i / (half_dim-1)` via arange + affine + exp.
    inv_freq_log_step: f64,
    cached_inv_freq: Option<Tensor>,
}

impl SinusoidalPosEmb {
    pub fn new(dim: usize) -> Result<Self> {
        assert_eq!(dim % 2, 0, "SinusoidalPosEmb requires dim to be even");
        let half_dim = dim / 2;
        let inv_freq_log_step = 10000.0_f64.ln() / (half_dim - 1) as f64;
        Ok(Self {
            half_dim,
            inv_freq_log_step,
            cached_inv_freq: None,
        })
    }
    pub fn forward(&mut self, x: &Tensor, scale: usize) -> Result<Tensor> {
        let x = if x.rank() < 1 {
            x.unsqueeze(0)?
        } else {
            x.clone()
        };

        if self.cached_inv_freq.is_none()
            || self.cached_inv_freq.as_ref().unwrap().device().location() != x.device().location()
            || self.cached_inv_freq.as_ref().unwrap().dtype() != x.dtype()
        {
            // Build `inv_freq` directly on the inference device (avoids a static CPU tensor + H2D on first use).
            let inv_freq = Tensor::arange(0.0, self.half_dim as f32, x.device())?
                .affine(-self.inv_freq_log_step, 0.0)?
                .exp()?
                .to_dtype(x.dtype())?;
            self.cached_inv_freq = Some(inv_freq);
        }
        let inv_freq = self.cached_inv_freq.as_ref().unwrap();

        let emb = x
            .unsqueeze(1)?
            .affine(scale as f64, 0.0)?
            .matmul(&inv_freq.unsqueeze(0)?)?;
        let emb = Tensor::cat(&[emb.sin()?, emb.cos()?], D::Minus1)?;
        Ok(emb)
    }
}

pub struct TimestepEmbedding {
    linear_1: LinearX,
    linear_2: LinearX,
}

impl TimestepEmbedding {
    pub fn new(
        vb: VarBuilder,
        in_channels: usize,
        time_embed_dim: usize,
        out_dim: Option<usize>,
        qctx: &QuantBuildCtx,
    ) -> Result<Self> {
        let linear_1 = linear_x(
            in_channels,
            time_embed_dim,
            vb.pp("linear_1"),
            &qctx.pp("linear_1"),
            true,
        )?;
        let time_embed_dim_out = out_dim.unwrap_or(time_embed_dim);
        let linear_2 = linear_x(
            time_embed_dim,
            time_embed_dim_out,
            vb.pp("linear_2"),
            &qctx.pp("linear_2"),
            true,
        )?;
        Ok(Self { linear_1, linear_2 })
    }

    pub fn forward(&self, sample: &Tensor) -> Result<Tensor> {
        let sample = self.linear_1.forward(sample)?.silu()?;
        let sample = self.linear_2.forward(&sample)?;
        Ok(sample)
    }
}

pub struct VoxCPMLocDiT {
    in_proj: LinearX,
    cond_proj: LinearX,
    out_proj: LinearX,
    time_embeddings: SinusoidalPosEmb,
    time_mlp: TimestepEmbedding,
    delta_time_mlp: TimestepEmbedding,
    decoder: MiniCPMModel,
    /// Cached output of `delta_time_mlp(time_embeddings(zeros))`.
    /// When `mean_mode=false` in `UnifiedCFM`, `dt_in` is always all-zeros, making this
    /// result a compile-time constant per forward session.  We lazily populate it on the
    /// first call and reuse it for every subsequent Euler step, eliminating ~8 redundant
    /// `SinusoidalPosEmb + TimestepEmbedding` dispatches per token step.
    zero_dt_emb_cache: Option<(usize, DType, Tensor)>,
    // DiT prefix K/V reuse across Euler steps was evaluated and deferred: the DiT sequence
    // is only ~6 tokens (mu + cond + x) so attention is not the dominant cost; MLP matmuls
    // dominate and require a full-sequence forward.  See metal_rtf benchmark (~0.82 RTF).
}

impl VoxCPMLocDiT {
    pub fn new(
        vb: VarBuilder,
        config: VoxMiniCPM4Config,
        in_channels: usize,
        qctx: QuantBuildCtx,
    ) -> Result<Self> {
        let in_proj = linear_x(
            in_channels,
            config.hidden_size,
            vb.pp("in_proj"),
            &qctx.pp("in_proj"),
            true,
        )?;
        let cond_proj = linear_x(
            in_channels,
            config.hidden_size,
            vb.pp("cond_proj"),
            &qctx.pp("cond_proj"),
            true,
        )?;
        let out_proj = linear_x(
            config.hidden_size,
            in_channels,
            vb.pp("out_proj"),
            &qctx.pp("out_proj"),
            true,
        )?;
        let time_embeddings = SinusoidalPosEmb::new(config.hidden_size)?;
        let time_mlp = TimestepEmbedding::new(
            vb.pp("time_mlp"),
            config.hidden_size,
            config.hidden_size,
            None,
            &qctx.pp("time_mlp"),
        )?;
        let delta_time_mlp = TimestepEmbedding::new(
            vb.pp("delta_time_mlp"),
            config.hidden_size,
            config.hidden_size,
            None,
            &qctx.pp("delta_time_mlp"),
        )?;
        assert_eq!(config.vocab_size, 0, "vocab_size must be 0 for local DiT");
        let decoder = MiniCPMModel::new(vb.pp("decoder"), config.clone(), qctx.pp("decoder"))?;
        Ok(Self {
            in_proj,
            cond_proj,
            out_proj,
            time_embeddings,
            time_mlp,
            delta_time_mlp,
            decoder,
            zero_dt_emb_cache: None,
        })
    }

    fn timestep_embedding(&mut self, t: &Tensor, dtype: DType) -> Result<Tensor> {
        let t = self.time_embeddings.forward(t, 1000)?.to_dtype(dtype)?;
        self.time_mlp.forward(&t)
    }

    pub fn forward(
        &mut self,
        x: &Tensor,
        mu: &Tensor,
        t: &Tensor,
        cond: &Tensor,
        // None = dt is all-zeros (mean_mode=false); embedding is cached. Some = mean_mode=true.
        dt: Option<&Tensor>,
        cond_is_projected: bool,
    ) -> Result<Tensor> {
        let dtype = x.dtype();
        let t = self.timestep_embedding(t, dtype)?;
        self.forward_with_t_emb(x, mu, &t, cond, dt, cond_is_projected)
    }

    pub fn forward_with_t_emb(
        &mut self,
        x: &Tensor,
        mu: &Tensor,
        t: &Tensor,
        cond: &Tensor,
        // None = dt is all-zeros (mean_mode=false); embedding is cached. Some = mean_mode=true.
        dt: Option<&Tensor>,
        cond_is_projected: bool,
    ) -> Result<Tensor> {
        let x = self.in_proj.forward(x)?;
        let cond = if cond_is_projected {
            cond.to_owned()
        } else {
            self.cond_proj.forward(cond)?
        };
        let prefix = cond.dim(1)?;
        let dtype = x.dtype();
        let dt_emb = match dt {
            None => {
                // dt is all-zeros: result is constant — compute once and reuse.
                let b = t.dim(0)?;
                let need_new = self.zero_dt_emb_cache.as_ref().map_or(
                    true,
                    |(cached_b, cached_dtype, cached)| {
                        *cached_b != b
                            || *cached_dtype != dtype
                            || cached.device().location() != t.device().location()
                    },
                );
                if need_new {
                    let zero_dt = Tensor::zeros(b, dtype, t.device())?;
                    let emb = self
                        .time_embeddings
                        .forward(&zero_dt, 1000)?
                        .to_dtype(dtype)?;
                    self.zero_dt_emb_cache = Some((b, dtype, self.delta_time_mlp.forward(&emb)?));
                }
                self.zero_dt_emb_cache.as_ref().unwrap().2.clone()
            }
            Some(dt) => {
                let emb = self.time_embeddings.forward(dt, 1000)?.to_dtype(dtype)?;
                self.delta_time_mlp.forward(&emb)?
            }
        };
        let t_total = t.add(&dt_emb)?;

        let hidden_dim = x.dim(D::Minus1)?;
        let mu_width = mu.dim(D::Minus1)?;
        let (x, prefix_tokens) = if mu_width == hidden_dim {
            (
                Tensor::cat(&[mu.add(&t_total)?.unsqueeze(1)?, cond, x], 1)?,
                1,
            )
        } else {
            let b = mu.dim(0)?;
            let mu_tokens = mu.reshape((b, (), hidden_dim))?;
            let prefix_tokens = mu_tokens.dim(1)? + 1;
            (
                Tensor::cat(&[&mu_tokens, &t_total.unsqueeze(1)?, &cond, &x], 1)?,
                prefix_tokens,
            )
        };
        let hidden = self.decoder.forward(&x, 0, false)?;
        let select_start = prefix + prefix_tokens;
        let select_len = hidden.dim(1)? - select_start;
        let hidden = hidden.narrow(1, select_start, select_len)?;
        let hidden = self.out_proj.forward(&hidden)?;
        Ok(hidden)
    }
}

pub struct UnifiedCFM {
    in_channels: usize,
    mean_mode: bool,
    estimator: VoxCPMLocDiT,
    t_span_cache: Option<(usize, f64, Tensor)>,
    solver_cache: Option<(usize, f64, usize, DType, Vec<Tensor>, Vec<Tensor>)>,
    /// `Tensor::ones(1, …)` reused across Euler steps when CFG scale is not identity.
    euler_cfg_ones: Option<(DType, Tensor)>,
    /// `Tensor::zeros_like(mu)` for the negative CFG branch when `mean_mode` is false.
    mu_zeros_cache: Option<(Vec<usize>, DType, Tensor)>,
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
            t_span_cache: None,
            solver_cache: None,
            euler_cfg_ones: None,
            mu_zeros_cache: None,
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
        let device = mu.device();
        let z = Tensor::randn(0.0f32, 1.0, (b, t, self.in_channels), device)?
            .to_dtype(dtype)?
            .affine(temperature, 0.0)?;

        let t_span = if let Some((cached_n, cached_sway, cached_t_span)) = &self.t_span_cache {
            if *cached_n == n_timesteps
                && *cached_sway == sway_sampling_coef
                && cached_t_span.device().location() == device.location()
            {
                cached_t_span.clone()
            } else {
                let ts = self.compute_t_span(n_timesteps, sway_sampling_coef, device, dtype)?;
                self.t_span_cache = Some((n_timesteps, sway_sampling_coef, ts.clone()));
                ts
            }
        } else {
            let ts = self.compute_t_span(n_timesteps, sway_sampling_coef, device, dtype)?;
            self.t_span_cache = Some((n_timesteps, sway_sampling_coef, ts.clone()));
            ts
        };

        self.ensure_solver_cache(&t_span, n_timesteps, sway_sampling_coef, 2 * b, dtype)?;
        let x = self.solve_euler(&z, &t_span, mu, cond, cfg_value, use_cfg_zero_star)?;
        Ok(x)
    }

    /// Fills `solver_cache` Euler timestep embeddings and `dt` tensors without cloning them per
    /// `forward` call (see `solve_euler`).
    fn ensure_solver_cache(
        &mut self,
        t_span: &Tensor,
        n_timesteps: usize,
        sway_sampling_coef: f64,
        batch_size: usize,
        dtype: DType,
    ) -> Result<()> {
        let need_new = self.solver_cache.as_ref().map_or(true, |cache| {
            let (cached_n, cached_sway, cached_b, cached_dtype, cached_embs, cached_dts) = cache;
            *cached_n != n_timesteps
                || *cached_sway != sway_sampling_coef
                || *cached_b != batch_size
                || *cached_dtype != dtype
                || cached_embs.first().map_or(true, |emb| {
                    emb.device().location() != t_span.device().location()
                })
                || cached_dts.first().map_or(true, |dt| {
                    dt.device().location() != t_span.device().location()
                })
        });

        if need_new {
            let mut embeddings = Vec::with_capacity(n_timesteps);
            let mut dt_steps = Vec::with_capacity(n_timesteps);
            for idx in 0..n_timesteps {
                let t_in = t_span.i(idx)?.broadcast_as(batch_size)?;
                embeddings.push(self.estimator.timestep_embedding(&t_in, dtype)?);
                dt_steps.push(t_span.i(idx)?.sub(&t_span.i(idx + 1)?)?);
            }
            self.solver_cache = Some((
                n_timesteps,
                sway_sampling_coef,
                batch_size,
                dtype,
                embeddings,
                dt_steps,
            ));
        }
        Ok(())
    }

    fn compute_t_span(
        &self,
        n_timesteps: usize,
        sway_sampling_coef: f64,
        device: &Device,
        dtype: DType,
    ) -> Result<Tensor> {
        let t_span = linspace(1.0, 0.0, n_timesteps + 1, device)?.to_dtype(dtype)?;
        let t_span = t_span
            .affine(std::f64::consts::PI / 2.0, 0.0)?
            .cos()?
            .affine(1.0, -1.0)?
            .add(&t_span)?
            .affine(sway_sampling_coef, 0.0)?
            .add(&t_span)?;
        Ok(t_span)
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
        let (t_embs, dt_steps) = match &self.solver_cache {
            Some((_, _, _, _, emb, dt)) => (emb.clone(), dt.clone()),
            None => {
                anyhow::bail!("solve_euler: solver cache missing; call ensure_solver_cache first")
            }
        };
        let t_span_len = t_span.dim(0)?;
        let zero_init_steps = max(1, (t_span_len as f32 * 0.04) as usize);
        let mut x = x.to_owned();

        let b = x.dim(0)?;
        let dtype = x.dtype();
        let device = x.device().clone();
        let mu_dims = mu.dims().to_vec();
        // `mu` and `cond` change every latent step — only cache within this solve_euler call,
        // never across UnifiedCFM::forward invocations (shape-keyed cross-call cache was incorrect).
        let mu_in = if self.mean_mode {
            Tensor::cat(&[mu, mu], 0)?
        } else {
            let mu_zeros = match &self.mu_zeros_cache {
                Some((cached_dims, cached_dtype, cached_t))
                    if cached_dims == &mu_dims
                        && *cached_dtype == dtype
                        && cached_t.device().location() == device.location() =>
                {
                    cached_t.clone()
                }
                _ => {
                    let z = Tensor::zeros(mu.dims(), dtype, &device)?;
                    self.mu_zeros_cache = Some((mu_dims, dtype, z.clone()));
                    z
                }
            };
            Tensor::cat(&[mu, &mu_zeros], 0)?
        };
        let cond_proj = self.estimator.cond_proj.forward(cond)?;
        let cond_in = Tensor::cat(&[&cond_proj, &cond_proj], 0)?;

        let b2 = 2 * b;

        let n_euler_steps = t_span_len - 1;
        // CFG matters most in early denoising; skip the 2× batch on the second half.
        let cfg_full_steps = (n_euler_steps as f64 * 0.5).ceil() as usize;

        for step in 1..t_span_len {
            let dt = &dt_steps[step - 1];
            let next_dphi_dt = {
                if use_cfg_zero_star && step <= zero_init_steps {
                    Tensor::zeros(x.dims(), x.dtype(), x.device())?
                } else if cfg_value != 1.0 && step <= cfg_full_steps {
                    let x_in = Tensor::cat(&[&x, &x], 0)?;
                    // When mean_mode=false, dt_in is always zeros; pass None so the
                    // estimator can use its cached zero-dt embedding instead of
                    // recomputing SinusoidalPosEmb + delta_time_mlp every Euler step.
                    let dt_opt = if self.mean_mode {
                        Some(dt.broadcast_as(b2)?)
                    } else {
                        None
                    };
                    let dphi_dt_combined = self.estimator.forward_with_t_emb(
                        &x_in,
                        &mu_in,
                        &t_embs[step - 1],
                        &cond_in,
                        dt_opt.as_ref(),
                        true,
                    )?;
                    let split = dphi_dt_combined.chunk(2, 0)?;
                    let dphi_dt_pos = &split[0];
                    let cfg_dphi_dt = &split[1];

                    if use_cfg_zero_star {
                        // Compute st_star: ones when cfg_zero_star is off, adaptive scale when on.
                        let positive_flat = dphi_dt_pos.reshape((b, ()))?;
                        let negative_flat = cfg_dphi_dt.reshape((b, ()))?;
                        let scale = self.optimized_scale(&positive_flat, &negative_flat)?;
                        let mut vec_shape = vec![b];
                        vec_shape.extend(vec![1; dphi_dt_pos.rank() - 1]);
                        let st_star = scale.reshape(vec_shape)?;
                        let cfg = cfg_dphi_dt.broadcast_mul(&st_star)?;
                        cfg.add(&dphi_dt_pos.sub(&cfg)?.affine(cfg_value, 0.0)?)?
                    } else {
                        let st_star = match &self.euler_cfg_ones {
                            Some((cached_dtype, cached_t))
                                if *cached_dtype == dtype
                                    && cached_t.device().location() == device.location() =>
                            {
                                cached_t.clone()
                            }
                            _ => {
                                let one = Tensor::ones(1, dtype, &device)?;
                                self.euler_cfg_ones = Some((dtype, one.clone()));
                                one
                            }
                        };
                        let cfg = cfg_dphi_dt.broadcast_mul(&st_star)?;
                        cfg.add(&dphi_dt_pos.sub(&cfg)?.affine(cfg_value, 0.0)?)?
                    }
                } else {
                    // Positive-only branch: halve DiT cost on late Euler steps.
                    let dt_opt = if self.mean_mode {
                        Some(dt.broadcast_as(b)?)
                    } else {
                        None
                    };
                    let t_emb_b = t_embs[step - 1].narrow(0, 0, b)?;
                    self.estimator.forward_with_t_emb(
                        &x,
                        mu,
                        &t_emb_b,
                        &cond_proj,
                        dt_opt.as_ref(),
                        true,
                    )?
                }
            };
            x = x.broadcast_sub(&next_dphi_dt.broadcast_mul(&dt)?)?;
        }
        Ok(x)
    }
}

pub struct VoxCPMLocEnc {
    special_token: Tensor,
    in_proj: LinearX,
    encoder: MiniCPMModel,
    hidden_size: usize,
}

impl VoxCPMLocEnc {
    pub fn new(
        vb: VarBuilder,
        config: VoxMiniCPM4Config,
        input_dim: usize,
        qctx: QuantBuildCtx,
    ) -> Result<Self> {
        let special_token = vb.get((1, 1, 1, config.hidden_size), "special_token")?;
        let in_proj = linear_x(
            input_dim,
            config.hidden_size,
            vb.pp("in_proj"),
            &qctx.pp("in_proj"),
            true,
        )?;
        assert_eq!(
            config.vocab_size, 0,
            "vocab_size must be 0 for local encoder"
        );
        let hidden_size = config.hidden_size;
        let encoder = MiniCPMModel::new(vb.pp("encoder"), config, qctx.pp("encoder"))?;
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
    ref_audio_start_token: u32,
    ref_audio_end_token: u32,
    chunk_size: usize,
    sample_rate: usize,
    #[allow(dead_code)]
    out_sample_rate: usize,
    tokenizer: SingleChineseTokenizer,
    audio_vae: AudioVAE,
    base_lm: MiniCPMModel,
    residual_lm: MiniCPMModel,
    feat_encoder: VoxCPMLocEnc,
    feat_decoder: UnifiedCFM,
    fsq_layer: ScalarQuantizationLayer,
    enc_to_lm_proj: LinearX,
    lm_to_dit_proj: LinearX,
    res_to_dit_proj: LinearX,
    fusion_concat_proj: Option<LinearX>,
    stop_proj: LinearX,
    stop_head: LinearX,
    device: Device,
    dtype: DType,
    quant_stats: QuantStats,
}

impl VoxCPMModel {
    pub fn new(
        vb: VarBuilder,
        config: VoxCPMConfig,
        tokenizer: SingleChineseTokenizer,
        audio_vae: AudioVAE,
        quant: VoxCPMQuantConfig,
    ) -> Result<Self> {
        let qroot = QuantBuildCtx::root(quant);
        let base_lm = MiniCPMModel::new(
            vb.pp("base_lm"),
            config.lm_config.clone(),
            qroot.pp("base_lm"),
        )?;
        let audio_start_token = 101usize;
        let ref_audio_start_token = 103u32;
        let ref_audio_end_token = 104u32;
        let mut residual_lm_config = config.lm_config.clone();
        residual_lm_config.num_hidden_layers = config.residual_lm_num_layers;
        residual_lm_config.vocab_size = 0;
        residual_lm_config.no_rope = config.residual_lm_no_rope;
        let residual_lm = MiniCPMModel::new(
            vb.pp("residual_lm"),
            residual_lm_config,
            qroot.pp("residual_lm"),
        )?;
        let mut encoder_config = config.lm_config.clone();
        encoder_config.hidden_size = config.encoder_config.hidden_dim;
        encoder_config.intermediate_size = config.encoder_config.ffn_dim;
        encoder_config.num_attention_heads = config.encoder_config.num_heads;
        encoder_config.num_hidden_layers = config.encoder_config.num_layers;
        encoder_config.kv_channels = config.encoder_config.kv_channels;
        encoder_config.vocab_size = 0;
        let feat_encoder = VoxCPMLocEnc::new(
            vb.pp("feat_encoder"),
            encoder_config,
            config.feat_dim,
            qroot.pp("feat_encoder"),
        )?;

        let mut decoder_config = config.lm_config.clone();
        decoder_config.hidden_size = config.dit_config.hidden_dim;
        decoder_config.intermediate_size = config.dit_config.ffn_dim;
        decoder_config.num_attention_heads = config.dit_config.num_heads;
        decoder_config.num_hidden_layers = config.dit_config.num_layers;
        decoder_config.kv_channels = config.dit_config.kv_channels;
        decoder_config.vocab_size = 0;
        let estimator = VoxCPMLocDiT::new(
            vb.pp("feat_decoder.estimator"),
            decoder_config,
            config.feat_dim,
            qroot.pp("feat_decoder.estimator"),
        )?;
        let mean_mode = config.dit_config.mean_mode.unwrap_or(false);
        let feat_decoder = UnifiedCFM::new(
            config.feat_dim,
            config.dit_config.cfm_config.clone(),
            estimator,
            mean_mode,
        )?;
        let fsq_layer = ScalarQuantizationLayer::new(
            vb.pp("fsq_layer"),
            config.lm_config.hidden_size,
            config.lm_config.hidden_size,
            config.scalar_quantization_latent_dim,
            config.scalar_quantization_scale,
            &qroot.pp("fsq_layer"),
        )?;
        let enc_to_lm_proj = linear_x(
            config.encoder_config.hidden_dim,
            config.lm_config.hidden_size,
            vb.pp("enc_to_lm_proj"),
            &qroot.pp("enc_to_lm_proj"),
            true,
        )?;
        let lm_to_dit_proj = linear_x(
            config.lm_config.hidden_size,
            config.dit_config.hidden_dim,
            vb.pp("lm_to_dit_proj"),
            &qroot.pp("lm_to_dit_proj"),
            true,
        )?;
        let res_to_dit_proj = linear_x(
            config.lm_config.hidden_size,
            config.dit_config.hidden_dim,
            vb.pp("res_to_dit_proj"),
            &qroot.pp("res_to_dit_proj"),
            true,
        )?;

        let fusion_concat_proj = if config.is_voxcpm2() {
            Some(linear_x(
                config.lm_config.hidden_size * 2,
                config.lm_config.hidden_size,
                vb.pp("fusion_concat_proj"),
                &qroot.pp("fusion_concat_proj"),
                true,
            )?)
        } else {
            None
        };

        let stop_proj = linear_x(
            config.lm_config.hidden_size,
            config.lm_config.hidden_size,
            vb.pp("stop_proj"),
            &qroot.pp("stop_proj"),
            true,
        )?;
        let stop_head = linear_x(
            config.lm_config.hidden_size,
            2,
            vb.pp("stop_head"),
            &qroot.pp("stop_head"),
            false,
        )?;

        let patch_size = config.patch_size;
        let quant_stats = qroot.stats();
        Ok(Self {
            config,
            patch_size,
            audio_start_token,
            ref_audio_start_token,
            ref_audio_end_token,
            chunk_size: audio_vae.chunk_size,
            sample_rate: audio_vae.sample_rate,
            out_sample_rate: audio_vae.out_sample_rate,
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
            fusion_concat_proj,
            stop_proj,
            stop_head,
            device: vb.device().clone(),
            dtype: vb.dtype(),
            quant_stats,
        })
    }

    #[must_use]
    pub fn quant_stats(&self) -> QuantStats {
        self.quant_stats.clone()
    }

    #[must_use]
    pub fn device(&self) -> &Device {
        &self.device
    }

    fn fusion_forward(&self, left: &Tensor, right: &Tensor) -> Result<Tensor> {
        if let Some(fusion) = &self.fusion_concat_proj {
            Ok(fusion.forward(&Tensor::cat(&[left, right], D::Minus1)?)?)
        } else {
            Ok(left.add(right)?)
        }
    }

    fn dit_hidden_from_lm(&self, lm_hidden: &Tensor, residual_hidden: &Tensor) -> Result<Tensor> {
        if self.fusion_concat_proj.is_some() {
            let dit_hidden_1 = self.lm_to_dit_proj.forward(lm_hidden)?;
            let dit_hidden_2 = self.res_to_dit_proj.forward(residual_hidden)?;
            Ok(Tensor::cat(&[&dit_hidden_1, &dit_hidden_2], D::Minus1)?)
        } else {
            let dit_hidden_1 = self.lm_to_dit_proj.forward(lm_hidden)?;
            let dit_hidden_2 = self.res_to_dit_proj.forward(residual_hidden)?;
            Ok(dit_hidden_1.add(&dit_hidden_2)?)
        }
    }

    fn preprocess_audio_to_feat(&self, wav_path: &str) -> Result<Tensor> {
        let mut audio = load_audio_with_resample(wav_path, &self.device, Some(self.sample_rate))?;
        let patch_len = self.patch_size * self.chunk_size;
        if audio.dim(1)? % patch_len != 0 {
            audio = audio.pad_with_zeros(D::Minus1, patch_len - audio.dim(1)? % patch_len, 0)?;
        }
        let audio_feat = self.audio_vae.encode(&audio, Some(self.sample_rate))?;
        let audio_feat = audio_feat
            .reshape((self.audio_vae.latent_dim, (), self.patch_size))?
            .permute((1, 2, 0))?;
        Ok(audio_feat)
    }

    fn prepare_full_inputs(
        &self,
        target_text: String,
        prompt_text: Option<String>,
        prompt_wav_path: Option<String>,
        prompt_cache: Option<&HashMap<String, Tensor>>,
    ) -> Result<(
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        usize,
        Option<VoxCPM2InputLayout>,
    )> {
        let target_text_length = self.tokenizer.encode(target_text.clone())?.len();

        if let Some(cache) = prompt_cache {
            let target_text_token = self.tokenizer.encode(target_text.clone())?;
            let target_text_token =
                Tensor::from_slice(&target_text_token, target_text_token.len(), &self.device)?;
            let text_token = match cache.get("text_token") {
                Some(token) => Tensor::cat(&[token, &target_text_token], 0)?,
                None => target_text_token,
            };
            let audio_start = Tensor::new(vec![self.audio_start_token as u32], &self.device)?;
            let mut text_token = Tensor::cat(&[text_token, audio_start], D::Minus1)?;
            let text_length = text_token.dim(0)?;
            let (audio_length, cached_feat) = match cache.get("audio_feat") {
                Some(feat) => (feat.dim(0)?, Some(feat.clone())),
                None => (0, None),
            };
            if audio_length > 0 {
                let text_pad_token = Tensor::zeros(audio_length, DType::U32, &self.device)?;
                text_token = Tensor::cat(&[text_token, text_pad_token], D::Minus1)?;
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
                let feat = cached_feat.unwrap().to_dtype(self.dtype)?;
                if self.config.is_voxcpm2() {
                    let layout = VoxCPM2InputLayout {
                        has_audio: audio_length > 0,
                        audio_ranges: vec![(text_length, text_length + audio_length)],
                    };
                    return Ok((
                        text_token,
                        text_mask,
                        feat,
                        audio_mask,
                        target_text_length,
                        Some(layout),
                    ));
                }
                let audio_pad_feat = Tensor::zeros(
                    (text_length, self.patch_size, self.audio_vae.latent_dim),
                    self.dtype,
                    &self.device,
                )?;
                let audio_feat = Tensor::cat(&[audio_pad_feat, feat], 0)?;
                return Ok((
                    text_token,
                    text_mask,
                    audio_feat,
                    audio_mask,
                    target_text_length,
                    None,
                ));
            }
            let audio_feat = Tensor::zeros(
                (text_length, self.patch_size, self.audio_vae.latent_dim),
                self.dtype,
                &self.device,
            )?;
            let text_mask = Tensor::ones(text_length, self.dtype, &self.device)?;
            let audio_mask = Tensor::zeros(text_length, self.dtype, &self.device)?;
            return Ok((
                text_token,
                text_mask,
                audio_feat,
                audio_mask,
                target_text_length,
                if self.config.is_voxcpm2() {
                    Some(VoxCPM2InputLayout {
                        has_audio: false,
                        audio_ranges: vec![],
                    })
                } else {
                    None
                },
            ));
        }

        let has_prompt_text = prompt_text.is_some();
        let encoded_text = if let Some(p_text) = prompt_text {
            p_text + &target_text
        } else {
            target_text.clone()
        };
        let text_token = self.tokenizer.encode(encoded_text)?;
        let mut text_token = Tensor::from_slice(&text_token, text_token.len(), &self.device)?;
        let audio_feat = if let Some(path) = prompt_wav_path {
            Some(self.preprocess_audio_to_feat(&path)?)
        } else {
            None
        };

        let audio_start = Tensor::new(vec![self.audio_start_token as u32], &self.device)?;

        if let Some(feat) = audio_feat {
            let audio_length = feat.dim(0)?;
            if !has_prompt_text && self.config.is_voxcpm2() {
                let target_only = self.tokenizer.encode(target_text.clone())?;
                let mut text_token =
                    Tensor::from_slice(&target_only, target_only.len(), &self.device)?;
                text_token = Tensor::cat(&[text_token, audio_start], D::Minus1)?;
                let text_length = text_token.dim(0)?;
                let ref_start = Tensor::new(vec![self.ref_audio_start_token], &self.device)?;
                let ref_end = Tensor::new(vec![self.ref_audio_end_token], &self.device)?;
                let ref_token = Tensor::zeros(audio_length, DType::U32, &self.device)?;
                text_token = Tensor::cat(&[&ref_start, &ref_token, &ref_end, &text_token], 0)?;
                let text_mask = Tensor::cat(
                    &[
                        Tensor::zeros(1, self.dtype, &self.device)?,
                        Tensor::zeros(audio_length, self.dtype, &self.device)?,
                        Tensor::zeros(1, self.dtype, &self.device)?,
                        Tensor::ones(text_length, self.dtype, &self.device)?,
                    ],
                    D::Minus1,
                )?;
                let audio_mask = Tensor::cat(
                    &[
                        Tensor::zeros(1, self.dtype, &self.device)?,
                        Tensor::ones(audio_length, self.dtype, &self.device)?,
                        Tensor::zeros(1, self.dtype, &self.device)?,
                        Tensor::zeros(text_length, self.dtype, &self.device)?,
                    ],
                    D::Minus1,
                )?;
                return Ok((
                    text_token,
                    text_mask,
                    feat,
                    audio_mask,
                    target_text_length,
                    Some(VoxCPM2InputLayout {
                        has_audio: true,
                        audio_ranges: vec![(1, 1 + audio_length)],
                    }),
                ));
            }

            text_token = Tensor::cat(&[text_token, audio_start], D::Minus1)?;
            let text_length = text_token.dim(0)?;
            let text_pad_token = Tensor::zeros(audio_length, DType::U32, &self.device)?;
            text_token = Tensor::cat(&[text_token, text_pad_token], D::Minus1)?;
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
            if self.config.is_voxcpm2() {
                return Ok((
                    text_token,
                    text_mask,
                    feat,
                    audio_mask,
                    target_text_length,
                    Some(VoxCPM2InputLayout {
                        has_audio: true,
                        audio_ranges: vec![(text_length, text_length + audio_length)],
                    }),
                ));
            }
            let audio_pad_feat = Tensor::zeros(
                (text_length, self.patch_size, self.audio_vae.latent_dim),
                feat.dtype(),
                &self.device,
            )?;
            let audio_feat = Tensor::cat(&[audio_pad_feat, feat], 0)?;
            Ok((
                text_token,
                text_mask,
                audio_feat,
                audio_mask,
                target_text_length,
                None,
            ))
        } else {
            text_token = Tensor::cat(&[text_token, audio_start], D::Minus1)?;
            let text_length = text_token.dim(0)?;
            let audio_feat = Tensor::zeros(
                (text_length, self.patch_size, self.audio_vae.latent_dim),
                self.dtype,
                &self.device,
            )?;
            let text_mask = Tensor::ones(text_length, self.dtype, &self.device)?;
            let audio_mask = Tensor::zeros(text_length, self.dtype, &self.device)?;
            Ok((
                text_token,
                text_mask,
                audio_feat,
                audio_mask,
                target_text_length,
                if self.config.is_voxcpm2() {
                    Some(VoxCPM2InputLayout {
                        has_audio: false,
                        audio_ranges: vec![],
                    })
                } else {
                    None
                },
            ))
        }
    }

    pub fn generate(
        &mut self,
        target_text: String,
        prompt_text: Option<String>,
        prompt_wav_path: Option<String>,
        config: VoxCPMGenerationConfig,
    ) -> Result<Tensor> {
        let (text_token, text_mask, audio_feat, audio_mask, target_text_length, _layout) =
            self.prepare_full_inputs(target_text, prompt_text, prompt_wav_path, None)?;
        let params = GenerationParams::resolve(config, target_text_length);
        let decode_audio = self._generate(
            &text_token,
            &text_mask,
            &audio_feat,
            &audio_mask,
            params,
            None,
        )?;
        Ok(decode_audio)
    }

    fn _generate(
        &mut self,
        text_token: &Tensor,
        text_mask: &Tensor,
        audio_feat: &Tensor,
        audio_mask: &Tensor,
        params: GenerationParams,
        layout: Option<VoxCPM2InputLayout>,
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
            params,
            layout,
        )?;
        let vae_start = stage_capture_enabled().then(Instant::now);
        let decode_audio = self.audio_vae.decode(&latent_pred, None)?.squeeze(1)?;
        if let Some(start) = vae_start {
            StageProfile::add_vae_decode(start.elapsed());
        }
        let boundary_trim = self.chunk_size;
        let audio_len = decode_audio.dim(D::Minus1)?;
        let decode_audio = decode_audio.narrow(
            D::Minus1,
            boundary_trim,
            audio_len.saturating_sub(2 * boundary_trim),
        )?;
        Ok(decode_audio)
    }

    fn _generate_stream(
        &mut self,
        text_token: &Tensor,
        text_mask: &Tensor,
        audio_feat: &Tensor,
        audio_mask: &Tensor,
        params: GenerationParams,
        layout: Option<VoxCPM2InputLayout>,
    ) -> Result<VoxCPMGenerateStream<'_>> {
        let text_token = text_token.unsqueeze(0)?;
        let text_mask = text_mask.unsqueeze(0)?;
        let audio_feat = audio_feat.unsqueeze(0)?.to_dtype(self.dtype)?;
        let audio_mask = audio_mask.unsqueeze(0)?;
        let boundary_trim_samples = self.chunk_size;
        let inf_stream = self.inference_stream(
            &text_token,
            &text_mask,
            &audio_feat,
            &audio_mask,
            params,
            layout,
        )?;

        Ok(VoxCPMGenerateStream {
            inf_stream,
            vae_state: None,
            pending_latents: Vec::with_capacity(params.stream_decode_initial_latent_batch),
            stream_decode_latent_batch: params.stream_decode_latent_batch,
            stream_decode_initial_latent_batch: params.stream_decode_initial_latent_batch,
            initial_decode_done: false,
            pending_chunk: None,
            boundary_trim_samples,
            leading_trim_remaining: boundary_trim_samples,
            profile_enabled: profile_stream_enabled() || stage_capture_enabled(),
            profile_inf_time: Duration::ZERO,
            profile_vae_time: Duration::ZERO,
            profile_prefill_time: Duration::ZERO,
            profile_chunks: 0,
            finished: false,
        })
    }

    fn inference(
        &mut self,
        text: &Tensor,
        text_mask: &Tensor,
        feat: &Tensor,
        feat_mask: &Tensor,
        params: GenerationParams,
        layout: Option<VoxCPM2InputLayout>,
    ) -> Result<Tensor> {
        let mut pred_feat_seq = Vec::with_capacity(params.max_len);
        for chunk in self.inference_stream(text, text_mask, feat, feat_mask, params, layout)? {
            pred_feat_seq.push(chunk?);
        }

        if pred_feat_seq.is_empty() {
            anyhow::bail!("inference produced no latent chunks");
        }

        // Each step is (B, D, P); stack along the latent time axis (last dim), not channel dim 1.
        // `AudioVAE::decode` forces contiguous internally; avoid a duplicate full-tensor copy here.
        let feat_pred = Tensor::cat(&pred_feat_seq, D::Minus1)?;
        self.base_lm.clear_kv_cache();
        self.residual_lm.clear_kv_cache();
        Ok(feat_pred)
    }

    fn inference_stream(
        &mut self,
        text: &Tensor,
        text_mask: &Tensor,
        feat: &Tensor,
        feat_mask: &Tensor,
        params: GenerationParams,
        layout: Option<VoxCPM2InputLayout>,
    ) -> Result<VoxCPMInferenceStream<'_>> {
        if self.config.is_voxcpm2() {
            return self.inference_stream_voxcpm2(
                text,
                text_mask,
                feat,
                feat_mask,
                params,
                layout.unwrap_or(VoxCPM2InputLayout {
                    has_audio: false,
                    audio_ranges: vec![],
                }),
            );
        }

        let (_, t, _, _) = feat.dims4()?;
        let feat_embed = self.feat_encoder.forward(feat)?;
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
        let text_mask_bm = text_mask.unsqueeze(D::Minus1)?;
        let feat_mask_bm = feat_mask.unsqueeze(D::Minus1)?;
        let combined_embed = text_mask_bm
            .broadcast_mul(&text_embed)?
            .add(&feat_mask_bm.broadcast_mul(&feat_embed)?)?;
        let prefix_feat_cond = feat.i((.., t - 1, ..))?;

        let position_id = 0;
        let seq_len = t;
        let enc_outputs = self
            .base_lm
            .forward_with_cache(&combined_embed, position_id)?;
        let enc_outputs = self
            .fsq_layer
            .forward(&enc_outputs)?
            .broadcast_mul(&feat_mask_bm)?
            .add(&enc_outputs.broadcast_mul(&text_mask_bm)?)?;

        let lm_hidden = enc_outputs.i((.., t - 1, ..))?;

        let input_embeds = enc_outputs.add(&feat_mask_bm.broadcast_mul(&feat_embed)?)?;
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
            params,
            finished: false,
            use_voxcpm2: false,
            feat_embed_cache: None,
            seq_tokens: t,
            step_profile: if stage_capture_enabled() {
                Some(InferenceStepProfile::default())
            } else {
                None
            },
            prefill_elapsed: None,
        })
    }

    fn inference_stream_voxcpm2(
        &mut self,
        text: &Tensor,
        _text_mask: &Tensor,
        feat: &Tensor,
        feat_mask: &Tensor,
        params: GenerationParams,
        layout: VoxCPM2InputLayout,
    ) -> Result<VoxCPMInferenceStream<'_>> {
        let prefill_start = stage_capture_enabled().then(Instant::now);
        let (_, seq_tokens) = text.dims2()?;
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

        let has_audio = layout.has_audio;
        let (combined_embed, prefix_feat_cond, feat_embed_cache) = if has_audio {
            let audio_t = feat.dim(1)?;
            let feat_embed = self.feat_encoder.forward(feat)?;
            let feat_embed = self.enc_to_lm_proj.forward(&feat_embed)?.squeeze(0)?;
            let embeds = scatter_ranges_dim0(&text_embed, &feat_embed, &layout.audio_ranges)?;
            let prefix_feat_cond = feat.i((.., audio_t - 1, ..))?;
            (embeds, prefix_feat_cond, Some(feat_embed))
        } else {
            let prefix_feat_cond = Tensor::zeros(
                (1, self.patch_size, self.audio_vae.latent_dim),
                self.dtype,
                &self.device,
            )?;
            (text_embed, prefix_feat_cond, None)
        };

        let position_id = 0;
        let seq_len = seq_tokens;
        let enc_outputs = self
            .base_lm
            .forward_with_cache(&combined_embed, position_id)?;

        let (lm_hidden, input_embeds) = if has_audio {
            let feat_embed = feat_embed_cache.as_ref().unwrap();
            let fsq_emb = self.fsq_layer.forward(&enc_outputs)?;
            let audio_mask_broadcast = feat_mask
                .gt(0.0)?
                .unsqueeze(D::Minus1)?
                .broadcast_as(fsq_emb.shape())?;
            let enc_outputs = audio_mask_broadcast.where_cond(&fsq_emb, &enc_outputs)?;
            let lm_hidden = enc_outputs.i((.., seq_tokens - 1, ..))?;
            let input_embeds = if let Some(fusion) = &self.fusion_concat_proj {
                let feat = enc_outputs.zeros_like()?;
                let feat = scatter_ranges_dim0(&feat, feat_embed, &layout.audio_ranges)?;
                fusion.forward(&Tensor::cat(&[&enc_outputs, &feat], D::Minus1)?)?
            } else {
                let feat = enc_outputs.zeros_like()?;
                let feat = scatter_ranges_dim0(&feat, feat_embed, &layout.audio_ranges)?;
                enc_outputs.add(&feat)?
            };
            (lm_hidden, input_embeds)
        } else {
            let lm_hidden = enc_outputs.i((.., seq_tokens - 1, ..))?;
            let input_embeds = if let Some(fusion) = &self.fusion_concat_proj {
                let feat = enc_outputs.zeros_like()?;
                fusion.forward(&Tensor::cat(&[&enc_outputs, &feat], D::Minus1)?)?
            } else {
                enc_outputs
            };
            (lm_hidden, input_embeds)
        };

        let residual_enc_outputs = self
            .residual_lm
            .forward_with_cache(&input_embeds, position_id)?;
        let residual_hidden = residual_enc_outputs.i((.., seq_tokens - 1, ..))?;

        let prefill_elapsed = prefill_start.map(|s| s.elapsed());

        Ok(VoxCPMInferenceStream {
            model: self,
            lm_hidden,
            residual_hidden,
            prefix_feat_cond,
            position_id,
            seq_len,
            i: 0,
            params,
            finished: false,
            use_voxcpm2: true,
            feat_embed_cache,
            seq_tokens,
            step_profile: if stage_capture_enabled() {
                Some(InferenceStepProfile::default())
            } else {
                None
            },
            prefill_elapsed,
        })
    }

    pub fn generate_stream(
        &mut self,
        target_text: String,
        prompt_text: Option<String>,
        prompt_wav_path: Option<String>,
        config: VoxCPMGenerationConfig,
    ) -> Result<VoxCPMGenerateStream<'_>> {
        let (text_token, text_mask, audio_feat, audio_mask, target_text_length, layout) =
            self.prepare_full_inputs(target_text, prompt_text, prompt_wav_path, None)?;
        let params = GenerationParams::resolve(config, target_text_length);

        self._generate_stream(
            &text_token,
            &text_mask,
            &audio_feat,
            &audio_mask,
            params,
            layout,
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
    params: GenerationParams,
    finished: bool,
    use_voxcpm2: bool,
    #[allow(dead_code)]
    feat_embed_cache: Option<Tensor>,
    #[allow(dead_code)]
    seq_tokens: usize,
    step_profile: Option<InferenceStepProfile>,
    prefill_elapsed: Option<Duration>,
}

impl<'a> Iterator for VoxCPMInferenceStream<'a> {
    type Item = Result<Tensor>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.finished || self.i >= self.params.max_len {
            return None;
        }

        let dit_hidden = match self
            .model
            .dit_hidden_from_lm(&self.lm_hidden, &self.residual_hidden)
        {
            Ok(t) => t,
            Err(e) => return Some(Err(e.into())),
        };

        let cfm_start = self.step_profile.as_ref().map(|_| Instant::now());
        let pred_feat = match self.model.feat_decoder.forward(
            &dit_hidden,
            self.params.inference_timesteps,
            self.model.patch_size,
            &self.prefix_feat_cond,
            1.0,
            self.params.cfg_value,
            1.0,
            true,
        ) {
            Ok(t) => t,
            Err(e) => return Some(Err(e.into())),
        };
        if let (Some(profile), Some(start)) = (&mut self.step_profile, cfm_start) {
            profile.record_cfm(start.elapsed());
        }

        let vae_latent = match pred_feat.permute((0, 2, 1)) {
            Ok(t) => t,
            Err(e) => return Some(Err(e.into())),
        };
        self.prefix_feat_cond = pred_feat;
        let should_check_stop = self.i > self.params.min_len
            && (self.i % self.params.stop_check_interval == 0 || self.i + 1 >= self.params.max_len);
        if should_check_stop {
            let stop_start = self.step_profile.as_ref().map(|_| Instant::now());
            let stop_flag = match self.model.stop_proj.forward(&self.lm_hidden) {
                Ok(t) => match t.silu().and_then(|t| self.model.stop_head.forward(&t)) {
                    Ok(t) => match t
                        .argmax(D::Minus1)
                        .and_then(|t| t.i(0).and_then(|t| t.to_scalar::<u32>()))
                    {
                        Ok(v) => v,
                        Err(e) => return Some(Err(e.into())),
                    },
                    Err(e) => return Some(Err(e.into())),
                },
                Err(e) => return Some(Err(e.into())),
            };
            if let (Some(profile), Some(start)) = (&mut self.step_profile, stop_start) {
                profile.stop += start.elapsed();
            }

            if stop_flag == 1 {
                self.finished = true;
            }
        }

        self.position_id += self.seq_len;
        self.seq_len = 1;

        // Skip LM decode when already finished: the next lm_hidden / residual_hidden
        // would never be consumed, so computing the feature encoder is pure waste too.
        if !self.finished {
            let lm_start = self.step_profile.as_ref().map(|_| Instant::now());
            let pred_feat_unsqueezed = match self.prefix_feat_cond.unsqueeze(1) {
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

            let curr_embed_val = match curr_embed.i((.., 0, ..)) {
                Ok(t) => t,
                Err(e) => return Some(Err(anyhow::Error::from(e))),
            };

            self.lm_hidden = match self
                .model
                .base_lm
                .forward_with_cache(&curr_embed_val, self.position_id)
            {
                Ok(t) => match t
                    .squeeze(1)
                    .map_err(anyhow::Error::from)
                    .and_then(|t| self.model.fsq_layer.forward(&t))
                {
                    Ok(t) => t,
                    Err(e) => return Some(Err(e)),
                },
                Err(e) => return Some(Err(anyhow::Error::from(e))),
            };

            self.residual_hidden = if self.use_voxcpm2 {
                match self.model.fusion_forward(&self.lm_hidden, &curr_embed_val) {
                    Ok(t) => match self
                        .model
                        .residual_lm
                        .forward_with_cache(&t, self.position_id)
                    {
                        Ok(t) => match t.squeeze(1) {
                            Ok(t) => t,
                            Err(e) => return Some(Err(anyhow::Error::from(e))),
                        },
                        Err(e) => return Some(Err(anyhow::Error::from(e))),
                    },
                    Err(e) => return Some(Err(e)),
                }
            } else {
                match self.lm_hidden.add(&curr_embed_val) {
                    Ok(t) => match self
                        .model
                        .residual_lm
                        .forward_with_cache(&t, self.position_id)
                    {
                        Ok(t) => match t.squeeze(1) {
                            Ok(t) => t,
                            Err(e) => return Some(Err(anyhow::Error::from(e))),
                        },
                        Err(e) => return Some(Err(anyhow::Error::from(e))),
                    },
                    Err(e) => return Some(Err(anyhow::Error::from(e))),
                }
            };
            if let (Some(profile), Some(start)) = (&mut self.step_profile, lm_start) {
                profile.lm_advance += start.elapsed();
            }
        }

        self.i += 1;

        Some(Ok(vae_latent))
    }
}

impl<'a> Drop for VoxCPMInferenceStream<'a> {
    fn drop(&mut self) {
        if let Some(profile) = &self.step_profile {
            profile.print_summary();
            if stage_capture_enabled() {
                StageProfile::store_inference(profile, self.prefill_elapsed);
            }
        } else if let Some(prefill) = self.prefill_elapsed {
            if stage_capture_enabled() {
                eprintln!("VOXCPM_PROFILE prefill={:.3}s", prefill.as_secs_f64());
                StageProfile::store_inference(&InferenceStepProfile::default(), Some(prefill));
            }
        }
        self.model.base_lm.clear_kv_cache();
        self.model.residual_lm.clear_kv_cache();
    }
}

pub struct VoxCPMGenerateStream<'a> {
    inf_stream: VoxCPMInferenceStream<'a>,
    vae_state: Option<crate::audio_vae::DecoderState>,
    pending_latents: Vec<Tensor>,
    stream_decode_latent_batch: usize,
    stream_decode_initial_latent_batch: usize,
    initial_decode_done: bool,
    pending_chunk: Option<Tensor>,
    /// VAE hop / chunk size; matches batch `decode()` leading+trailing trim (one chunk each end).
    boundary_trim_samples: usize,
    leading_trim_remaining: usize,
    profile_enabled: bool,
    profile_inf_time: Duration,
    profile_vae_time: Duration,
    profile_prefill_time: Duration,
    profile_chunks: usize,
    finished: bool,
}

/// Trim VAE boundary padding from a streaming decode chunk to match batch `decode()` output.
///
/// Returns `None` when the chunk is fully consumed by trim (caller should skip yield).
pub(crate) fn trim_stream_audio_chunk(
    audio: Tensor,
    boundary_trim: usize,
    leading_trim_remaining: &mut usize,
    apply_trailing_trim: bool,
) -> Result<Option<Tensor>> {
    let audio_len = audio.dim(D::Minus1)?;
    let mut start = 0usize;
    if *leading_trim_remaining > 0 {
        let lead = (*leading_trim_remaining).min(audio_len);
        start = lead;
        *leading_trim_remaining -= lead;
    }
    let available = audio_len.saturating_sub(start);
    let trailing = if apply_trailing_trim {
        boundary_trim.min(available)
    } else {
        0
    };
    let len = available.saturating_sub(trailing);
    if len == 0 {
        return Ok(None);
    }
    Ok(Some(audio.narrow(D::Minus1, start, len)?))
}

impl<'a> VoxCPMGenerateStream<'a> {
    fn prepare_stream_chunk(&mut self, audio: Tensor, final_chunk: bool) -> Result<Option<Tensor>> {
        trim_stream_audio_chunk(
            audio,
            self.boundary_trim_samples,
            &mut self.leading_trim_remaining,
            final_chunk,
        )
    }

    fn required_latent_batch(&self, force: bool) -> usize {
        if force {
            1
        } else if self.initial_decode_done {
            self.stream_decode_latent_batch
        } else {
            self.stream_decode_initial_latent_batch
        }
    }

    fn decode_pending_latents(&mut self, force: bool) -> Option<Result<Tensor>> {
        let required = self.required_latent_batch(force);
        if self.pending_latents.is_empty() || (!force && self.pending_latents.len() < required) {
            return None;
        }

        let latents = std::mem::replace(
            &mut self.pending_latents,
            Vec::with_capacity(self.stream_decode_latent_batch.max(required)),
        );
        let model_dtype = self.inf_stream.model.dtype;
        let latents: Vec<Tensor> = match latents
            .into_iter()
            .map(|t| {
                if t.dtype() == model_dtype {
                    Ok(t)
                } else {
                    t.to_dtype(model_dtype)
                }
            })
            .collect::<Result<Vec<_>, _>>()
        {
            Ok(v) => v,
            Err(e) => return Some(Err(anyhow::Error::from(e))),
        };
        let latent_refs: Vec<&Tensor> = latents.iter().collect();
        let latent_vae = match Tensor::cat(&latent_refs, D::Minus1) {
            Ok(t) => t,
            Err(e) => return Some(Err(anyhow::Error::from(e))),
        };

        let vae_start = self.profile_enabled.then(Instant::now);
        let audio_chunk = match self.inf_stream.model.audio_vae.decode_stream(
            &latent_vae,
            self.vae_state.as_mut().unwrap(),
            None,
        ) {
            Ok(t) => match t.squeeze(1) {
                Ok(t) => t,
                Err(e) => return Some(Err(anyhow::Error::from(e))),
            },
            Err(e) => return Some(Err(anyhow::Error::from(e))),
        };
        if let Some(start) = vae_start {
            self.profile_vae_time += start.elapsed();
            self.profile_chunks += latents.len();
        }
        self.initial_decode_done = true;

        Some(Ok(audio_chunk))
    }
}

impl<'a> Iterator for VoxCPMGenerateStream<'a> {
    type Item = Result<Tensor>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.finished {
            return None;
        }

        if self.profile_prefill_time.is_zero() {
            if let Some(prefill) = self.inf_stream.prefill_elapsed.take() {
                self.profile_prefill_time = prefill;
            }
        }

        loop {
            let inf_start = self.profile_enabled.then(Instant::now);
            let next_latent = self.inf_stream.next();
            if let Some(start) = inf_start {
                self.profile_inf_time += start.elapsed();
            }

            let Some(next_latent) = next_latent else {
                break;
            };

            match next_latent {
                Ok(latent) => {
                    // latent: [B, D, P]
                    let (b, _, _) = match latent.dims3() {
                        Ok(dims) => dims,
                        Err(e) => return Some(Err(anyhow::Error::from(e))),
                    };

                    if self.vae_state.is_none() {
                        // Streaming decoder state stays F32 on Metal (see `get_vae_compute_dtype`).
                        let state = match self.inf_stream.model.audio_vae.init_decoder_state(
                            b,
                            &latent.device(),
                            DType::F32,
                        ) {
                            Ok(s) => s,
                            Err(e) => return Some(Err(anyhow::Error::from(e))),
                        };
                        self.vae_state = Some(state);
                    }

                    self.pending_latents.push(latent);
                    if let Some(decoded) = self.decode_pending_latents(false) {
                        match decoded {
                            Ok(audio_chunk) => {
                                match self.prepare_stream_chunk(audio_chunk, false) {
                                    Ok(Some(trimmed)) => {
                                        if let Some(to_yield) = self.pending_chunk.replace(trimmed)
                                        {
                                            return Some(Ok(to_yield));
                                        }
                                    }
                                    Ok(None) => {}
                                    Err(e) => return Some(Err(e)),
                                }
                            }
                            Err(e) => return Some(Err(e)),
                        }
                    }
                }
                Err(e) => return Some(Err(e)),
            }
        }
        if let Some(decoded) = self.decode_pending_latents(true) {
            match decoded {
                Ok(audio_chunk) => match self.prepare_stream_chunk(audio_chunk, false) {
                    Ok(Some(trimmed)) => {
                        if let Some(to_yield) = self.pending_chunk.replace(trimmed) {
                            return Some(Ok(to_yield));
                        }
                    }
                    Ok(None) => {}
                    Err(e) => return Some(Err(e)),
                },
                Err(e) => return Some(Err(e)),
            }
        }
        if let Some(to_yield) = self.pending_chunk.take() {
            match self.prepare_stream_chunk(to_yield, true) {
                Ok(Some(trimmed)) => return Some(Ok(trimmed)),
                Ok(None) => {}
                Err(e) => return Some(Err(e)),
            }
        }

        self.finished = true;
        None
    }
}

impl<'a> Drop for VoxCPMGenerateStream<'a> {
    fn drop(&mut self) {
        if self.profile_enabled || stage_capture_enabled() {
            eprintln!(
                "VOXCPM_PROFILE_STREAM chunks={} prefill={:.3}s inference_next={:.3}s vae_decode={:.3}s latents={}",
                self.profile_chunks,
                self.profile_prefill_time.as_secs_f64(),
                self.profile_inf_time.as_secs_f64(),
                self.profile_vae_time.as_secs_f64(),
                self.inf_stream.i,
            );
            if stage_capture_enabled() {
                StageProfile::store_stream(
                    self.profile_prefill_time,
                    self.profile_inf_time,
                    self.profile_vae_time,
                    self.inf_stream.i,
                );
            }
        }
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
        let audio_feat = self
            .preprocess_audio_to_feat(&prompt_wav_path)?
            .to_dtype(self.dtype)?;
        let mut hashmap = HashMap::new();
        hashmap.insert("text_token".to_string(), text_token);
        hashmap.insert("audio_feat".to_string(), audio_feat);
        Ok(hashmap)
    }

    pub fn generate_with_prompt_cache(
        &mut self,
        target_text: String,
        prompt_cache: &HashMap<String, Tensor>,
        config: VoxCPMGenerationConfig,
    ) -> Result<Tensor> {
        let (text_token, text_mask, audio_feat, audio_mask, target_text_length, layout) =
            self.prepare_full_inputs(target_text, None, None, Some(prompt_cache))?;

        let params = GenerationParams::resolve(config, target_text_length);
        let decode_audio = self._generate(
            &text_token,
            &text_mask,
            &audio_feat,
            &audio_mask,
            params,
            layout,
        )?;
        Ok(decode_audio)
    }

    pub fn generate_stream_with_prompt_cache(
        &mut self,
        target_text: String,
        prompt_cache: &HashMap<String, Tensor>,
        config: VoxCPMGenerationConfig,
    ) -> Result<VoxCPMGenerateStream<'_>> {
        let (text_token, text_mask, audio_feat, audio_mask, target_text_length, layout) =
            self.prepare_full_inputs(target_text, None, None, Some(prompt_cache))?;

        let params = GenerationParams::resolve(config, target_text_length);
        self._generate_stream(
            &text_token,
            &text_mask,
            &audio_feat,
            &audio_mask,
            params,
            layout,
        )
    }
}

#[cfg(test)]
mod stream_trim_tests {
    use super::*;
    use candle_core::Device;

    fn mono_chunk(values: &[f32]) -> Result<Tensor> {
        Ok(Tensor::from_slice(values, values.len(), &Device::Cpu)?.unsqueeze(0)?)
    }

    fn chunk_values(chunk: &Tensor) -> Result<Vec<f32>> {
        Ok(chunk.squeeze(0)?.to_vec1()?)
    }

    #[test]
    fn leading_trim_removes_boundary_prefix() -> Result<()> {
        let boundary = 4usize;
        let mut leading = boundary;
        let audio = mono_chunk(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])?;
        let trimmed = trim_stream_audio_chunk(audio, boundary, &mut leading, false)?.unwrap();
        assert_eq!(leading, 0);
        assert_eq!(chunk_values(&trimmed)?, vec![4.0, 5.0, 6.0, 7.0]);
        Ok(())
    }

    #[test]
    fn trailing_trim_on_final_chunk() -> Result<()> {
        let boundary = 3usize;
        let mut leading = 0usize;
        let audio = mono_chunk(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])?;
        let trimmed = trim_stream_audio_chunk(audio, boundary, &mut leading, true)?.unwrap();
        assert_eq!(chunk_values(&trimmed)?, vec![1.0, 2.0, 3.0]);
        Ok(())
    }

    #[test]
    fn leading_and_trailing_trim_together() -> Result<()> {
        let boundary = 2usize;
        let mut leading = boundary;
        let audio = mono_chunk(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])?;
        let trimmed = trim_stream_audio_chunk(audio, boundary, &mut leading, true)?.unwrap();
        assert_eq!(leading, 0);
        assert_eq!(chunk_values(&trimmed)?, vec![2.0, 3.0, 4.0, 5.0]);
        Ok(())
    }

    #[test]
    fn leading_trim_spans_multiple_chunks() -> Result<()> {
        let boundary = 5usize;
        let mut leading = boundary;
        let first = mono_chunk(&[0.0, 1.0, 2.0])?;
        assert!(trim_stream_audio_chunk(first, boundary, &mut leading, false)?.is_none());
        assert_eq!(leading, 2);
        let second = mono_chunk(&[3.0, 4.0, 5.0, 6.0, 7.0])?;
        let trimmed = trim_stream_audio_chunk(second, boundary, &mut leading, false)?.unwrap();
        assert_eq!(leading, 0);
        assert_eq!(chunk_values(&trimmed)?, vec![5.0, 6.0, 7.0]);
        Ok(())
    }

    #[test]
    fn fully_consumed_by_trim_returns_none() -> Result<()> {
        let boundary = 4usize;
        let mut leading = 0usize;
        let audio = mono_chunk(&[1.0, 2.0, 3.0, 4.0])?;
        assert!(trim_stream_audio_chunk(audio, boundary, &mut leading, true)?.is_none());
        Ok(())
    }
}
