use anyhow::Result;
#[cfg(feature = "metal")]
use candle_core::DeviceLocation;
use candle_core::{D, DType, Tensor};
use candle_nn::{Activation, Module, VarBuilder};

use crate::kv_cache::KvCache;
use crate::linear::{FusedLinearX, LinearX, fused_linear_x, linear_x};
use crate::position_embed::rope::apply_rotary_pos_emb;
use crate::quant::QuantBuildCtx;
use crate::utils::tensor::repeat_kv;

/// Head dims supported by candle's Metal fused SDPA kernels.
const METAL_SDPA_HEAD_DIMS: &[usize] = &[32, 64, 72, 80, 96, 128, 256, 512];

/// Candle's Metal *vector* SDPA kernel accepts `q_seq <= 8`. Above that it routes to
/// the tiled full kernel (BQ=32), which is numerically unsafe for VoxCPM2 DiT.
const METAL_SDPA_VECTOR_MAX_QLEN: usize = 8;

/// Whether fused Metal SDPA is globally disabled (`VOXCPM_FUSED_SDPA=0`).
fn fused_sdpa_env_disabled() -> bool {
    match std::env::var("VOXCPM_FUSED_SDPA") {
        Ok(v) => {
            let v = v.trim().to_ascii_lowercase();
            v == "0" || v == "false" || v == "off" || v == "eager"
        }
        Err(_) => false,
    }
}

/// Max query length **per fused SDPA block**.
///
/// Default **1** (decode/vector kernel only). Candle's Metal vector SDPA produces NaNs
/// at `q_seq ∈ {4,8}` with VoxCPM2's GQA (16/2, head_dim=128); the tiled full kernel
/// at `q_seq > 8` previously corrupted DiT audio (corr≈0.03). Chunking longer sequences
/// into vector blocks is therefore unsafe. Set `VOXCPM_FUSED_SDPA_MAX_QLEN` only for
/// experiments; values are clamped to ≤8.
fn fused_sdpa_max_qlen() -> usize {
    let requested = std::env::var("VOXCPM_FUSED_SDPA_MAX_QLEN")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1);
    requested.clamp(1, METAL_SDPA_VECTOR_MAX_QLEN)
}

/// Eligibility for candle `ops::sdpa` on Metal (must fall back on CPU — no cpu impl).
///
/// Only sequences with `q_len <= fused_sdpa_max_qlen()` (default 1) take the fused path.
#[must_use]
pub fn fused_sdpa_eligible(
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    attention_mask: Option<&Tensor>,
) -> bool {
    if fused_sdpa_env_disabled() {
        return false;
    }
    #[cfg(not(feature = "metal"))]
    {
        let _ = (query, key, value, attention_mask);
        return false;
    }
    #[cfg(feature = "metal")]
    {
        if !matches!(query.device().location(), DeviceLocation::Metal { .. }) {
            return false;
        }
        let Ok(q_dims) = query.dims4() else {
            return false;
        };
        let Ok(k_dims) = key.dims4() else {
            return false;
        };
        let Ok(v_dims) = value.dims4() else {
            return false;
        };
        let (b, q_heads, q_len, head_dim) = q_dims;
        let (kb, kv_heads, kv_len, k_dim) = k_dims;
        let (vb, v_heads, v_len, v_dim) = v_dims;
        if b != kb || b != vb || kv_heads != v_heads || kv_len != v_len {
            return false;
        }
        if head_dim != k_dim || head_dim != v_dim {
            return false;
        }
        if !METAL_SDPA_HEAD_DIMS.contains(&head_dim) {
            return false;
        }
        if !matches!(query.dtype(), DType::F16 | DType::BF16 | DType::F32) {
            return false;
        }
        if query.dtype() != key.dtype() || query.dtype() != value.dtype() {
            return false;
        }
        if q_heads % kv_heads != 0 || q_heads == 0 || kv_heads == 0 {
            return false;
        }
        if q_len > fused_sdpa_max_qlen() || q_len > kv_len {
            return false;
        }
        if let Some(mask) = attention_mask {
            let Ok(m) = mask.dims4() else {
                return false;
            };
            if m != (b, q_heads, q_len, kv_len) {
                return false;
            }
        }
        true
    }
}

/// Unified attention entry: Metal fused SDPA when eligible, else eager GQA (no `repeat_kv`).
pub fn attention_forward(
    query_states: &Tensor,
    key_states: &Tensor,
    value_states: &Tensor,
    num_key_value_groups: Option<usize>,
    attention_mask: Option<&Tensor>,
    scaling: f64,
) -> Result<Tensor> {
    if fused_sdpa_eligible(query_states, key_states, value_states, attention_mask) {
        #[cfg(feature = "metal")]
        {
            let q = query_states.contiguous()?;
            let k = key_states.contiguous()?;
            let v = value_states.contiguous()?;
            let out = candle_nn::ops::sdpa(&q, &k, &v, attention_mask, false, scaling as f32, 1.0)?;
            return Ok(out.transpose(1, 2)?.contiguous()?);
        }
    }
    eager_attention_forward(
        query_states,
        key_states,
        value_states,
        num_key_value_groups,
        attention_mask,
        scaling,
    )
}

pub struct GateUpDownMLP {
    /// Fused gate+up when available; else separate projections.
    gate_up: GateUpProjs,
    down_proj: LinearX,
    act_fn: Activation,
}

enum GateUpProjs {
    Fused(FusedLinearX),
    Separate { gate: LinearX, up: LinearX },
}

impl GateUpDownMLP {
    pub fn new(
        vb: VarBuilder,
        hidden_size: usize,
        intermediate_size: usize,
        act_fn: Activation,
        bias: bool,
        qctx: &QuantBuildCtx,
    ) -> Result<Self> {
        // Fusion only for the common no-bias case (all VoxCPM MiniCPM layers).
        let gate_up = if !bias {
            match fused_linear_x(
                hidden_size,
                &[
                    ("gate_proj", intermediate_size),
                    ("up_proj", intermediate_size),
                ],
                vb.clone(),
                qctx,
            )? {
                Some(fused) => GateUpProjs::Fused(fused),
                None => GateUpProjs::Separate {
                    gate: linear_x(
                        hidden_size,
                        intermediate_size,
                        vb.pp("gate_proj"),
                        &qctx.pp("gate_proj"),
                        false,
                    )?,
                    up: linear_x(
                        hidden_size,
                        intermediate_size,
                        vb.pp("up_proj"),
                        &qctx.pp("up_proj"),
                        false,
                    )?,
                },
            }
        } else {
            GateUpProjs::Separate {
                gate: linear_x(
                    hidden_size,
                    intermediate_size,
                    vb.pp("gate_proj"),
                    &qctx.pp("gate_proj"),
                    bias,
                )?,
                up: linear_x(
                    hidden_size,
                    intermediate_size,
                    vb.pp("up_proj"),
                    &qctx.pp("up_proj"),
                    bias,
                )?,
            }
        };
        let down_proj = linear_x(
            intermediate_size,
            hidden_size,
            vb.pp("down_proj"),
            &qctx.pp("down_proj"),
            bias,
        )?;

        Ok(Self {
            gate_up,
            down_proj,
            act_fn,
        })
    }
}

impl Module for GateUpDownMLP {
    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        let (gate, up) = match &self.gate_up {
            GateUpProjs::Fused(fused) => {
                let parts = fused.forward_split(xs)?;
                (parts[0].clone(), parts[1].clone())
            }
            GateUpProjs::Separate { gate, up } => (xs.apply(gate)?, xs.apply(up)?),
        };
        let res = (gate.apply(&self.act_fn)? * up)?;
        res.apply(&self.down_proj)
    }
}

/// Naive multi-head attention with separate Q/K/V projections (quantized individually).
#[derive(Debug)]
pub struct NaiveAttention {
    qkv: QkvProjs,
    o_proj: LinearX,
    num_heads: usize,
    num_kv_heads: usize,
    num_kv_groups: usize,
    head_dim: usize,
    middle_size: usize,
    scale: f64,
    kv_cache: KvCache,
}

#[derive(Debug)]
enum QkvProjs {
    Fused(FusedLinearX),
    Separate { q: LinearX, k: LinearX, v: LinearX },
}

impl NaiveAttention {
    pub fn new(
        vb: VarBuilder,
        hidden_size: usize,
        num_attention_heads: usize,
        num_key_value_heads: usize,
        head_dim: Option<usize>,
        bias: bool,
        o_proj_pp_name: Option<&str>,
        qctx: &QuantBuildCtx,
    ) -> Result<Self> {
        let num_kv_groups = num_attention_heads / num_key_value_heads;
        let head_dim = head_dim.unwrap_or(hidden_size / num_attention_heads);
        let scale = 1f64 / f64::sqrt(head_dim as f64);
        let o_proj_pp_name = o_proj_pp_name.unwrap_or("o_proj");
        let q_out = num_attention_heads * head_dim;
        let kv_out = num_key_value_heads * head_dim;

        let qkv = if !bias {
            match fused_linear_x(
                hidden_size,
                &[("q_proj", q_out), ("k_proj", kv_out), ("v_proj", kv_out)],
                vb.clone(),
                qctx,
            )? {
                Some(fused) => QkvProjs::Fused(fused),
                None => QkvProjs::Separate {
                    q: linear_x(
                        hidden_size,
                        q_out,
                        vb.pp("q_proj"),
                        &qctx.pp("q_proj"),
                        false,
                    )?,
                    k: linear_x(
                        hidden_size,
                        kv_out,
                        vb.pp("k_proj"),
                        &qctx.pp("k_proj"),
                        false,
                    )?,
                    v: linear_x(
                        hidden_size,
                        kv_out,
                        vb.pp("v_proj"),
                        &qctx.pp("v_proj"),
                        false,
                    )?,
                },
            }
        } else {
            QkvProjs::Separate {
                q: linear_x(
                    hidden_size,
                    q_out,
                    vb.pp("q_proj"),
                    &qctx.pp("q_proj"),
                    bias,
                )?,
                k: linear_x(
                    hidden_size,
                    kv_out,
                    vb.pp("k_proj"),
                    &qctx.pp("k_proj"),
                    bias,
                )?,
                v: linear_x(
                    hidden_size,
                    kv_out,
                    vb.pp("v_proj"),
                    &qctx.pp("v_proj"),
                    bias,
                )?,
            }
        };
        let o_proj = linear_x(
            q_out,
            hidden_size,
            vb.pp(o_proj_pp_name),
            &qctx.pp(o_proj_pp_name),
            bias,
        )?;

        Ok(Self {
            qkv,
            o_proj,
            num_heads: num_attention_heads,
            num_kv_heads: num_key_value_heads,
            num_kv_groups,
            head_dim,
            middle_size: q_out,
            scale,
            kv_cache: KvCache::default(),
        })
    }

    #[inline]
    fn qkv(&self, xs: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
        match &self.qkv {
            QkvProjs::Fused(fused) => {
                let parts = fused.forward_split(xs)?;
                Ok((parts[0].clone(), parts[1].clone(), parts[2].clone()))
            }
            QkvProjs::Separate { q, k, v } => Ok((q.forward(xs)?, k.forward(xs)?, v.forward(xs)?)),
        }
    }

    pub fn forward(
        &self,
        xs: &Tensor,
        cos: Option<&Tensor>,
        sin: Option<&Tensor>,
        attention_mask: Option<&Tensor>,
        tof32: bool,
    ) -> Result<Tensor> {
        let (b_sz, q_len, _) = xs.dims3()?;
        let (query_states, key_states, value_states) = self.qkv(xs)?;

        let query_states = query_states
            .reshape((b_sz, q_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let key_states = key_states
            .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let value_states = value_states
            .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        let (query_states, key_states) = if let (Some(cos), Some(sin)) = (cos, sin) {
            apply_rotary_pos_emb(&query_states, &key_states, cos, sin, tof32)?
        } else {
            (query_states, key_states)
        };

        let attn_output = attention_forward(
            &query_states,
            &key_states,
            &value_states,
            Some(self.num_kv_groups),
            attention_mask,
            self.scale,
        )?;
        let attn_output = attn_output.reshape((b_sz, q_len, self.middle_size))?;
        let attn_output = attn_output.apply(&self.o_proj)?;
        Ok(attn_output)
    }

    pub fn forward_with_cache(
        &mut self,
        xs: &Tensor,
        cos: Option<&Tensor>,
        sin: Option<&Tensor>,
        attention_mask: Option<&Tensor>,
        tof32: bool,
    ) -> Result<Tensor> {
        let (b_sz, q_len, _) = xs.dims3()?;
        let (query_states, key_states, value_states) = self.qkv(xs)?;

        let query_states = query_states
            .reshape((b_sz, q_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let key_states = key_states
            .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let value_states = value_states
            .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        let (query_states, key_states) = if let (Some(cos), Some(sin)) = (cos, sin) {
            apply_rotary_pos_emb(&query_states, &key_states, cos, sin, tof32)?
        } else {
            (query_states, key_states)
        };

        self.kv_cache.append(key_states, value_states)?;
        let (key_states, value_states) = self.kv_cache.keys_values()?;

        let attn_output = attention_forward(
            &query_states,
            &key_states,
            &value_states,
            Some(self.num_kv_groups),
            attention_mask,
            self.scale,
        )?;
        let attn_output = attn_output.reshape((b_sz, q_len, self.middle_size))?;
        let attn_output = attn_output.apply(&self.o_proj)?;
        Ok(attn_output)
    }

    pub fn clear_kv_cache(&mut self) {
        self.kv_cache.clear();
    }
}

pub fn eager_attention_forward(
    query_states: &Tensor,
    key_states: &Tensor,
    value_states: &Tensor,
    num_key_value_groups: Option<usize>,
    attention_mask: Option<&Tensor>,
    scaling: f64,
) -> Result<Tensor> {
    let attn_output = match (num_key_value_groups, attention_mask.is_some()) {
        // Untiled GQA: reshape Q instead of repeating K/V (DiT / unmasked decode).
        (Some(g), false) if g > 1 => {
            eager_attention_gqa(query_states, key_states, value_states, g, scaling)?
        }
        // Masked GQA (LM prefill): classic repeat_kv so (b,1,q,k) masks broadcast.
        (Some(g), true) if g > 1 => eager_attention_inner(
            query_states,
            &repeat_kv(key_states, g)?,
            &repeat_kv(value_states, g)?,
            attention_mask,
            scaling,
        )?,
        _ => eager_attention_inner(
            query_states,
            key_states,
            value_states,
            attention_mask,
            scaling,
        )?,
    };

    let attn_output = attn_output.transpose(1, 2)?.contiguous()?;
    Ok(attn_output)
}

/// GQA without materializing `repeat_kv`: reshape Q into `(b, kv_heads, groups*q_len, d)`
/// and matmul against untiled K/V. Only used when there is no attention mask.
fn eager_attention_gqa(
    query_states: &Tensor,
    key_states: &Tensor,
    value_states: &Tensor,
    num_key_value_groups: usize,
    scaling: f64,
) -> Result<Tensor> {
    let (b, q_heads, q_len, head_dim) = query_states.dims4()?;
    let kv_heads = key_states.dim(1)?;
    debug_assert_eq!(q_heads, kv_heads * num_key_value_groups);

    let q = query_states
        .reshape((b, kv_heads, num_key_value_groups, q_len, head_dim))?
        .reshape((b, kv_heads, num_key_value_groups * q_len, head_dim))?
        .contiguous()?;

    let attn = eager_attention_inner(
        &q,
        &key_states.contiguous()?,
        &value_states.contiguous()?,
        None,
        scaling,
    )?;
    Ok(attn
        .reshape((b, kv_heads, num_key_value_groups, q_len, head_dim))?
        .reshape((b, q_heads, q_len, head_dim))?
        .contiguous()?)
}

fn eager_attention_inner(
    query_states: &Tensor,
    key_states: &Tensor,
    value_states: &Tensor,
    attention_mask: Option<&Tensor>,
    scaling: f64,
) -> Result<Tensor> {
    #[cfg(not(feature = "flash-attn"))]
    {
        let query_states = query_states.contiguous()?;
        let key_transposed = key_states.transpose(D::Minus2, D::Minus1)?;
        if std::env::var_os("VOXCPM_DEBUG_ATTN").is_some() {
            eprintln!(
                "VOXCPM_DEBUG_ATTN q={:?}s{:?} kT={:?}s{:?} v={:?}s{:?} mask={}",
                query_states.shape(),
                query_states.stride(),
                key_transposed.shape(),
                key_transposed.stride(),
                value_states.shape(),
                value_states.stride(),
                attention_mask.is_some()
            );
        }
        let mut attn_weights = query_states.matmul(&key_transposed)?.affine(scaling, 0.0)?;

        if let Some(mask) = attention_mask {
            attn_weights = attn_weights.broadcast_add(&mask.to_dtype(attn_weights.dtype())?)?;
        }

        Ok(candle_nn::ops::softmax_last_dim(&attn_weights)?.matmul(value_states)?)
    }
    #[cfg(feature = "flash-attn")]
    {
        let query_states = query_states.contiguous()?.transpose(1, 2)?;
        let key_states = key_states.contiguous()?.transpose(1, 2)?;
        let value_states = value_states.contiguous()?.transpose(1, 2)?;
        Ok(candle_flash_attn::flash_attn(
            &query_states,
            &key_states,
            &value_states,
            scaling as f32,
            attention_mask.is_some(),
        )?
        .transpose(1, 2)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn fused_sdpa_default_max_qlen_is_decode_only() {
        // Default MAX_QLEN=1: DiT stays on eager (vector SDPA NaNs at q_seq>1).
        assert_eq!(fused_sdpa_max_qlen(), 1);
    }

    #[test]
    fn fused_sdpa_ineligible_on_cpu() {
        let device = Device::Cpu;
        let q = Tensor::zeros((1, 16, 1, 128), DType::F32, &device).unwrap();
        let k = Tensor::zeros((1, 2, 1, 128), DType::F32, &device).unwrap();
        let v = Tensor::zeros((1, 2, 1, 128), DType::F32, &device).unwrap();
        assert!(
            !fused_sdpa_eligible(&q, &k, &v, None),
            "CPU must fall back to eager (candle SDPA has no cpu impl)"
        );
    }

    #[test]
    fn fused_sdpa_rejects_bad_head_dim() {
        let device = Device::Cpu;
        let q = Tensor::zeros((1, 8, 1, 48), DType::F32, &device).unwrap();
        let k = Tensor::zeros((1, 2, 1, 48), DType::F32, &device).unwrap();
        let v = Tensor::zeros((1, 2, 1, 48), DType::F32, &device).unwrap();
        assert!(!fused_sdpa_eligible(&q, &k, &v, None));
    }

    #[test]
    fn fused_sdpa_rejects_mask_with_wrong_head_axis() {
        let device = Device::Cpu;
        let q = Tensor::zeros((1, 16, 4, 128), DType::F32, &device).unwrap();
        let k = Tensor::zeros((1, 2, 4, 128), DType::F32, &device).unwrap();
        let v = Tensor::zeros((1, 2, 4, 128), DType::F32, &device).unwrap();
        let mask = Tensor::zeros((1, 1, 4, 4), DType::F32, &device).unwrap();
        assert!(!fused_sdpa_eligible(&q, &k, &v, Some(&mask)));
    }

    #[test]
    fn attention_forward_cpu_matches_eager() {
        let device = Device::Cpu;
        let q = Tensor::randn(0f32, 1.0, (1, 4, 2, 32), &device).unwrap();
        let k = Tensor::randn(0f32, 1.0, (1, 2, 2, 32), &device).unwrap();
        let v = Tensor::randn(0f32, 1.0, (1, 2, 2, 32), &device).unwrap();
        let scale = 1.0 / (32f64).sqrt();
        let a = attention_forward(&q, &k, &v, Some(2), None, scale).unwrap();
        let b = eager_attention_forward(&q, &k, &v, Some(2), None, scale).unwrap();
        let diff = (a - b)
            .unwrap()
            .abs()
            .unwrap()
            .max(D::Minus1)
            .unwrap()
            .max(D::Minus1)
            .unwrap()
            .max(D::Minus1)
            .unwrap()
            .max(0)
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(
            diff < 1e-5,
            "attention_forward must equal eager on CPU, max_diff={diff}"
        );
    }

    #[test]
    fn eager_gqa_matches_repeat_kv() {
        let device = Device::Cpu;
        let q = Tensor::randn(0f32, 1.0, (1, 16, 11, 128), &device).unwrap();
        let k = Tensor::randn(0f32, 1.0, (1, 2, 11, 128), &device).unwrap();
        let v = Tensor::randn(0f32, 1.0, (1, 2, 11, 128), &device).unwrap();
        let scale = 1.0 / (128f64).sqrt();
        let gqa = eager_attention_gqa(&q, &k, &v, 8, scale).unwrap();
        let tiled = eager_attention_inner(
            &q,
            &repeat_kv(&k, 8).unwrap(),
            &repeat_kv(&v, 8).unwrap(),
            None,
            scale,
        )
        .unwrap();
        let diff = (&gqa - &tiled)
            .unwrap()
            .abs()
            .unwrap()
            .mean_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(diff < 1e-5, "GQA reshape vs repeat_kv mean abs diff={diff}");
    }

    #[cfg(feature = "metal")]
    #[test]
    fn fused_sdpa_metal_close_to_eager() {
        let device = match std::panic::catch_unwind(|| Device::new_metal(0)) {
            Ok(Ok(d)) => d,
            _ => return,
        };
        let q = Tensor::randn(0f32, 1.0, (1, 16, 1, 128), &device)
            .unwrap()
            .to_dtype(DType::F16)
            .unwrap()
            .contiguous()
            .unwrap();
        let k = Tensor::randn(0f32, 1.0, (1, 2, 1, 128), &device)
            .unwrap()
            .to_dtype(DType::F16)
            .unwrap()
            .contiguous()
            .unwrap();
        let v = Tensor::randn(0f32, 1.0, (1, 2, 1, 128), &device)
            .unwrap()
            .to_dtype(DType::F16)
            .unwrap()
            .contiguous()
            .unwrap();
        if !fused_sdpa_eligible(&q, &k, &v, None) {
            return;
        }
        let scale = 1.0 / (128f64).sqrt();
        let fused = match candle_nn::ops::sdpa(&q, &k, &v, None, false, scale as f32, 1.0) {
            Ok(t) => t.transpose(1, 2).unwrap().contiguous().unwrap(),
            Err(_) => return,
        };
        let eager = eager_attention_forward(&q, &k, &v, Some(8), None, scale).unwrap();
        let diff = (fused.to_dtype(DType::F32).unwrap() - eager.to_dtype(DType::F32).unwrap())
            .unwrap()
            .abs()
            .unwrap()
            .mean_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(
            diff < 5e-2,
            "Metal fused vs eager mean abs diff too large: {diff}"
        );
    }

    /// Documented failure: Metal vector SDPA produces NaNs at q_seq>1 for VoxCPM2 GQA.
    #[cfg(feature = "metal")]
    #[test]
    fn fused_sdpa_metal_vector_qseq_gt1_is_unsafe() {
        let device = match std::panic::catch_unwind(|| Device::new_metal(0)) {
            Ok(Ok(d)) => d,
            _ => return,
        };
        for q_seq in [4usize, 8] {
            let q = Tensor::randn(0f32, 1.0, (1, 16, q_seq, 128), &device)
                .unwrap()
                .to_dtype(DType::F16)
                .unwrap()
                .contiguous()
                .unwrap();
            let k = Tensor::randn(0f32, 1.0, (1, 2, q_seq, 128), &device)
                .unwrap()
                .to_dtype(DType::F16)
                .unwrap()
                .contiguous()
                .unwrap();
            let v = Tensor::randn(0f32, 1.0, (1, 2, q_seq, 128), &device)
                .unwrap()
                .to_dtype(DType::F16)
                .unwrap()
                .contiguous()
                .unwrap();
            let scale = 1.0 / (128f64).sqrt();
            let Ok(fused) = candle_nn::ops::sdpa(&q, &k, &v, None, false, scale as f32, 1.0) else {
                continue;
            };
            let mean = fused
                .to_dtype(DType::F32)
                .unwrap()
                .abs()
                .unwrap()
                .mean_all()
                .unwrap()
                .to_scalar::<f32>()
                .unwrap();
            // If this ever becomes finite and small-diff vs eager, Phase 3 chunking can return.
            assert!(
                !mean.is_finite() || mean > 1.0,
                "unexpected: vector SDPA at q_seq={q_seq} looks healthy (mean={mean}); revisit chunking"
            );
        }
    }
}
