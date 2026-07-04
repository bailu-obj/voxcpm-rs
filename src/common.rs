use anyhow::Result;
use candle_core::{Tensor, D};
use candle_nn::{Activation, Module, VarBuilder};

use crate::kv_cache::KvCache;
use crate::linear::{linear_x, LinearX};
use crate::position_embed::rope::apply_rotary_pos_emb;
use crate::quant::QuantBuildCtx;
use crate::utils::tensor::repeat_kv;

pub struct GateUpDownMLP {
    gate_proj: LinearX,
    up_proj: LinearX,
    down_proj: LinearX,
    act_fn: Activation,
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
        let gate_proj = linear_x(
            hidden_size,
            intermediate_size,
            vb.pp("gate_proj"),
            &qctx.pp("gate_proj"),
            bias,
        )?;
        let up_proj = linear_x(
            hidden_size,
            intermediate_size,
            vb.pp("up_proj"),
            &qctx.pp("up_proj"),
            bias,
        )?;
        let down_proj = linear_x(
            intermediate_size,
            hidden_size,
            vb.pp("down_proj"),
            &qctx.pp("down_proj"),
            bias,
        )?;

        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
            act_fn,
        })
    }
}

impl Module for GateUpDownMLP {
    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        let gate = xs.apply(&self.gate_proj)?;
        let up = xs.apply(&self.up_proj)?;
        let res = (gate.apply(&self.act_fn)? * up)?;
        res.apply(&self.down_proj)
    }
}

/// Naive multi-head attention with separate Q/K/V projections (quantized individually).
#[derive(Debug)]
pub struct NaiveAttention {
    q_proj: LinearX,
    k_proj: LinearX,
    v_proj: LinearX,
    o_proj: LinearX,
    num_heads: usize,
    num_kv_heads: usize,
    num_kv_groups: usize,
    head_dim: usize,
    middle_size: usize,
    q_out: usize,
    kv_out: usize,
    scale: f64,
    kv_cache: KvCache,
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

        let q_proj = linear_x(
            hidden_size,
            q_out,
            vb.pp("q_proj"),
            &qctx.pp("q_proj"),
            bias,
        )?;
        let k_proj = linear_x(
            hidden_size,
            kv_out,
            vb.pp("k_proj"),
            &qctx.pp("k_proj"),
            bias,
        )?;
        let v_proj = linear_x(
            hidden_size,
            kv_out,
            vb.pp("v_proj"),
            &qctx.pp("v_proj"),
            bias,
        )?;
        let o_proj = linear_x(
            q_out,
            hidden_size,
            vb.pp(o_proj_pp_name),
            &qctx.pp(o_proj_pp_name),
            bias,
        )?;

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_heads: num_attention_heads,
            num_kv_heads: num_key_value_heads,
            num_kv_groups,
            head_dim,
            middle_size: q_out,
            q_out,
            kv_out,
            scale,
            kv_cache: KvCache::default(),
        })
    }

    #[inline]
    fn qkv(&self, xs: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
        Ok((
            self.q_proj.forward(xs)?,
            self.k_proj.forward(xs)?,
            self.v_proj.forward(xs)?,
        ))
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

        let attn_output = eager_attention_forward(
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

        let attn_output = eager_attention_forward(
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
    let attn_output = match num_key_value_groups {
        Some(g) if g > 1 => eager_attention_inner(
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
