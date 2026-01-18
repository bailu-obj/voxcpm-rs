use anyhow::Result;
use candle_core::{Tensor, D};
use candle_nn::{
    linear, linear_no_bias, rms_norm, Activation, Linear, Module, RmsNorm, VarBuilder,
};

use crate::position_embed::rope::apply_rotary_pos_emb;
use crate::utils::tensor::{prepare_causal_attention_mask, repeat_kv};

/// Gate-Up-Down MLP (SwiGLU style)
#[derive(Debug, Clone)]
pub struct GateUpDownMLP {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
    act_fn: Activation,
}

impl GateUpDownMLP {
    pub fn new(
        vb: VarBuilder,
        hidden_size: usize,
        intermediate_size: usize,
        act_fn: Activation,
        bias: bool,
    ) -> Result<Self> {
        let (gate_proj, up_proj, down_proj) = if bias {
            (
                linear(hidden_size, intermediate_size, vb.pp("gate_proj"))?,
                linear(hidden_size, intermediate_size, vb.pp("up_proj"))?,
                linear(intermediate_size, hidden_size, vb.pp("down_proj"))?,
            )
        } else {
            (
                linear_no_bias(hidden_size, intermediate_size, vb.pp("gate_proj"))?,
                linear_no_bias(hidden_size, intermediate_size, vb.pp("up_proj"))?,
                linear_no_bias(intermediate_size, hidden_size, vb.pp("down_proj"))?,
            )
        };
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
        let lhs = xs.apply(&self.gate_proj)?.apply(&self.act_fn)?;
        let rhs = xs.apply(&self.up_proj)?;
        (lhs * rhs)?.apply(&self.down_proj)
    }
}

/// Naive multi-head attention
#[derive(Debug, Clone)]
pub struct NaiveAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    num_heads: usize,
    num_kv_heads: usize,
    num_kv_groups: usize,
    head_dim: usize,
    middle_size: usize,
    scale: f64,
    kv_cache: Option<(Tensor, Tensor)>,
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
    ) -> Result<Self> {
        let num_kv_groups = num_attention_heads / num_key_value_heads;
        let head_dim = head_dim.unwrap_or(hidden_size / num_attention_heads);
        let scale = 1f64 / f64::sqrt(head_dim as f64);
        let o_proj_pp_name = o_proj_pp_name.unwrap_or("o_proj");

        let (q_proj, k_proj, v_proj, o_proj) = if bias {
            (
                linear(hidden_size, num_attention_heads * head_dim, vb.pp("q_proj"))?,
                linear(hidden_size, num_key_value_heads * head_dim, vb.pp("k_proj"))?,
                linear(hidden_size, num_key_value_heads * head_dim, vb.pp("v_proj"))?,
                linear(
                    num_attention_heads * head_dim,
                    hidden_size,
                    vb.pp(o_proj_pp_name),
                )?,
            )
        } else {
            (
                linear_no_bias(hidden_size, num_attention_heads * head_dim, vb.pp("q_proj"))?,
                linear_no_bias(hidden_size, num_key_value_heads * head_dim, vb.pp("k_proj"))?,
                linear_no_bias(hidden_size, num_key_value_heads * head_dim, vb.pp("v_proj"))?,
                linear_no_bias(
                    num_attention_heads * head_dim,
                    hidden_size,
                    vb.pp(o_proj_pp_name),
                )?,
            )
        };

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_heads: num_attention_heads,
            num_kv_heads: num_key_value_heads,
            num_kv_groups,
            head_dim,
            middle_size: num_attention_heads * head_dim,
            scale,
            kv_cache: None,
        })
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
        let query_states = self.q_proj.forward(xs)?;
        let key_states = self.k_proj.forward(xs)?;
        let value_states = self.v_proj.forward(xs)?;

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
        cos: &Tensor,
        sin: &Tensor,
        attention_mask: Option<&Tensor>,
        tof32: bool,
    ) -> Result<Tensor> {
        let (b_sz, q_len, _) = xs.dims3()?;
        let query_states = self.q_proj.forward(xs)?;
        let key_states = self.k_proj.forward(xs)?;
        let value_states = self.v_proj.forward(xs)?;

        let query_states = query_states
            .reshape((b_sz, q_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let key_states = key_states
            .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let value_states = value_states
            .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        let (query_states, key_states) =
            apply_rotary_pos_emb(&query_states, &key_states, cos, sin, tof32)?;

        let (key_states, value_states) = match self.kv_cache.take() {
            None => (key_states, value_states),
            Some((prev_k, prev_v)) => {
                let key_states = Tensor::cat(&[&prev_k, &key_states], 2)?;
                let value_states = Tensor::cat(&[&prev_v, &value_states], 2)?;
                (key_states, value_states)
            }
        };

        self.kv_cache = Some((key_states.clone(), value_states.clone()));
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
        self.kv_cache = None
    }
}

/// Eager attention forward (without flash attention optimization)
pub fn eager_attention_forward(
    query_states: &Tensor,
    key_states: &Tensor,
    value_states: &Tensor,
    num_key_value_groups: Option<usize>,
    attention_mask: Option<&Tensor>,
    scaling: f64,
) -> Result<Tensor> {
    // Apply KV grouping if needed and ensure contiguous layout
    let (key_states, value_states) = match num_key_value_groups {
        Some(g) => {
            let k = repeat_kv(key_states.clone(), g)?.contiguous()?;
            let v = repeat_kv(value_states.clone(), g)?.contiguous()?;
            (k, v)
        }
        None => (key_states.contiguous()?, value_states.contiguous()?),
    };

    let query_states = query_states.contiguous()?;

    let attn_output = {
        #[cfg(not(feature = "flash-attn"))]
        {
            let mut attn_weights = query_states
                .matmul(&key_states.transpose(D::Minus2, D::Minus1)?)?
                .affine(scaling, 0.0)?;

            if let Some(mask) = attention_mask {
                attn_weights = attn_weights.broadcast_add(&mask.to_dtype(attn_weights.dtype())?)?;
            }

            candle_nn::ops::softmax_last_dim(&attn_weights)?.matmul(&value_states)?
        }
        #[cfg(feature = "flash-attn")]
        {
            let query_states = query_states.transpose(1, 2)?;
            let key_states = key_states.transpose(1, 2)?;
            let value_states = value_states.transpose(1, 2)?;
            let attn_output = candle_flash_attn::flash_attn(
                &query_states,
                &key_states,
                &value_states,
                scaling as f32,
                attention_mask.is_some(),
            )?
            .transpose(1, 2)?;
            attn_output
        }
    };

    let attn_output = attn_output.transpose(1, 2)?.contiguous()?;
    Ok(attn_output)
}
