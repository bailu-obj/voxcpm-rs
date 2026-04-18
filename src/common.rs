use anyhow::Result;
use candle_core::{Tensor, D};
use candle_nn::{linear, linear_no_bias, Activation, Linear, Module, VarBuilder};

use crate::position_embed::rope::apply_rotary_pos_emb;
use crate::utils::tensor::repeat_kv;

pub struct GateUpDownMLP {
    gate_up_proj: Linear, // hidden → 2 * intermediate
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
        let gate_proj = if bias {
            linear(hidden_size, intermediate_size, vb.pp("gate_proj"))?
        } else {
            linear_no_bias(hidden_size, intermediate_size, vb.pp("gate_proj"))?
        };

        let up_proj = if bias {
            linear(hidden_size, intermediate_size, vb.pp("up_proj"))?
        } else {
            linear_no_bias(hidden_size, intermediate_size, vb.pp("up_proj"))?
        };

        // 2️⃣ Fuse weights
        let gate_w = gate_proj.weight();
        let up_w = up_proj.weight();
        let fused_w = Tensor::cat(&[gate_w, up_w], 0)?; // [2I, H]

        let fused_b = if bias {
            Some(Tensor::cat(
                &[gate_proj.bias().unwrap(), up_proj.bias().unwrap()],
                0,
            )?)
        } else {
            None
        };

        let gate_up_proj = Linear::new(fused_w, fused_b);

        let down_proj = if bias {
            linear(intermediate_size, hidden_size, vb.pp("down_proj"))?
        } else {
            linear_no_bias(intermediate_size, hidden_size, vb.pp("down_proj"))?
        };

        Ok(Self {
            gate_up_proj,
            down_proj,
            act_fn,
        })
    }
}

impl Module for GateUpDownMLP {
    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        let gate_up = xs.apply(&self.gate_up_proj)?;
        let chunks = gate_up.chunk(2, D::Minus1)?;
        let (gate, up) = (&chunks[0], &chunks[1]);
        let res = (gate.apply(&self.act_fn)? * up)?;
        res.apply(&self.down_proj)
    }
}

/// Naive multi-head attention.
///
/// Q/K/V projections are **fused** into a single matmul at construction time:
/// per-call we read the activation once and split the result via `narrow`, instead
/// of dispatching three small matmuls on the same input. For a typical VoxCPM
/// utterance this removes ~4.7k Metal kernel launches and lets the GPU reuse the
/// activation in L2 rather than reloading it three times.
#[derive(Debug, Clone)]
pub struct NaiveAttention {
    qkv_proj: Linear,
    o_proj: Linear,
    num_heads: usize,
    num_kv_heads: usize,
    num_kv_groups: usize,
    head_dim: usize,
    middle_size: usize,
    q_out: usize,
    kv_out: usize,
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
        let q_out = num_attention_heads * head_dim;
        let kv_out = num_key_value_heads * head_dim;

        // Load the three separate projections, then fuse them along the output dim.
        // Concatenation happens once at init; all forwards use the fused weight.
        let (q_proj, k_proj, v_proj, o_proj) = if bias {
            (
                linear(hidden_size, q_out, vb.pp("q_proj"))?,
                linear(hidden_size, kv_out, vb.pp("k_proj"))?,
                linear(hidden_size, kv_out, vb.pp("v_proj"))?,
                linear(q_out, hidden_size, vb.pp(o_proj_pp_name))?,
            )
        } else {
            (
                linear_no_bias(hidden_size, q_out, vb.pp("q_proj"))?,
                linear_no_bias(hidden_size, kv_out, vb.pp("k_proj"))?,
                linear_no_bias(hidden_size, kv_out, vb.pp("v_proj"))?,
                linear_no_bias(q_out, hidden_size, vb.pp(o_proj_pp_name))?,
            )
        };

        let fused_w = Tensor::cat(
            &[q_proj.weight(), k_proj.weight(), v_proj.weight()],
            0,
        )?
        .contiguous()?;
        let fused_b = if bias {
            Some(
                Tensor::cat(
                    &[
                        q_proj
                            .bias()
                            .expect("NaiveAttention bias=true but q_proj has no bias"),
                        k_proj
                            .bias()
                            .expect("NaiveAttention bias=true but k_proj has no bias"),
                        v_proj
                            .bias()
                            .expect("NaiveAttention bias=true but v_proj has no bias"),
                    ],
                    0,
                )?
                .contiguous()?,
            )
        } else {
            None
        };
        let qkv_proj = Linear::new(fused_w, fused_b);

        Ok(Self {
            qkv_proj,
            o_proj,
            num_heads: num_attention_heads,
            num_kv_heads: num_key_value_heads,
            num_kv_groups,
            head_dim,
            middle_size: q_out,
            q_out,
            kv_out,
            scale,
            kv_cache: None,
        })
    }

    /// Run the fused QKV projection and return `(q, k, v)` split views.
    #[inline]
    fn qkv(&self, xs: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
        let qkv = self.qkv_proj.forward(xs)?;
        let q = qkv.narrow(D::Minus1, 0, self.q_out)?;
        let k = qkv.narrow(D::Minus1, self.q_out, self.kv_out)?;
        let v = qkv.narrow(D::Minus1, self.q_out + self.kv_out, self.kv_out)?;
        Ok((q, k, v))
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
        cos: &Tensor,
        sin: &Tensor,
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

        let (query_states, key_states) =
            apply_rotary_pos_emb(&query_states, &key_states, cos, sin, tof32)?;

        if self.kv_cache.is_none() {
            self.kv_cache = Some((key_states, value_states));
        } else if let Some((k, v)) = &mut self.kv_cache {
            *k = Tensor::cat(&[&*k, &key_states], 2)?;
            *v = Tensor::cat(&[&*v, &value_states], 2)?;
        }
        let (key_states, value_states) = self.kv_cache.as_ref().unwrap();

        let attn_output = eager_attention_forward(
            &query_states,
            key_states,
            value_states,
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

/// Eager attention forward (without flash attention optimization).
///
/// When `num_key_value_groups` is `None` or `Some(1)`, K/V tensors are used in-place (no `repeat_kv`
/// full-tensor clone). GMA-style MQA only uses `repeat_kv` when `g > 1`.
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
        // Contiguous Q improves matmul on Metal/CUDA; K/V stay strided when possible.
        let query_states = query_states.contiguous()?;
        let key_transposed = key_states.transpose(D::Minus2, D::Minus1)?;
        let mut attn_weights = query_states
            .matmul(&key_transposed)?
            .affine(scaling, 0.0)?;

        if let Some(mask) = attention_mask {
            attn_weights =
                attn_weights.broadcast_add(&mask.to_dtype(attn_weights.dtype())?)?;
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
