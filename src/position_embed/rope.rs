use anyhow::Result;
use candle_core::{DType, Tensor, D};

#[inline]
pub fn compute_default_rope_parameters(dim: usize, base: f32) -> Vec<f32> {
    let inv_dim = 1.0f32 / dim as f32;
    (0..dim)
        .step_by(2)
        .map(|i| base.powf(-(i as f32) * inv_dim))
        .collect()
}

pub fn rotate_half(x: &Tensor) -> Result<Tensor> {
    let half_dim = x.dim(D::Minus1)? / 2;
    let x1 = x.narrow(D::Minus1, 0, half_dim)?;
    let x2 = x.narrow(D::Minus1, half_dim, half_dim)?;
    Ok(Tensor::cat(&[&x2.neg()?, &x1], D::Minus1)?)
}

pub fn apply_rotary_pos_emb(
    q: &Tensor,
    k: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    tof32: bool,
) -> Result<(Tensor, Tensor)> {
    // sin/cos: to (bs, 1, seq_len, head_dim)
    // q/k: (bs, n_head, seq_len, head_dim)
    let (cos, sin) = match cos.rank() {
        2 => (
            cos.unsqueeze(0)?.unsqueeze(0)?,
            sin.unsqueeze(0)?.unsqueeze(0)?,
        ),
        3 => (cos.unsqueeze(1)?, sin.unsqueeze(1)?),
        _ => (cos.clone(), sin.clone()),
    };

    let orig_dtype = q.dtype();

    // Only dispatch dtype-conversion kernels when they are actually needed.
    // The unconditional `to_dtype` calls in the original fire Metal kernels even
    // when the tensor is already the right type — across ~30 layers that is
    // hundreds of no-op kernel dispatches per decode step.
    let (q_work, k_work) = if tof32 && orig_dtype != DType::F32 {
        (q.to_dtype(DType::F32)?, k.to_dtype(DType::F32)?)
    } else {
        (q.clone(), k.clone())
    };

    let cos = if cos.dtype() != q_work.dtype() {
        cos.to_dtype(q_work.dtype())?
    } else {
        cos
    };
    let sin = if sin.dtype() != q_work.dtype() {
        sin.to_dtype(q_work.dtype())?
    } else {
        sin
    };

    // Fuse q/k RoPE application into one tensor path:
    // fewer kernel launches than doing q and k separately.
    let q_heads = q_work.dim(1)?;
    let k_heads = k_work.dim(1)?;
    let qk = Tensor::cat(&[&q_work, &k_work], 1)?;
    let qk_embed = rope_single(&qk, &cos, &sin)?;
    let q_embed = qk_embed.narrow(1, 0, q_heads)?;
    let k_embed = qk_embed.narrow(1, q_heads, k_heads)?;

    // Only cast back if we actually elevated to F32 above.
    if q_embed.dtype() != orig_dtype {
        Ok((q_embed.to_dtype(orig_dtype)?, k_embed.to_dtype(orig_dtype)?))
    } else {
        Ok((q_embed, k_embed))
    }
}

#[inline(always)]
fn rope_single(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    Ok(x.broadcast_mul(cos)?
        .add(&rotate_half(x)?.broadcast_mul(sin)?)?)
}
