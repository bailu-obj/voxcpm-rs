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
    let x2 = x2.affine(-1.0, 0.0)?;
    let rotate_x = Tensor::cat(&[&x2, &x1], D::Minus1)?.contiguous()?;
    Ok(rotate_x)
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
    let mut cos = cos.clone();
    let mut sin = sin.clone();
    if cos.rank() == 2 {
        // (seq_len, head_dim) -> (1, 1, seq_len, head_dim)
        cos = cos.unsqueeze(0)?.unsqueeze(0)?;
        sin = sin.unsqueeze(0)?.unsqueeze(0)?;
    }
    if cos.rank() == 3 {
        // (bs, seq_len, head_dim) -> (bs, 1, seq_len, head_dim)
        cos = cos.unsqueeze(1)?;
        sin = sin.unsqueeze(1)?;
    }
    let orig_dtype = q.dtype();
    let q = if tof32 { &q.to_dtype(DType::F32)? } else { q };
    let k = if tof32 { &k.to_dtype(DType::F32)? } else { k };
    let cos = cos.to_dtype(q.dtype())?;
    let sin = sin.to_dtype(q.dtype())?;

    let q_embed = q
        .broadcast_mul(&cos)?
        .add(&rotate_half(q)?.broadcast_mul(&sin)?)?
        .to_dtype(orig_dtype)?;
    let k_embed = k
        .broadcast_mul(&cos)?
        .add(&rotate_half(k)?.broadcast_mul(&sin)?)?
        .to_dtype(orig_dtype)?;
    Ok((q_embed, k_embed))
}
