use anyhow::Result;
use candle_core::{D, DType, Tensor};

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

/// Whether fused candle RoPE is disabled (`VOXCPM_FUSED_ROPE=0`).
#[must_use]
pub fn fused_rope_env_disabled() -> bool {
    match std::env::var("VOXCPM_FUSED_ROPE") {
        Ok(v) => {
            let v = v.trim().to_ascii_lowercase();
            v == "0" || v == "false" || v == "off" || v == "eager"
        }
        Err(_) => false,
    }
}

pub fn apply_rotary_pos_emb(
    q: &Tensor,
    k: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    tof32: bool,
) -> Result<(Tensor, Tensor)> {
    // sin/cos:
    //   - full-width (head_dim): used by eager rotate-half path
    //   - half-width (head_dim/2): used by candle_nn::rotary_emb::rope
    // q/k: (bs, n_head, seq_len, head_dim)
    let orig_dtype = q.dtype();
    let head_dim = q.dim(D::Minus1)?;
    let cos_last = cos.dim(D::Minus1)?;
    let half_width = cos_last * 2 == head_dim;

    let (q_work, k_work) = if tof32 && orig_dtype != DType::F32 {
        (q.to_dtype(DType::F32)?, k.to_dtype(DType::F32)?)
    } else {
        (q.clone(), k.clone())
    };

    let cos = if cos.dtype() != q_work.dtype() {
        cos.to_dtype(q_work.dtype())?
    } else {
        cos.clone()
    };
    let sin = if sin.dtype() != q_work.dtype() {
        sin.to_dtype(q_work.dtype())?
    } else {
        sin.clone()
    };

    let (q_embed, k_embed) = if !fused_rope_env_disabled() && half_width {
        // candle fused Metal/CPU kernel: cos/sin are (seq, head_dim/2).
        // Run the fused op in F32 on Metal F16 activations — the half-precision
        // Metal rope kernel drifts the CFM trajectory enough to miss corr≥0.98.
        let rope_dtype = if matches!(q_work.dtype(), DType::F16 | DType::BF16) {
            DType::F32
        } else {
            q_work.dtype()
        };
        let cos = match cos.rank() {
            2 => cos,
            3 => cos.squeeze(0)?,
            _ => cos.squeeze(0)?.squeeze(0)?,
        };
        let sin = match sin.rank() {
            2 => sin,
            3 => sin.squeeze(0)?,
            _ => sin.squeeze(0)?.squeeze(0)?,
        };
        let cos = if cos.dtype() != rope_dtype {
            cos.to_dtype(rope_dtype)?
        } else {
            cos
        }
        .contiguous()?;
        let sin = if sin.dtype() != rope_dtype {
            sin.to_dtype(rope_dtype)?
        } else {
            sin
        }
        .contiguous()?;
        let q_heads = q_work.dim(1)?;
        let k_heads = k_work.dim(1)?;
        let qk = Tensor::cat(&[&q_work, &k_work], 1)?;
        let qk = if qk.dtype() != rope_dtype {
            qk.to_dtype(rope_dtype)?
        } else {
            qk
        }
        .contiguous()?;
        let qk_embed = candle_nn::rotary_emb::rope(&qk, &cos, &sin)?;
        let q_embed = qk_embed.narrow(1, 0, q_heads)?;
        let k_embed = qk_embed.narrow(1, q_heads, k_heads)?;
        (q_embed, k_embed)
    } else {
        // Eager rotate-half path; expand half-width caches if needed.
        let (cos, sin) = if half_width {
            (
                Tensor::cat(&[&cos, &cos], D::Minus1)?,
                Tensor::cat(&[&sin, &sin], D::Minus1)?,
            )
        } else {
            (cos, sin)
        };
        let (cos, sin) = match cos.rank() {
            2 => (
                cos.unsqueeze(0)?.unsqueeze(0)?,
                sin.unsqueeze(0)?.unsqueeze(0)?,
            ),
            3 => (cos.unsqueeze(1)?, sin.unsqueeze(1)?),
            _ => (cos, sin),
        };
        let q_heads = q_work.dim(1)?;
        let k_heads = k_work.dim(1)?;
        let qk = Tensor::cat(&[&q_work, &k_work], 1)?;
        let qk_embed = rope_single(&qk, &cos, &sin)?;
        let q_embed = qk_embed.narrow(1, 0, q_heads)?;
        let k_embed = qk_embed.narrow(1, q_heads, k_heads)?;
        (q_embed, k_embed)
    };

    if q_embed.dtype() != orig_dtype {
        Ok((q_embed.to_dtype(orig_dtype)?, k_embed.to_dtype(orig_dtype)?))
    } else {
        Ok((q_embed, k_embed))
    }
}

#[inline(always)]
pub fn rope_single(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    Ok(x.broadcast_mul(cos)?
        .add(&rotate_half(x)?.broadcast_mul(sin)?)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn candle_rope_matches_rope_single() -> Result<()> {
        let device = Device::Cpu;
        let b = 1usize;
        let h = 4usize;
        let t = 8usize;
        let d = 64usize;
        let x = Tensor::randn(0f32, 1f32, (b, h, t, d), &device)?.contiguous()?;
        // Half-width freqs duplicated to full for eager; candle wants half.
        let freqs = Tensor::randn(0f32, 0.5f32, (t, d / 2), &device)?;
        let cos_half = freqs.cos()?.contiguous()?;
        let sin_half = freqs.sin()?.contiguous()?;
        let cos_full = Tensor::cat(&[&cos_half, &cos_half], D::Minus1)?
            .unsqueeze(0)?
            .unsqueeze(0)?;
        let sin_full = Tensor::cat(&[&sin_half, &sin_half], D::Minus1)?
            .unsqueeze(0)?
            .unsqueeze(0)?;

        let eager = rope_single(&x, &cos_full, &sin_full)?;
        let fused = candle_nn::rotary_emb::rope(&x, &cos_half, &sin_half)?;
        let diff = (&eager - &fused)?.abs()?.max_all()?.to_scalar::<f32>()?;
        assert!(diff < 1e-5, "candle rope vs rope_single max_diff={diff}");
        Ok(())
    }
}
