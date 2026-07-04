//! Unified linear layer with optional Candle weight-only quantization.

use anyhow::Result;
use candle_core::quantized::{GgmlDType, QMatMul, QTensor};
use candle_core::{Device, DType, Module, Tensor};
use candle_nn::{linear, linear_no_bias, Linear, VarBuilder};

use crate::quant::{quant_audit_enabled, quant_audit_log, QuantBuildCtx, VoxCPMWeightQuant};

/// Resolved GGUF dtype for a weight matrix, including K-quant → Q8_0 fallback.
pub struct ResolvedGgmlDtype {
    pub dtype: GgmlDType,
    pub fallback_to_q8: bool,
}

fn ggml_dtype_name(dtype: GgmlDType) -> &'static str {
    match dtype {
        GgmlDType::Q8_0 => "Q8_0",
        GgmlDType::Q4K => "Q4K",
        GgmlDType::Q5K => "Q5K",
        GgmlDType::Q6K => "Q6K",
        other => {
            let _ = other;
            "other"
        }
    }
}

fn estimate_dense_bytes(weight: &Tensor) -> u64 {
    let n = weight.elem_count();
    let bpe = match weight.dtype() {
        DType::F32 | DType::F64 => 4,
        DType::F16 | DType::BF16 => 2,
        _ => 4,
    };
    (n * bpe) as u64
}

fn estimate_quantized_bytes(weight: &Tensor, dtype: GgmlDType) -> u64 {
    let n = weight.elem_count();
    let bpe = match dtype {
        GgmlDType::Q8_0 => 1,
        GgmlDType::Q4K => 4,
        GgmlDType::Q5K => 5,
        GgmlDType::Q6K => 6,
        _ => 4,
    };
    (n * bpe) as u64
}

/// When true, materialize quantized weights back to dense `Linear` at load.
/// Default is false (use live `QMatMul` for real memory/compute savings).
/// Set `VOXCPM_QUANT_DEQUANT_LINEAR=1` only for debugging parity against FP `Linear`.
#[must_use]
pub fn quant_dequant_linear_enabled() -> bool {
    match std::env::var("VOXCPM_QUANT_DEQUANT_LINEAR") {
        Ok(v) => !v.is_empty() && v != "0",
        Err(_) => false,
    }
}

#[derive(Debug, Clone)]
pub enum LinearX {
    Linear(Linear),
    QLinear(QLinear),
}

impl Module for LinearX {
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        match self {
            Self::Linear(ln) => ln.forward(x),
            Self::QLinear(ln) => ln.forward(x),
        }
    }
}

#[derive(Debug, Clone)]
pub struct QLinear {
    inner: QMatMul,
    bias: Option<Tensor>,
}

impl QLinear {
    fn ggml_dtype(quant: VoxCPMWeightQuant) -> GgmlDType {
        match quant {
            VoxCPMWeightQuant::None => GgmlDType::Q8_0,
            VoxCPMWeightQuant::Q8_0 => GgmlDType::Q8_0,
            VoxCPMWeightQuant::Q4K => GgmlDType::Q4K,
            VoxCPMWeightQuant::Q5K => GgmlDType::Q5K,
            VoxCPMWeightQuant::Q6K => GgmlDType::Q6K,
        }
    }

    fn resolve_ggml_dtype(weight: &Tensor, ggml_dtype: GgmlDType) -> Option<ResolvedGgmlDtype> {
        let last_dim = weight.dim(candle_core::D::Minus1).ok()?;
        if last_dim % ggml_dtype.block_size() == 0 {
            Some(ResolvedGgmlDtype {
                dtype: ggml_dtype,
                fallback_to_q8: false,
            })
        } else if last_dim % GgmlDType::Q8_0.block_size() == 0 {
            if quant_audit_enabled() {
                eprintln!(
                    "voxcpm quant: weight {weight:?} incompatible with {ggml_dtype:?}, falling back to Q8_0"
                );
            }
            Some(ResolvedGgmlDtype {
                dtype: GgmlDType::Q8_0,
                fallback_to_q8: true,
            })
        } else {
            if quant_audit_enabled() {
                eprintln!(
                    "voxcpm quant: weight {weight:?} incompatible with any GGUF dtype, keeping unquantized"
                );
            }
            None
        }
    }

    fn compatible_ggml_dtype(weight: &Tensor, ggml_dtype: GgmlDType) -> Option<GgmlDType> {
        Self::resolve_ggml_dtype(weight, ggml_dtype).map(|r| r.dtype)
    }

    fn load_bias(linear: &Linear, device: &Device, dtype: DType) -> Result<Option<Tensor>> {
        match linear.bias() {
            None => Ok(None),
            Some(b) => {
                let b = b.to_device(&Device::Cpu)?
                    .to_dtype(dtype)?
                    .to_device(device)?;
                Ok(Some(b))
            }
        }
    }

    pub fn from_linear(linear: Linear, quant: VoxCPMWeightQuant, device: &Device) -> Result<Self> {
        let ggml_dtype = Self::ggml_dtype(quant);
        let actual = Self::compatible_ggml_dtype(&linear.weight(), ggml_dtype)
            .ok_or_else(|| anyhow::anyhow!("weight shape incompatible with quantization"))?;
        Self::from_linear_with_dtype(linear, actual, device)
    }

    pub fn from_linear_with_dtype(
        linear: Linear,
        ggml_dtype: GgmlDType,
        device: &Device,
    ) -> Result<Self> {
        let bias = Self::load_bias(&linear, device, DType::F32)?;
        let weight = linear.weight().clone();
        let qtensor = quantize_weight(&weight, ggml_dtype, device)?;
        Ok(Self {
            inner: QMatMul::from_qtensor(qtensor)?,
            bias,
        })
    }
}

impl Module for QLinear {
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let in_dtype = x.dtype();
        let x = x.contiguous()?;
        // QMatMul accumulates in F32; restore input activation dtype afterward.
        let matmul_in = if in_dtype == DType::F32 {
            x
        } else {
            x.to_dtype(DType::F32)?
        };
        let mut out = self.inner.forward(&matmul_in)?;
        if let Some(bias) = &self.bias {
            out = out.broadcast_add(bias)?;
        }
        if out.dtype() != in_dtype {
            out = out.to_dtype(in_dtype)?;
        }
        Ok(out)
    }
}

fn quantize_weight(weight: &Tensor, dtype: GgmlDType, device: &Device) -> Result<QTensor> {
    let weight = weight.to_dtype(DType::F32)?;
    if weight.device().same_device(device) {
        QTensor::quantize(&weight, dtype).map_err(Into::into)
    } else {
        let weight_cpu = weight.to_device(&Device::Cpu)?;
        QTensor::quantize_onto(&weight_cpu, dtype, device).map_err(Into::into)
    }
}

pub fn linear_x(
    in_dim: usize,
    out_dim: usize,
    vb: VarBuilder,
    ctx: &QuantBuildCtx,
    bias: bool,
) -> Result<LinearX> {
    let device = vb.device().clone();
    let ln = if bias {
        linear(in_dim, out_dim, vb)?
    } else {
        linear_no_bias(in_dim, out_dim, vb)?
    };
    linear_x_from_linear(ln, ctx, &device)
}

pub fn linear_x_from_linear(linear: Linear, ctx: &QuantBuildCtx, device: &Device) -> Result<LinearX> {
    if !ctx.should_quantize() {
        if quant_audit_enabled() {
            quant_audit_log(&ctx.module_path, "skipped");
        }
        ctx.record_skipped();
        return Ok(LinearX::Linear(linear));
    }
    let requested = QLinear::ggml_dtype(ctx.config.weight);
    let resolved = QLinear::resolve_ggml_dtype(linear.weight(), requested);
    let Some(resolved) = resolved else {
        if quant_audit_enabled() {
            quant_audit_log(&ctx.module_path, "fallback_fp");
        }
        ctx.record_fallback_fp(estimate_dense_bytes(linear.weight()));
        return Ok(LinearX::Linear(linear));
    };
    if quant_audit_enabled() {
        quant_audit_log(&ctx.module_path, "quantized");
    }
    if quant_dequant_linear_enabled() {
        let t0 = std::time::Instant::now();
        let qtensor = quantize_weight(linear.weight(), resolved.dtype, device)?;
        ctx.add_quant_load(t0.elapsed());
        let mut weight = qtensor.dequantize(device)?;
        if weight.dtype() != linear.weight().dtype() {
            weight = weight.to_dtype(linear.weight().dtype())?;
        }
        let bias = QLinear::load_bias(&linear, device, weight.dtype())?;
        if quant_audit_enabled() {
            quant_audit_log(&ctx.module_path, "dequant_linear");
        }
        return Ok(LinearX::Linear(Linear::new(weight, bias)));
    }
    let dense_bytes = estimate_dense_bytes(linear.weight());
    let quant_bytes = estimate_quantized_bytes(linear.weight(), resolved.dtype);
    let t0 = std::time::Instant::now();
    let qlinear = QLinear::from_linear_with_dtype(linear, resolved.dtype, device)?;
    ctx.add_quant_load(t0.elapsed());
    ctx.record_quantized(
        ggml_dtype_name(resolved.dtype),
        dense_bytes,
        quant_bytes,
        resolved.fallback_to_q8,
    );
    Ok(LinearX::QLinear(qlinear))
}

pub fn linear_x_from_parts(
    weight: Tensor,
    bias: Option<Tensor>,
    ctx: &QuantBuildCtx,
    device: &Device,
) -> Result<LinearX> {
    linear_x_from_linear(Linear::new(weight, bias), ctx, device)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quant::VoxCPMQuantConfig;

    fn max_abs_diff(a: &Tensor, b: &Tensor) -> Result<f32> {
        Ok((a.to_dtype(DType::F32)? - b.to_dtype(DType::F32)?)?
            .abs()?
            .max_all()?
            .to_scalar()?)
    }

    fn rmse(a: &Tensor, b: &Tensor) -> Result<f32> {
        let diff = (a.to_dtype(DType::F32)? - b.to_dtype(DType::F32)?)?;
        let sq = diff.sqr()?.mean_all()?.to_scalar::<f32>()?;
        Ok(sq.sqrt())
    }

    #[test]
    fn qlinear_parity_cpu_small() -> Result<()> {
        let device = Device::Cpu;
        let weight = Tensor::randn(0f32, 1f32, (32, 16), &device)?;
        let bias = Some(Tensor::randn(0f32, 1f32, (32,), &device)?);
        let x = Tensor::randn(0f32, 1f32, (2, 8, 16), &device)?;
        let ln = Linear::new(weight, bias.clone());
        let y_ref = ln.forward(&x)?;

        let ctx = QuantBuildCtx::root(VoxCPMQuantConfig::with_weight(VoxCPMWeightQuant::Q8_0))
            .pp("test");
        let q = linear_x_from_linear(ln, &ctx, &device)?;
        let y_q = q.forward(&x)?;

        assert!(
            max_abs_diff(&y_ref, &y_q)? < 0.02,
            "QLinear max_abs_diff too large"
        );
        Ok(())
    }

    #[test]
    fn qlinear_parity_cpu_lm_shapes() -> Result<()> {
        let device = Device::Cpu;
        // Typical hidden=1024, intermediate gate row
        for (out_dim, in_dim) in [(4096, 1024), (3072, 1024), (1024, 4096)] {
            let weight = Tensor::randn(0f32, 0.02f32, (out_dim, in_dim), &device)?;
            let bias = Some(Tensor::randn(0f32, 0.02f32, (out_dim,), &device)?);
            let x = Tensor::randn(0f32, 1f32, (1, 8, in_dim), &device)?;
            let ln = Linear::new(weight, bias);
            let y_ref = ln.forward(&x)?;
            let ctx = QuantBuildCtx::root(VoxCPMQuantConfig::with_weight(VoxCPMWeightQuant::Q8_0))
                .pp("test");
            let q = linear_x_from_linear(ln, &ctx, &device)?;
            let y_q = q.forward(&x)?;
            assert!(
                rmse(&y_ref, &y_q)? < 0.05,
                "RMSE too large for shape ({out_dim}, {in_dim})"
            );
        }
        Ok(())
    }

    #[test]
    fn dequant_linear_path_cpu() -> Result<()> {
        std::env::set_var("VOXCPM_QUANT_DEQUANT_LINEAR", "1");
        let device = Device::Cpu;
        let weight = Tensor::randn(0f32, 0.02f32, (64, 32), &device)?;
        let x = Tensor::randn(0f32, 1f32, (2, 4, 32), &device)?;
        let ln = Linear::new(weight, None);
        let y_ref = ln.forward(&x)?;
        let ctx = QuantBuildCtx::root(VoxCPMQuantConfig::with_weight(VoxCPMWeightQuant::Q8_0))
            .pp("test");
        let q = linear_x_from_linear(ln, &ctx, &device)?;
        let y_q = q.forward(&x)?;
        assert!(max_abs_diff(&y_ref, &y_q)? < 0.05);
        std::env::remove_var("VOXCPM_QUANT_DEQUANT_LINEAR");
        Ok(())
    }

    fn qlinear_parity_for_quant(quant: VoxCPMWeightQuant) -> Result<()> {
        let device = Device::Cpu;
        let weight = Tensor::randn(0f32, 0.02f32, (128, 256), &device)?;
        let bias = Some(Tensor::randn(0f32, 0.02f32, (128,), &device)?);
        let x = Tensor::randn(0f32, 1f32, (1, 4, 256), &device)?;
        let ln = Linear::new(weight, bias);
        let y_ref = ln.forward(&x)?;
        let ctx = QuantBuildCtx::root(VoxCPMQuantConfig::with_weight(quant)).pp("test");
        let q = linear_x_from_linear(ln, &ctx, &device)?;
        let y_q = q.forward(&x)?;
        assert!(
            rmse(&y_ref, &y_q)? < 0.08,
            "RMSE too large for quant {:?}",
            quant
        );
        Ok(())
    }

    #[test]
    fn qlinear_parity_cpu_k_quants() -> Result<()> {
        qlinear_parity_for_quant(VoxCPMWeightQuant::Q4K)?;
        qlinear_parity_for_quant(VoxCPMWeightQuant::Q5K)?;
        qlinear_parity_for_quant(VoxCPMWeightQuant::Q6K)?;
        Ok(())
    }
}
