//! Inference-only Metal fusion for transformer and VAE activations.

use candle_core::{
    CpuStorage, CustomOp1, CustomOp3, DType, Layout, MetalStorage, Shape, backend::BackendStorage,
};
use candle_metal_kernels::metal::{ComputePipeline, Device};
use objc2_metal::{MTLResourceUsage, MTLSize};
use std::{
    collections::HashMap,
    sync::{Mutex, OnceLock},
};

const SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;
kernel void voxcpm_snake(
    device const float *x [[buffer(0)]],
    device const float *alpha [[buffer(1)]],
    device const float *recip [[buffer(2)]],
    device float *out [[buffer(3)]],
    constant size_t &count [[buffer(4)]],
    constant size_t &channels [[buffer(5)]],
    constant size_t &length [[buffer(6)]],
    uint i [[thread_position_in_grid]]) {
    if (i >= count) return;
    size_t c = (i / length) % channels;
    float s = sin(alpha[c] * x[i]);
    float square = s * s;
    float periodic = square * recip[c];
    out[i] = x[i] + periodic;
}
kernel void voxcpm_silu_mul(
    device const float *input [[buffer(0)]],
    device float *out [[buffer(1)]],
    constant size_t &count [[buffer(2)]],
    constant size_t &width [[buffer(3)]],
    uint i [[thread_position_in_grid]]) {
    if (i >= count) return;
    size_t offset = (i / width) * (2 * width) + i % width;
    float gate = input[offset];
    float up = input[offset + width];
    float activation = gate / (1.0f + exp(-gate));
    out[i] = activation * up;
}
"#;

// One pipeline per kernel and physical Metal device, shared by every layer and model load.
fn pipeline(device: &Device, name: &'static str) -> candle_core::Result<ComputePipeline> {
    static PIPELINES: OnceLock<Mutex<HashMap<(u64, &'static str), ComputePipeline>>> =
        OnceLock::new();
    let mut cache = PIPELINES
        .get_or_init(Mutex::default)
        .lock()
        .map_err(|e| candle_core::Error::Msg(format!("VoxCPM pipeline cache: {e}")))?;
    let key = (device.registry_id(), name);
    if let Some(pipeline) = cache.get(&key) {
        return Ok(pipeline.clone());
    }
    let options = objc2_metal::MTLCompileOptions::new();
    #[allow(deprecated)]
    options.setFastMathEnabled(false);
    let library = device
        .new_library_with_source(SOURCE, Some(&options))
        .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
    let function = library
        .get_function(name, None)
        .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
    let pipeline = device
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
    cache.insert(key, pipeline.clone());
    Ok(pipeline)
}

pub(crate) fn snake_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("VOXCPM_FUSED_SNAKE").map_or(true, |v| {
            !matches!(
                v.trim().to_ascii_lowercase().as_str(),
                "0" | "false" | "off"
            )
        })
    })
}

pub(crate) struct Snake;

impl CustomOp3 for Snake {
    fn name(&self) -> &'static str {
        "voxcpm_snake"
    }

    fn cpu_fwd(
        &self,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
    ) -> candle_core::Result<(CpuStorage, Shape)> {
        candle_core::bail!("fused Snake is Metal-only")
    }

    fn metal_fwd(
        &self,
        x: &MetalStorage,
        xl: &Layout,
        alpha: &MetalStorage,
        al: &Layout,
        recip: &MetalStorage,
        rl: &Layout,
    ) -> candle_core::Result<(MetalStorage, Shape)> {
        let (_, channels, length) = xl.shape().dims3()?;
        if !xl.is_contiguous()
            || !al.is_contiguous()
            || !rl.is_contiguous()
            || al.dims() != [1, channels, 1]
            || rl.dims() != [1, channels, 1]
            || x.dtype() != DType::F32
            || alpha.dtype() != DType::F32
            || recip.dtype() != DType::F32
        {
            candle_core::bail!("fused Snake requires contiguous F32 [B,C,T] and [1,C,1] parameters")
        }
        let count = xl.shape().elem_count();
        let device = x.device();
        let out = device.new_buffer(count, DType::F32, "voxcpm_snake")?;
        if count > 0 {
            let pipeline = pipeline(device.device(), "voxcpm_snake")?;
            let encoder = device.command_encoder()?;
            encoder.set_compute_pipeline_state(&pipeline);
            encoder.set_buffer(0, Some(x.buffer()), xl.start_offset() * 4);
            encoder.set_buffer(1, Some(alpha.buffer()), al.start_offset() * 4);
            encoder.set_buffer(2, Some(recip.buffer()), rl.start_offset() * 4);
            encoder.set_buffer(3, Some(&out), 0);
            encoder.set_bytes(4, &count);
            encoder.set_bytes(5, &channels);
            encoder.set_bytes(6, &length);
            encoder.use_resource(x.buffer(), MTLResourceUsage::Read);
            encoder.use_resource(alpha.buffer(), MTLResourceUsage::Read);
            encoder.use_resource(recip.buffer(), MTLResourceUsage::Read);
            encoder.use_resource(&*out, MTLResourceUsage::Write);
            let width = pipeline.max_total_threads_per_threadgroup().min(256);
            encoder.dispatch_thread_groups(
                MTLSize {
                    width: count.div_ceil(width),
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width,
                    height: 1,
                    depth: 1,
                },
            );
        }
        Ok((
            MetalStorage::new(out, device.clone(), count, DType::F32),
            xl.shape().clone(),
        ))
    }
}

pub(crate) fn silu_mul_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("VOXCPM_FUSED_SILU_MUL").map_or(true, |v| {
            !matches!(
                v.trim().to_ascii_lowercase().as_str(),
                "0" | "false" | "off"
            )
        })
    })
}

pub(crate) struct SiluMul;
impl CustomOp1 for SiluMul {
    fn name(&self) -> &'static str {
        "voxcpm_silu_mul"
    }
    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout) -> candle_core::Result<(CpuStorage, Shape)> {
        candle_core::bail!("fused SiLU multiply is Metal-only")
    }
    fn metal_fwd(
        &self,
        input: &MetalStorage,
        layout: &Layout,
    ) -> candle_core::Result<(MetalStorage, Shape)> {
        let mut dims = layout.dims().to_vec();
        let full_width = *dims.last().ok_or_else(|| {
            candle_core::Error::Msg("SiLU multiply requires non-scalar input".into())
        })?;
        if !layout.is_contiguous()
            || input.dtype() != DType::F32
            || full_width == 0
            || full_width % 2 != 0
        {
            candle_core::bail!("fused SiLU multiply requires contiguous F32 gate/up pairs")
        }
        let width = full_width / 2;
        *dims.last_mut().unwrap() = width;
        let shape = Shape::from(dims);
        let count = shape.elem_count();
        let device = input.device();
        let out = device.new_buffer(count, DType::F32, "voxcpm_silu_mul")?;
        if count > 0 {
            let pipeline = pipeline(device.device(), "voxcpm_silu_mul")?;
            let encoder = device.command_encoder()?;
            encoder.set_compute_pipeline_state(&pipeline);
            encoder.set_buffer(0, Some(input.buffer()), layout.start_offset() * 4);
            encoder.set_buffer(1, Some(&out), 0);
            encoder.set_bytes(2, &count);
            encoder.set_bytes(3, &width);
            encoder.use_resource(input.buffer(), MTLResourceUsage::Read);
            encoder.use_resource(&*out, MTLResourceUsage::Write);
            let threads = pipeline.max_total_threads_per_threadgroup().min(256);
            encoder.dispatch_thread_groups(
                MTLSize {
                    width: count.div_ceil(threads),
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: threads,
                    height: 1,
                    depth: 1,
                },
            );
        }
        Ok((
            MetalStorage::new(out, device.clone(), count, DType::F32),
            shape,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Tensor};

    #[test]
    #[ignore = "requires a Metal GPU"]
    fn fused_silu_mul_matches_strided_gate_up() -> candle_core::Result<()> {
        let device = Device::new_metal(0)?;
        for (batch, length, width) in [(1, 1, 6144), (2, 11, 4096), (2, 4, 32)] {
            let input = Tensor::randn(0f32, 3f32, (batch + 1, length, 2 * width), &device)?
                .narrow(0, 1, batch)?;
            let gate = input.narrow(2, 0, width)?;
            let up = input.narrow(2, width, width)?;
            let eager = gate.silu()?.mul(&up)?;
            let fused = input.apply_op1_no_bwd(&SiluMul)?;
            let diff = (fused - eager)?.abs()?.max_all()?.to_scalar::<f32>()?;
            assert!(diff < 1e-5, "fused SiLU multiply max error={diff}");
        }
        Ok(())
    }

    #[test]
    #[ignore = "requires a Metal GPU"]
    fn fused_snake_matches_eager_with_batches_and_offsets() -> candle_core::Result<()> {
        let device = Device::new_metal(0)?;
        for (batch, channels, length) in [(1, 2048, 48), (2, 128, 129), (1, 16, 48000)] {
            let x = Tensor::randn(0f32, 1f32, (batch + 1, channels, length), &device)?
                .narrow(0, 1, batch)?;
            let alpha = Tensor::rand(0.01f32, 5f32, (2, channels, 1), &device)?.narrow(0, 1, 1)?;
            let recip = alpha.affine(1.0, 1e-9)?.recip()?;
            let eager = x
                .broadcast_mul(&alpha)?
                .sin()?
                .sqr()?
                .broadcast_mul(&recip)?
                .add(&x)?;
            let fused = x.apply_op3_no_bwd(&alpha, &recip, &Snake)?;
            let diff = (fused - eager)?.abs()?.max_all()?.to_scalar::<f32>()?;
            assert!(diff < 2e-6, "fused Snake max error={diff}");
        }
        Ok(())
    }
}
