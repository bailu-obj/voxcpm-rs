use candle_core::{DType, Device, DeviceLocation};

/// Resolve device from explicit handle, optional ordinal, or auto-detect.
pub fn get_device(device: Option<&Device>, device_id: Option<usize>) -> Device {
    if let Some(d) = device {
        return d.clone();
    }
    if env_force_cpu() {
        return Device::Cpu;
    }
    let id = device_id.unwrap_or(0);
    #[cfg(feature = "cuda")]
    {
        if let Ok(d) = Device::new_cuda(id) {
            return d;
        }
    }
    #[cfg(all(not(feature = "cuda"), feature = "metal"))]
    {
        if let Some(d) = try_new_metal(id) {
            return d;
        }
    }
    Device::Cpu
}

fn env_force_cpu() -> bool {
    std::env::var("VOXCPM_DEVICE")
        .map(|v| v.eq_ignore_ascii_case("cpu"))
        .unwrap_or(false)
}

#[cfg(all(not(feature = "cuda"), feature = "metal"))]
fn try_new_metal(id: usize) -> Option<Device> {
    // Candle may panic when Metal is unavailable (headless CI/sandbox); fall back to CPU.
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| Device::new_metal(id))) {
        Ok(Ok(device)) => Some(device),
        Ok(Err(_)) | Err(_) => None,
    }
}

/// Backward-compatible wrapper (device 0).
pub fn get_device_auto(device: Option<&Device>) -> Device {
    get_device(device, None)
}

fn dtype_from_str(dtype: &str) -> DType {
    match dtype.trim().to_lowercase().as_str() {
        "auto" => DType::F32,
        "float32" | "float" | "f32" => DType::F32,
        "float64" | "double" | "f64" => DType::F64,
        "float16" | "half" | "f16" => DType::F16,
        "bfloat16" | "bf16" => DType::BF16,
        "uint8" | "u8" => DType::U8,
        "int64" | "i64" => DType::I64,
        _ => DType::F32,
    }
}

/// Parse dtype string (`auto`, `f16`, `bf16`, `f32`) into an optional override.
pub fn parse_dtype_option(s: Option<&str>) -> Option<DType> {
    let s = s?;
    if s.eq_ignore_ascii_case("auto") {
        None
    } else {
        Some(dtype_from_str(s))
    }
}

fn lower_precision_dtype_for_device(device: &Device) -> Option<DType> {
    match device.location() {
        DeviceLocation::Cuda { .. } | DeviceLocation::Metal { .. } => Some(DType::F16),
        DeviceLocation::Cpu => None,
    }
}

fn vae_dtype_for_device(device: &Device, explicit: Option<DType>) -> Option<DType> {
    if explicit.is_some() {
        return explicit;
    }
    match device.location() {
        DeviceLocation::Cuda { .. } if device.supports_bf16() => Some(DType::BF16),
        DeviceLocation::Metal { .. } | DeviceLocation::Cpu | DeviceLocation::Cuda { .. } => None,
    }
}

/// Get dtype from config string.
pub fn get_dtype(dtype: Option<DType>, cfg_dtype: &str) -> DType {
    dtype.unwrap_or_else(|| dtype_from_str(cfg_dtype))
}

/// Get compute dtype, preferring lower precision on GPU when the model config is F32.
pub fn get_compute_dtype(dtype: Option<DType>, cfg_dtype: &str, device: &Device) -> DType {
    match dtype {
        Some(d) => d,
        None => {
            let cfg_dtype = dtype_from_str(cfg_dtype);
            match cfg_dtype {
                DType::F32 => lower_precision_dtype_for_device(device).unwrap_or(cfg_dtype),
                _ => cfg_dtype,
            }
        }
    }
}

/// Compute dtype when weight quantization is enabled.
///
/// QMatMul accumulates in F32; default `auto` uses F32 activations for quant runs to
/// avoid per-layer F16/BF16 ↔ F32 cast churn. Explicit `dtype=f16`/`bf16` still honored.
pub fn get_quant_compute_dtype(dtype: Option<DType>, cfg_dtype: &str, device: &Device) -> DType {
    match dtype {
        Some(d) => d,
        None => {
            let cfg = dtype_from_str(cfg_dtype);
            if cfg == DType::F32 {
                DType::F32
            } else {
                get_compute_dtype(None, cfg_dtype, device)
            }
        }
    }
}

/// Get VAE compute dtype.
pub fn get_vae_compute_dtype(dtype: Option<DType>, weight_dtype: DType, device: &Device) -> DType {
    match dtype {
        Some(d) => d,
        None => match weight_dtype {
            DType::F32 => vae_dtype_for_device(device, None).unwrap_or(weight_dtype),
            _ => weight_dtype,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_auto_dtype_is_none() {
        assert_eq!(parse_dtype_option(Some("auto")), None);
        assert_eq!(parse_dtype_option(Some("f16")), Some(DType::F16));
    }

    #[test]
    fn quant_auto_dtype_is_f32() {
        let device = Device::Cpu;
        assert_eq!(
            get_quant_compute_dtype(None, "float32", &device),
            DType::F32
        );
    }
}
