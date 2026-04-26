use candle_core::{DType, Device, DeviceLocation};

/// Get device (auto-detect or use provided)
pub fn get_device(device: Option<&Device>) -> Device {
    match device {
        Some(d) => d.clone(),
        None => {
            #[cfg(feature = "cuda")]
            {
                Device::new_cuda(0).unwrap_or(Device::Cpu)
            }
            #[cfg(all(not(feature = "cuda"), feature = "metal"))]
            {
                Device::new_metal(0).unwrap_or(Device::Cpu)
            }
            #[cfg(all(not(feature = "cuda"), not(feature = "metal")))]
            {
                Device::Cpu
            }
        }
    }
}

fn dtype_from_str(dtype: &str) -> DType {
    match dtype {
        "float32" | "float" | "f32" => DType::F32,
        "float64" | "double" | "f64" => DType::F64,
        "float16" | "half" | "f16" => DType::F16,
        "bfloat16" | "bf16" => DType::BF16,
        "uint8" | "u8" => DType::U8,
        "int64" | "i64" => DType::I64,
        _ => DType::F32,
    }
}

fn lower_precision_dtype_for_device(device: &Device) -> Option<DType> {
    match device.location() {
        DeviceLocation::Cuda { .. } => Some(DType::F16),
        DeviceLocation::Metal { .. } | DeviceLocation::Cpu => None,
    }
}

fn vae_dtype_for_device(device: &Device) -> Option<DType> {
    match device.location() {
        // Metal BF16/F16 is supported, but VoxCPM's conv-heavy streaming path did not
        // improve in lower precision on current Apple GPUs due to kernel/cast overhead.
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

/// Get VAE compute dtype.
pub fn get_vae_compute_dtype(dtype: Option<DType>, weight_dtype: DType, device: &Device) -> DType {
    match dtype {
        Some(d) => d,
        None => match weight_dtype {
            DType::F32 => vae_dtype_for_device(device).unwrap_or(weight_dtype),
            _ => weight_dtype,
        },
    }
}
