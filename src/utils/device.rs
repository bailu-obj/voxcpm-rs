use candle_core::{DType, Device};

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

/// Get dtype from config string
pub fn get_dtype(dtype: Option<DType>, cfg_dtype: &str) -> DType {
    match dtype {
        Some(d) => d,
        None => match cfg_dtype {
            "float32" | "float" => DType::F32,
            "float64" | "double" => DType::F64,
            "float16" => DType::F16,
            "bfloat16" => DType::F16, // Candle does not support BFloat16 yet
            "uint8" => DType::U8,
            "int64" => DType::I64,
            _ => DType::F32,
        },
    }
}
