use candle_core::Device;

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
