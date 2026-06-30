//! Batched KV cache to reduce per-token `Tensor::cat` churn during autoregressive decode.

use anyhow::Result;
use candle_core::Tensor;

const CONSOLIDATE_THRESHOLD: usize = 8;

/// Append-only K/V cache with periodic consolidation into a single tensor pair.
#[derive(Debug, Default)]
pub struct KvCache {
    segments: Vec<(Tensor, Tensor)>,
    consolidated: Option<(Tensor, Tensor)>,
}

impl KvCache {
    pub fn clear(&mut self) {
        self.segments.clear();
        self.consolidated = None;
    }

    pub fn append(&mut self, key_states: Tensor, value_states: Tensor) -> Result<()> {
        if self.consolidated.is_none() && self.segments.is_empty() {
            self.segments.push((key_states, value_states));
            return Ok(());
        }

        self.segments.push((key_states, value_states));
        if self.segments.len() >= CONSOLIDATE_THRESHOLD {
            self.consolidate()?;
        }
        Ok(())
    }

    fn consolidate(&mut self) -> Result<()> {
        let mut keys: Vec<Tensor> = Vec::new();
        let mut values: Vec<Tensor> = Vec::new();
        if let Some((k, v)) = self.consolidated.take() {
            keys.push(k);
            values.push(v);
        }
        for (k, v) in self.segments.drain(..) {
            keys.push(k);
            values.push(v);
        }
        let key_refs: Vec<&Tensor> = keys.iter().collect();
        let value_refs: Vec<&Tensor> = values.iter().collect();
        self.consolidated = Some((Tensor::cat(&key_refs, 2)?, Tensor::cat(&value_refs, 2)?));
        Ok(())
    }

    pub fn keys_values(&mut self) -> Result<(Tensor, Tensor)> {
        if !self.segments.is_empty() {
            self.consolidate()?;
        }
        if let Some((k, v)) = &self.consolidated {
            return Ok((k.clone(), v.clone()));
        }
        anyhow::bail!("KvCache::keys_values called on empty cache");
    }

    pub fn is_empty(&self) -> bool {
        self.segments.is_empty() && self.consolidated.is_none()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, DType};

    #[test]
    fn kv_cache_append_and_read() -> Result<()> {
        let device = Device::Cpu;
        let k0 = Tensor::randn(0f32, 1f32, (1, 2, 3, 4), &device)?.to_dtype(DType::F32)?;
        let v0 = Tensor::zeros((1, 2, 3, 4), DType::F32, &device)?;
        let mut cache = KvCache::default();
        cache.append(k0.clone(), v0.clone())?;
        let (k, v) = cache.keys_values()?;
        assert_eq!(k.dims(), k0.dims());
        assert_eq!(v.dims(), v0.dims());
        Ok(())
    }

    #[test]
    fn kv_cache_consolidates_many_segments() -> Result<()> {
        let device = Device::Cpu;
        let mut cache = KvCache::default();
        for _ in 0..12 {
            let k = Tensor::randn(0f32, 1f32, (1, 2, 1, 4), &device)?.to_dtype(DType::F32)?;
            let v = Tensor::zeros((1, 2, 1, 4), DType::F32, &device)?;
            cache.append(k, v)?;
        }
        let (k, _) = cache.keys_values()?;
        assert_eq!(k.dim(2)?, 12);
        Ok(())
    }
}
