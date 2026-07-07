# voxcpm-rs

Pure Rust implementation of [VoxCPM](https://huggingface.co/openbmb) text-to-speech, built on [Candle](https://github.com/huggingface/candle). Supports zero-shot TTS, voice cloning from a reference clip, streaming audio output, and optional weight-only quantization.


## Features

- **VoxCPM family** — auto-detects VoxCPM, VoxCPM1.5, and VoxCPM2 from `config.json`
- **Voice cloning** — reference WAV + transcript prompt cache for timbre/style transfer
- **Streaming** — latent-to-audio chunks as `Iterator<Item = Result<Tensor>>`, plus PCM/WAV stream helpers
- **In-memory output** — `to_wav`, `to_pcm`, and streaming byte iterators without touching disk
- **GPU backends** — optional Metal (macOS) or CUDA via Candle feature flags
- **Weight quantization** — live `QMatMul` paths for `q8_0` with quality gates
- **Benchmarking** — RTF, TTFA, FP-vs-quant correlation checks

## Requirements

- Rust 1.85+ (Edition 2021)
- A downloaded VoxCPM checkpoint directory (see [Model layout](#model-layout))
- For GPU inference, build with `--features metal` (macOS) or `--features cuda` (Linux)

Initialize the submodule when cloning Bailu:

```bash
git submodule update --init vendor/voxcpm-rs
```

## Quick start

From the crate directory (or workspace root with `-p voxcpm-rs`):

```bash
# macOS
cargo build --release -p voxcpm-rs --features metal --example vox_cpm_tts_cli

# Linux + NVIDIA
cargo build --release -p voxcpm-rs --features cuda --example vox_cpm_tts_cli
```

Simple TTS:

```bash
cargo run --release -p voxcpm-rs --features metal --example vox_cpm_tts_cli -- \
  --model models/VoxCPM-0.5B \
  --text "你好，世界。" \
  --out output
```

Voice clone:

```bash
cargo run --release -p voxcpm-rs --features metal --example vox_cpm_tts_cli -- \
  --model models/VoxCPM2 \
  --ref-wav models/reference.wav \
  --ref-text "Reference transcript matching the clip." \
  --text "Target sentence to synthesize." \
  --out clone_out
```

Streaming (no reference clip):

```bash
cargo run --release -p voxcpm-rs --features metal --example vox_cpm_tts_cli -- \
  --model models/VoxCPM-0.5B \
  --text "Streaming synthesis test." \
  --stream \
  --out stream_out
```

Quantized inference with FP quality check:

```bash
cargo run --release -p voxcpm-rs --features metal --example vox_cpm_tts_cli -- \
  --model models/VoxCPM-0.5B \
  --text "测试" \
  --quant q8_0 \
  --compare-fp
```

RTF / TTFA benchmark:

```bash
cargo run --release -p voxcpm-rs --features metal --example voxcpm2_benchmark -- \
  --model models/VoxCPM2 \
  --text "Benchmark sentence." \
  --stream
```

## Library usage

```rust
use voxcpm_rs::{
    VoxCPMGenerationConfig, VoxCPMGenerator, VoxCPMGeneratorOptions,
    VoxCPMQuantConfig, VoxCPMWeightQuant,
};

fn main() -> anyhow::Result<()> {
    let mut options = VoxCPMGeneratorOptions::default();
    options.quant = VoxCPMQuantConfig::with_weight(VoxCPMWeightQuant::Q8_0);

    let mut gen = VoxCPMGenerator::new_with_options("models/VoxCPM2", &options)?;

    // Optional voice-clone prompt cache
    gen.build_prompt_cache(
        "Reference transcript.".into(),
        "models/reference.wav".into(),
    )?;

    let config = VoxCPMGenerationConfig::voice_clone();
    let audio = gen.generate_with_config("Hello from Rust.".into(), config)?;

    let wav_bytes = gen.to_wav(&audio.to_device(&candle_core::Device::Cpu)?)?;
    std::fs::write("out.wav", wav_bytes)?;
    Ok(())
}
```

### Streaming PCM

```rust
let config = VoxCPMGenerationConfig::simple();
for chunk in gen.generate_pcm_stream_with_config("Hello.".into(), config)? {
    let pcm: Vec<i16> = chunk?;
    // feed pcm to your audio sink
}
```

## Configuration

### Load-time options (`VoxCPMGeneratorOptions`)

| Field | Description |
|-------|-------------|
| `device` | Explicit Candle `Device` (overrides auto-detect) |
| `device_id` | GPU ordinal when auto-detecting CUDA/Metal |
| `dtype` | Main compute dtype override (`F16`, `BF16`, `F32`, …) |
| `vae_dtype` | Audio VAE compute dtype override |
| `quant` | `VoxCPMQuantConfig` (weight mode + skip patterns) |
| `seed` | Fixed RNG seed for reproducible GPU generation |

When `dtype` / `vae_dtype` are unset (`auto`), the loader picks lower precision on GPU for **non-quant** runs (typically `F16` for the LM). When **quant is enabled** and `dtype=auto`, activations default to **F32** to match `QMatMul` and avoid per-layer cast churn.

### Bailu integration

In [`bailu.toml`](../../bailu.toml):

```toml
[tts.voxcpm]
quant = "q8_0"
dtype = "auto"
vae_dtype = "auto"
```

Override at runtime: `BAILU_VOXCPM_QUANT=none|q8_0|…`. Full operator guide: [`docs/VOXCPM_QUANT.md`](../../docs/VOXCPM_QUANT.md).

### Generation presets (`VoxCPMGenerationConfig`)

| Preset | Use case |
|--------|----------|
| `simple()` | Short zero-ref utterances |
| `voice_clone()` | Reference-guided synthesis (default) |
| `low_latency()` | Smaller VAE batches, fewer stop checks |
| `metal_rtf()` | Metal throughput tuning |

Key fields: `inference_timesteps`, `cfg_value`, `max_len`, `stream_decode_latent_batch`, `stop_check_interval`, `retry_badcase`.

## Weight quantization

Quantization applies at load time to eligible linear layers via Candle `QTensor` + `QMatMul`. Attention uses **separate Q/K/V projections** (not fused QKV) so each matrix quantizes cleanly.

### Modes

| Mode | Status | Aliases |
|------|--------|---------|
| `none` | default (FP) | |
| **`q8_0`** | **recommended** | `q8`, `q80` |
| `q4_k` | experimental | `q4k` |
| `q5_k` | experimental | `q5k` |
| `q6_k` | experimental | `q6k` |

### Layer policy

**Quantized:** LM / residual LM / feature encoder / DiT linear layers (MLP + separate Q/K/V/O projections).

**Always FP:** `embed_tokens`, `stop_head`, `stop_proj`, `fsq_layer`, **Audio VAE**.

K-quants use block size 256; layers whose input dim is not divisible by 256 may fall back to `q8_0`. Inspect load output:

```text
VOXCPM_QUANT_STATS requested=q8_0 quantized=347 skipped=4 fallback_q8=0 mixed_k=false
VOXCPM_QUANT_JSON {...}
```

Programmatic stats: `generator.quant_stats()` after `new_with_options`.

### Dtype with quant

When quant is enabled and `dtype=auto`, activations default to **F32** (see `get_quant_compute_dtype` in `utils/device.rs`).

```rust
use voxcpm_rs::{VoxCPMQuantConfig, VoxCPMWeightQuant};

let quant = VoxCPMQuantConfig::with_weight(VoxCPMWeightQuant::Q8_0);

// Skip attention + bridge layers; quantize MLP/down_proj only
let quant = VoxCPMQuantConfig::quality_first(VoxCPMWeightQuant::Q8_0);
```

By default, weights stay in `QMatMul` form. `VOXCPM_QUANT_DEQUANT_LINEAR=1` dequantizes to dense `Linear` at load (**debug only**; no speed benefit).

### Quality gates

`--compare-fp` / `VOXCPM_COMPARE_FP=1` synthesizes an FP baseline. Mode-specific PCM correlation floors:

| Mode | Floor |
|------|-------|
| `q8_0` | 0.95 |
| `q6_k` | 0.90 |
| `q4_k` / `q5_k` | 0.85 |

WAV/mel comparison: `scripts/voxcpm_quant_analysis/compare_wav.py --strict --min-correlation 0.95`.

### Reference performance (Metal, release)

VoxCPM-0.5B, `"测试"`, batch — your numbers will vary:

| Mode | Load | RTF | FP corr |
|------|------|-----|---------|
| `none` | ~0.5s | ~0.62 | — |
| `q8_0` | ~1.0s | ~0.56 | ~0.998 |

Streaming often improves TTFA for `q8_0` vs FP. Stage split available via `--profile` (`cfm`, `lm`, `vae`, `VOXCPM_BOTTLENECK_HINT`).

### Benchmarking

```bash
cargo run --release -p voxcpm-rs --features metal --example voxcpm2_benchmark -- \
  --model models/VoxCPM-0.5B --text "测试" --quant q8_0 --compare-fp --profile --runs 3

# Repo scripts (from workspace root):
VOXCPM_QUANT=q8_0 ./scripts/benchmark_voxcpm_non_stream.sh
./scripts/voxcpm_quant_analysis/run_baseline.sh
./scripts/voxcpm_check.sh
```

Emits `VOXCPM_BENCH_JSON`, `VOXCPM_QUANT_JSON`, and `VOXCPM_BOTTLENECK_HINT`.

## Environment variables

| Variable | Effect |
|----------|--------|
| `VOXCPM_DEVICE=cpu` | Force CPU even when GPU features are enabled |
| `VOXCPM_SEED` | Default generation seed (GPU) |
| `VOXCPM_QUANT_AUDIT=1` | Log per-module quant decisions to stderr |
| `VOXCPM_QUANT_DEQUANT_LINEAR=1` | Dequantize to dense `Linear` at load (debug only) |
| `VOXCPM_COMPARE_FP=1` | Enable FP baseline comparison in examples |
| `VOXCPM_BENCH_PROFILE=1` | Capture stage timings for benchmark JSON |
| `VOXCPM_PROFILE` | Print prefill / inference stage timings |
| `VOXCPM_PROFILE_STREAM` | Stream chunk timing (inference vs VAE decode) |

## Model layout

Point `--model` / `new_with_options` at a directory containing:

```
models/VoxCPM2/
├── config.json          # required — architecture + hyperparameters
├── tokenizer files      # bundled with the HF checkpoint
├── *.safetensors        # main LM weights (VAE files may be separate *vae*.safetensors)
└── *.bin / *.pth        # alternative PyTorch weight layouts (also supported)
```

Download checkpoints from [OpenBMB on Hugging Face](https://huggingface.co/openbmb) (e.g. `VoxCPM-0.5B`, `VoxCPM2`).

## Cargo features

| Feature | Enables |
|---------|---------|
| `metal` | Apple GPU via Candle |
| `cuda` | NVIDIA GPU via Candle |
| `flash-attn` | Optional flash-attention backend |

Default features are empty; pick one GPU backend at build time for examples and downstream crates.

## Public API surface

Re-exported from the crate root:

- `VoxCPMGenerator`, `VoxCPMGeneratorOptions`, `VoxCPMGenerationConfig`
- `VoxCPMConfig`, `AudioVaeConfig`
- `VoxCPMQuantConfig`, `VoxCPMWeightQuant`, `QuantStats`
- `audio_quality_ok`, `pcm_correlation`, `compare_fp_min_correlation`
- `BenchmarkMetrics`, `StageProfile`, `bottleneck_hint`
- `COMPARE_FP_DEFAULT_SEED`, `COMPARE_FP_MIN_CORRELATION`

## Testing

Unit tests run on CPU without model files:

```bash
cargo test -p voxcpm-rs
# or from repo root:
./scripts/voxcpm_check.sh
```

Doc tests and GPU examples require downloaded weights and a GPU build; exclude this crate in CI smoke tests (as Bailu does with `--exclude voxcpm-rs`).

## Acknowledgments

- Original VoxCPM integration reference: [aha](https://github.com/jhqxxx/aha)
- ML framework: [Candle](https://github.com/huggingface/candle)
- Model weights: [OpenBMB / VoxCPM](https://huggingface.co/openbmb)

## License

Apache-2.0
