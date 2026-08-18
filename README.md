# voxcpm-rs

Pure Rust implementation of [VoxCPM](https://huggingface.co/openbmb) text-to-speech, built on [Candle](https://github.com/huggingface/candle). Supports zero-shot TTS, voice cloning from a reference clip, streaming audio output, and optional weight-only quantization.


## Features

- **VoxCPM family** — auto-detects VoxCPM, VoxCPM1.5, and VoxCPM2 from `config.json`
- **Voice cloning** — reference WAV + transcript prompt cache for timbre/style transfer
- **Streaming** — latent-to-audio chunks as `Iterator<Item = Result<Tensor>>`, plus PCM/WAV stream helpers
- **In-memory output** — `to_wav`, `to_pcm`, and streaming byte iterators without touching disk
- **GPU backends** — optional Metal (macOS) or CUDA via Candle feature flags
- **Weight quantization** — live `QMatMul` paths for `q8_0` / K-quants with quality gates
- **Metal dispatch cuts** — fused QKV / gate+up matmuls, fused RoPE, GQA without `repeat_kv` (decode-only fused SDPA)
- **Benchmarking** — RTF, TTFA, FP-vs-quant / reference-WAV correlation checks

## Requirements

- Rust 1.85+ (Edition 2021)
- A downloaded VoxCPM checkpoint directory (see [Model layout](#model-layout))
- For GPU inference, build with `--features metal` (macOS) or `--features cuda` (Linux)

> **Platform status:** developed and tuned for **macOS + Apple Silicon (Metal) only** — all reference numbers below were measured on Metal. The `cuda` feature compiles but is **not yet tested on Linux + NVIDIA** (no GPU available for validation). Reports and patches for CUDA are welcome.

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

RTF / TTFA benchmark (VoxCPM2 streaming, official full CFG):

```bash
cargo run --release -p voxcpm-rs --features metal --example voxcpm2_benchmark -- \
  --model models/VoxCPM2 --stream \
  --quant q8_0 --dtype f16 --cfg-full-fraction 1.0 \
  --inference-timesteps 10 \
  --stream-decode-initial-latent-batch 4 \
  --stream-decode-latent-batch 8 \
  --stop-check-interval 1 --min-len 2 \
  --ref-wav models/paimon_01.wav \
  --ref-text "Reference transcript matching the clip." \
  --text "好的，我来帮你查一下。请稍等片刻。" \
  --warmup 1 --runs 5 --profile
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

### Generation presets (`VoxCPMGenerationConfig`)

| Preset | Use case |
|--------|----------|
| `simple()` | Short zero-ref utterances |
| `voice_clone()` | Reference-guided synthesis (default) |
| `low_latency()` | Smaller VAE batches, fewer stop checks |
| `metal_rtf()` | Metal throughput tuning |

Default `voice_clone()` / `Default` (balanced preset): `inference_timesteps=10`, `cfg_value=2.0`, `cfg_full_fraction=1.0`, `min_len=2`, `max_len=500`, `retry_badcase_ratio_threshold=6.0`, `stream_decode_initial_latent_batch=4`, `stream_decode_latent_batch=8`, `stop_check_interval=1`. (`retry_badcase` only scales the latent budget in this port — it does not retry seeds.) After generation, `VoxCPMGenerator::last_diagnostics()` reports `stop_reason` (`stop_head` | `max_len`), latent count, and effective max length.

## Weight quantization

Quantization applies at load time to eligible linear layers via Candle `QTensor` + `QMatMul`. By default, **Q/K/V and gate/up are fused** into one matmul each (`FusedLinearX`): weights are row-concatenated then quantized once (bit-exact vs separate `q8_0` when `in_features` is block-aligned). Disable with `VOXCPM_FUSE_PROJ=0` (falls back to separate projections; also used automatically when `quality_first` skip patterns hit those modules).

### Modes

| Mode | Status | Aliases |
|------|--------|---------|
| `none` | default (FP) | |
| **`q8_0`** | **recommended** | `q8`, `q80` |
| `q4_k` | experimental | `q4k` |
| `q5_k` | experimental | `q5k` |
| `q6_k` | experimental | `q6k` |

### Layer policy

**Quantized:** LM / residual LM / feature encoder / DiT linear layers (fused QKV + gate/up when enabled, plus `o_proj` / `down_proj` and bridge projections).

**Always FP:** `embed_tokens`, `stop_head`, `stop_proj`, `fsq_layer`, **Audio VAE**.

K-quants use block size 256; layers whose input dim is not divisible by 256 may fall back to `q8_0`. Inspect load output:

```text
VOXCPM_QUANT_STATS requested=q8_0 quantized=252 skipped=4 fallback_q8=0 mixed_k=false
VOXCPM_QUANT_JSON {...}
```

(With fusion disabled, VoxCPM2 reports ~432 quantized matrices; fused QKV/gate+up merges three+two into one each → ~252.)

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

### Reference performance (Metal, release)

Measured on **Mac mini M4 Pro (Apple M4 Pro, 64 GB)**, 2026-08-19. Numbers vary by machine, text length, and cold vs warm load.

**VoxCPM-0.5B**, `"测试"`, batch (median of 3):

| Mode | Load | RTF | FP corr |
|------|------|-----|---------|
| `none` | ~0.5 s | ~0.70 | — |
| `q8_0` | ~1.2 s | ~0.69 | gate-checked via `--compare-fp` |

**VoxCPM2** streaming voice-clone (steps **10**, `cfg_full_fraction=1.0`, init/latent/stop **4/8/1**, medium Chinese utterance ~4 s audio; seed 42, median of 5):

| Build | RTF | TTFA | Notes |
|-------|----:|-----:|-------|
| Eager (`VOXCPM_FUSED_SDPA=0 VOXCPM_FUSE_PROJ=0 VOXCPM_FUSED_ROPE=0`, q8_0+f16) | 1.18 | 0.98 s | 432 quantized matrices |
| FP reference (none/auto) | 1.08 | 0.88 s | |
| **Current defaults** (fused proj + RoPE + GQA, q8_0+f16) | **0.98** | **0.85 s** | 252 quantized matrices; corr 0.990 vs FP |

Stage split via `--profile` (`cfm`, `lm`, `vae`, `VOXCPM_BOTTLENECK_HINT`). On M4 Pro the every-latent stop head is the dominant stage (~38%), ahead of CFM/DiT (~32%).

### Benchmarking

```bash
cargo run --release -p voxcpm-rs --features metal --example voxcpm2_benchmark -- \
  --model models/VoxCPM-0.5B --text "测试" --quant q8_0 --compare-fp --profile --runs 3
```

Emits `VOXCPM_BENCH_JSON`, `VOXCPM_QUANT_JSON`, and `VOXCPM_BOTTLENECK_HINT`.

## Metal inference optimizations

VoxCPM2 DiT/LM inference is **launch-overhead-bound** on Metal (many tiny kernels per Euler step), not weight-bandwidth-bound. Defaults cut dispatches without changing math:

| Optimization | Where | Escape hatch |
|--------------|-------|--------------|
| Fused QKV + gate/up `QMatMul` | `linear.rs` / `common.rs` | `VOXCPM_FUSE_PROJ=0` |
| Fused RoPE (`candle_nn::rotary_emb::rope`, F32 on F16 acts) | `position_embed/rope.rs` | `VOXCPM_FUSED_ROPE=0` |
| GQA without `repeat_kv` (unmasked DiT / decode) | `common.rs` | — (masked LM prefill still tiles) |
| Fused Metal SDPA | `common.rs` `attention_forward` | `VOXCPM_FUSED_SDPA=0` |

**SDPA policy:** default `VOXCPM_FUSED_SDPA_MAX_QLEN=1` (decode / vector kernel only). DiT seq≈11 stays on eager attention — the Metal vector kernel NaNs at `q_seq>1` for this GQA shape, and the tiled full kernel previously corrupted audio (corr≈0.03). Do not raise `MAX_QLEN` for production.

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
| `VOXCPM_PROFILE_SYNC=1` | Sync GPU around stage timers (inflates wall RTF) |
| `VOXCPM_FUSE_PROJ=0` | Disable fused QKV / gate+up projections |
| `VOXCPM_FUSED_ROPE=0` | Disable candle fused RoPE (eager rotate-half) |
| `VOXCPM_FUSED_SDPA=0` | Force eager attention (disable Metal SDPA) |
| `VOXCPM_FUSED_SDPA_MAX_QLEN` | Max fused query length (**default 1**; experimental only) |

## Model layout

Point `--model` / `new_with_options` at a directory containing:

```
models/VoxCPM2/
├── config.json          # required — architecture + hyperparameters
├── tokenizer files      # bundled with the HF checkpoint
├── *.safetensors        # main LM weights (VAE files may be separate *vae*.safetensors)
└── *.bin / *.pth        # alternative PyTorch weight layouts (also supported)
```

Download checkpoints from Hugging Face:

- [openbmb/VoxCPM2](https://huggingface.co/openbmb/VoxCPM2) — 2.29B, tokenizer-free diffusion-AR TTS, 48 kHz, voice cloning + streaming
- [openbmb/VoxCPM-0.5B](https://huggingface.co/openbmb/VoxCPM-0.5B) — 0.5B, 16 kHz

Both are Apache-2.0.

## Cargo features

| Feature | Enables |
|---------|---------|
| `metal` | Apple GPU via Candle (tested / tuned) |
| `cuda` | NVIDIA GPU via Candle (**untested** — see platform status) |
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
```

Doc tests and GPU examples require downloaded weights and a GPU build; exclude this crate (`--exclude voxcpm-rs`) in CI smoke tests without a GPU.

## Acknowledgments

- Original VoxCPM integration reference: [aha](https://github.com/jhqxxx/aha)
- ML framework: [Candle](https://github.com/huggingface/candle)
- Model weights: [OpenBMB / VoxCPM](https://huggingface.co/openbmb)

## License

Apache-2.0
