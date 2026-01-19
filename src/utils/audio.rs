use anyhow::{anyhow, Result};
use candle_core::{DType, Device, Tensor, D};
use candle_nn::{Conv1d, Conv1dConfig, Module};
use hound::{SampleFormat, WavReader};
use num::integer::gcd;
use std::io::Cursor;
use symphonia::core::audio::{AudioBufferRef, Signal};
use symphonia::core::codecs::DecoderOptions;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;

use super::tensor::linspace;

/// Resampling method
#[derive(Debug, Clone, Copy)]
pub enum ResamplingMethod {
    SincInterpHann,
}

/// Zero-order modified Bessel function I0
fn i0(x: f32) -> f32 {
    let mut result = 1.0;
    let mut term = 1.0;
    let half_x_sq = x * x / 4.0;

    for k in 1..50 {
        term = term * half_x_sq / (k * k) as f32;
        result += term;
        if term < 1e-12 {
            break;
        }
    }
    result
}

/// Get sinc resample kernel
pub fn get_sinc_resample_kernel(
    orig_freq: i64,
    new_freq: i64,
    gcd_val: i64,
    lowpass_filter_width: i64,
    rolloff: f64,
    resampling_method: ResamplingMethod,
    device: &Device,
) -> Result<(Tensor, i64)> {
    if orig_freq <= 0 || new_freq <= 0 {
        return Err(anyhow!("Frequencies must be positive"));
    }

    let orig_freq = orig_freq / gcd_val;
    let new_freq = new_freq / gcd_val;
    let base_freq = (orig_freq.min(new_freq) as f64) * rolloff;
    let width_f = (lowpass_filter_width as f64) * (orig_freq as f64) / base_freq;
    let width = width_f.ceil() as i64;

    let idx = Tensor::arange(-width as f32, (width + orig_freq) as f32, device)?
        .affine(1.0 / orig_freq as f64, 0.0)?
        .unsqueeze(0)?
        .unsqueeze(0)?;

    let t = Tensor::arange_step(0.0, -new_freq as f32, -1.0, device)?
        .affine(1.0 / new_freq as f64, 0.0)?
        .unsqueeze(D::Minus1)?
        .unsqueeze(D::Minus1)?
        .broadcast_add(&idx)?
        .affine(base_freq, 0.0)?;
    let t = t.clamp(-lowpass_filter_width as f32, lowpass_filter_width as f32)?;

    let window = match resampling_method {
        ResamplingMethod::SincInterpHann => {
            let window_arg = t.affine(
                std::f64::consts::PI / (lowpass_filter_width as f64) / 2.0,
                0.0,
            )?;
            window_arg.cos()?.sqr()?
        }
    };

    let scale = base_freq / (orig_freq as f64);
    let t_scaled = t.affine(std::f64::consts::PI, 0.0)?;
    let t_zeros = Tensor::zeros_like(&t_scaled)?;
    let t_ones = Tensor::ones_like(&t_scaled)?;
    let mask = t_scaled.eq(&t_zeros)?;
    let sinc = mask.where_cond(&t_ones, &t_scaled.sin()?.div(&t_scaled)?)?;
    let kernels = sinc.mul(&window)?.affine(scale, 0.0)?;

    Ok((kernels, width))
}

/// Apply sinc resample kernel
pub fn apply_sinc_resample_kernel(
    waveform: &Tensor,
    orig_freq: i64,
    new_freq: i64,
    gcd_val: i64,
    kernel: &Tensor,
    width: i64,
) -> Result<Tensor> {
    let orig_freq = orig_freq / gcd_val;
    let new_freq = new_freq / gcd_val;

    let dims = waveform.dims();
    let waveform_flat = waveform.reshape(((), dims[dims.len() - 1]))?;
    let (num_wavs, length) = waveform_flat.dims2()?;
    let padded_waveform =
        waveform.pad_with_zeros(D::Minus1, width as usize, (width + orig_freq) as usize)?;

    let waveform_3d = padded_waveform.unsqueeze(1)?;
    let config = Conv1dConfig {
        padding: 0,
        stride: orig_freq as usize,
        dilation: 1,
        groups: 1,
        cudnn_fwd_algo: None,
    };

    let conv1d = Conv1d::new(kernel.clone(), None, config);
    let conv_output = conv1d.forward(&waveform_3d)?;
    let conv_transposed = conv_output.transpose(1, 2)?.reshape((num_wavs, ()))?;
    let target_length = ((new_freq as f64 * length as f64) / orig_freq as f64).ceil() as usize;
    let resampled_flat =
        conv_transposed.narrow(1, 0, target_length.min(conv_transposed.dim(1)?))?;

    let mut new_dims = dims.to_vec();
    let last_dim = new_dims.len() - 1;
    new_dims[last_dim] = resampled_flat.dim(1)?;
    let resampled = resampled_flat.reshape(new_dims)?;

    Ok(resampled)
}

/// Resample audio
pub fn resample(
    waveform: &Tensor,
    orig_freq: i64,
    new_freq: i64,
    lowpass_filter_width: i64,
    rolloff: f64,
    resampling_method: ResamplingMethod,
) -> Result<Tensor> {
    if orig_freq <= 0 || new_freq <= 0 {
        return Err(anyhow!("Frequencies must be positive"));
    }

    if orig_freq == new_freq {
        return Ok(waveform.clone());
    }

    let gcd_val = gcd(orig_freq, new_freq);
    let device = waveform.device();

    let (kernel, width) = get_sinc_resample_kernel(
        orig_freq,
        new_freq,
        gcd_val,
        lowpass_filter_width,
        rolloff,
        resampling_method,
        device,
    )?;
    apply_sinc_resample_kernel(waveform, orig_freq, new_freq, gcd_val, &kernel, width)
}

/// Simple resample with default parameters
pub fn resample_simple(waveform: &Tensor, orig_freq: i64, new_freq: i64) -> Result<Tensor> {
    resample(
        waveform,
        orig_freq,
        new_freq,
        6,
        0.99,
        ResamplingMethod::SincInterpHann,
    )
}

/// Load audio using Symphonia (supports multiple formats)
pub fn load_audio_use_symphonia(audio_vec: Vec<u8>, device: &Device) -> Result<(Tensor, usize)> {
    let extension = get_audio_format_from_bytes(&audio_vec)?;
    let content = Cursor::new(audio_vec);
    let mss = MediaSourceStream::new(Box::new(content), Default::default());

    let mut hint = Hint::new();
    hint.with_extension(&extension);

    let probed = symphonia::default::get_probe().format(
        &hint,
        mss,
        &FormatOptions::default(),
        &MetadataOptions::default(),
    )?;

    let mut format = probed.format;
    let track = format
        .default_track()
        .ok_or("No default track found")
        .map_err(|e| anyhow!("symphonia read err: {}", e))?;
    let mut channels = 1;
    let sample_rate = track.codec_params.sample_rate.unwrap_or(0);

    let mut decoder =
        symphonia::default::get_codecs().make(&track.codec_params, &DecoderOptions::default())?;

    let mut all_samples: Vec<Vec<f32>> = Vec::new();

    while let Ok(packet) = format.next_packet() {
        match decoder.decode(&packet) {
            Ok(decoded) => match decoded {
                AudioBufferRef::F32(buf) => {
                    channels = buf.spec().channels.count();
                    for channel in 0..channels {
                        if all_samples.len() <= channel {
                            all_samples.push(Vec::new());
                        }
                        let channel_data = buf.chan(channel);
                        all_samples[channel].extend_from_slice(channel_data);
                    }
                }
                AudioBufferRef::S16(buf) => {
                    channels = buf.spec().channels.count();
                    for channel in 0..channels {
                        if all_samples.len() <= channel {
                            all_samples.push(Vec::new());
                        }
                        let channel_data = buf.chan(channel);
                        let float_samples: Vec<f32> =
                            channel_data.iter().map(|&s| s as f32 / 32768.0).collect();
                        all_samples[channel].extend(float_samples);
                    }
                }
                AudioBufferRef::S24(buf) => {
                    channels = buf.spec().channels.count();
                    for channel in 0..channels {
                        if all_samples.len() <= channel {
                            all_samples.push(Vec::new());
                        }
                        let channel_data = buf.chan(channel);
                        let float_samples: Vec<f32> = channel_data
                            .iter()
                            .map(|&s| s.inner() as f32 / 8388608.0)
                            .collect();
                        all_samples[channel].extend(float_samples);
                    }
                }
                _ => {}
            },
            Err(_) => break,
        }
    }

    let mut audio_tensor = Tensor::new(all_samples, device)?;
    if channels > 1 {
        audio_tensor = audio_tensor.mean_keepdim(0)?;
    }
    Ok((audio_tensor, sample_rate as usize))
}

/// Detect audio format from bytes
pub fn get_audio_format_from_bytes(bytes: &[u8]) -> Result<String> {
    if bytes.len() < 12 {
        return Err(anyhow!("bytes too short"));
    }

    if bytes.starts_with(&[0x52, 0x49, 0x46, 0x46]) && bytes.len() >= 12 {
        if bytes[8..12] == [0x57, 0x41, 0x56, 0x45] {
            return Ok("wav".to_string());
        }
    } else if bytes.starts_with(&[0xFF, 0xFB]) || bytes.starts_with(&[0xFF, 0xF3]) {
        return Ok("mp3".to_string());
    }

    Err(anyhow!("Unknown audio format"))
}

/// Load audio file with optional resampling
pub fn load_audio_with_resample(
    path: &str,
    device: &Device,
    target_sample_rate: Option<usize>,
) -> Result<Tensor> {
    let audio_vec = std::fs::read(path)?;
    let (mut audio, sr) = load_audio_use_symphonia(audio_vec, device)?;

    if let Some(target_sample_rate) = target_sample_rate {
        if target_sample_rate != sr {
            audio = resample_simple(&audio, sr as i64, target_sample_rate as i64)?;
        }
    }
    Ok(audio)
}

/// Save audio tensor to WAV file
pub fn save_wav(audio: &Tensor, save_path: &str, sample_rate: u32) -> Result<()> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };

    assert_eq!(audio.dim(0)?, 1, "audio channel must be 1");
    let max = audio.abs()?.max_all()?;
    let max = max.to_scalar::<f32>()?;
    let ratio = if max > 1.0 { 32767.0 / max } else { 32767.0 };
    let audio = audio.squeeze(0)?;
    let audio_vec = audio.to_vec1::<f32>()?;
    let mut writer = hound::WavWriter::create(save_path, spec)?;

    for i in audio_vec {
        let sample_i16 = (i * ratio).round() as i16;
        writer.write_sample(sample_i16)?;
    }
    writer.finalize()?;
    Ok(())
}

pub fn to_wav(audio: &Tensor, sample_rate: u32) -> Result<Vec<u8>> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };

    assert_eq!(audio.dim(0)?, 1, "audio channel must be 1");
    let max = audio.abs()?.max_all()?;
    let max = max.to_scalar::<f32>()?;
    let ratio = if max > 1.0 { 32767.0 / max } else { 32767.0 };
    let audio = audio.squeeze(0)?;
    let audio_vec = audio.to_vec1::<f32>()?;
    let mut buffer = Vec::new();
    let cursor = Cursor::new(&mut buffer);
    let mut writer = hound::WavWriter::new(cursor, spec).unwrap();

    for i in audio_vec {
        let sample_i16 = (i * ratio).round() as i16;
        writer.write_sample(sample_i16)?;
    }
    writer.finalize()?;
    Ok(buffer)
}
