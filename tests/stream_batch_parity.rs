//! Stream-concat vs batch PCM parity (onset region).
//!
//! Run when VoxCPM weights are available locally:
//! `VOXCPM2_MODEL_PATH=models/VoxCPM-0.5B cargo test -p voxcpm-rs --test stream_batch_parity -- --ignored --nocapture`

use voxcpm_rs::profile::pcm_correlation;
use voxcpm_rs::{VoxCPMGenerationConfig, VoxCPMGenerator, COMPARE_FP_DEFAULT_SEED};
use voxcpm_rs::VoxCPMGeneratorOptions;

const ONSET_SAMPLES: usize = 96_000; // 2 s @ 48 kHz
const MIN_ONSET_CORR: f64 = 0.95;

fn example_text() -> String {
    "诶嘿！派蒙正在等着旅行者带派蒙去冒险呢！".to_string()
}

#[test]
#[ignore = "requires GPU and downloaded VoxCPM weights"]
fn stream_concat_matches_batch_onset() -> anyhow::Result<()> {
    let model_path =
        std::env::var("VOXCPM2_MODEL_PATH").unwrap_or_else(|_| "models/VoxCPM-0.5B".to_string());
    let mut options = VoxCPMGeneratorOptions::default();
    options.seed = Some(COMPARE_FP_DEFAULT_SEED);
    let mut generator = VoxCPMGenerator::new_with_options(&model_path, &options)?;

    if let (Ok(ref_wav), Ok(ref_text)) = (
        std::env::var("VOXCPM_REF_WAV"),
        std::env::var("VOXCPM_REF_TEXT"),
    ) {
        generator.build_prompt_cache(ref_text, ref_wav)?;
    }

    let text = example_text();
    let config = VoxCPMGenerationConfig::voice_clone();

    let batch_tensor = generator.generate_with_config(text.clone(), config)?;
    let batch_pcm = generator.to_pcm(&batch_tensor)?;

    let mut stream_pcm = Vec::new();
    for chunk in generator.generate_pcm_stream_with_config(text, config)? {
        stream_pcm.extend(chunk?);
    }

    assert!(
        !batch_pcm.is_empty() && !stream_pcm.is_empty(),
        "expected non-empty batch and stream PCM"
    );

    let onset_len = ONSET_SAMPLES.min(batch_pcm.len()).min(stream_pcm.len());
    let onset_corr = pcm_correlation(
        &batch_pcm[..onset_len],
        &stream_pcm[..onset_len],
    );
    let full_len = batch_pcm.len().min(stream_pcm.len());
    let full_corr = pcm_correlation(&batch_pcm[..full_len], &stream_pcm[..full_len]);

    eprintln!(
        "stream_batch_parity onset_corr={onset_corr:.4} full_corr={full_corr:.4} onset_samples={onset_len} full_samples={full_len}"
    );

    assert!(
        onset_corr >= MIN_ONSET_CORR,
        "onset correlation {onset_corr:.4} below threshold {MIN_ONSET_CORR}"
    );
    assert!(
        full_corr >= MIN_ONSET_CORR,
        "full correlation {full_corr:.4} below threshold {MIN_ONSET_CORR}"
    );

    Ok(())
}
