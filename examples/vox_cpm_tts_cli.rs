use anyhow::{bail, Result};
use candle_core::Tensor;
use clap::Parser;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::time::Instant;
use voxcpm_rs::VoxCPMGenerator;

#[derive(Parser, Debug)]
#[command(name = "voxcpm-cli", about = "VoxCPM TTS / Voice Cloning CLI")]
struct Args {
    /// Model path
    #[arg(long, default_value = "./VoxCPM-0.5B")]
    model: PathBuf,

    /// Reference wav (optional)
    #[arg(long)]
    ref_wav: Option<PathBuf>,

    /// Reference text (required if --ref-wav is set)
    #[arg(long, requires = "ref_wav")]
    ref_text: Option<String>,

    /// Single input text
    #[arg(long, conflicts_with = "text_file")]
    text: Option<String>,

    /// Input text file (one line per query)
    #[arg(long, conflicts_with = "text")]
    text_file: Option<PathBuf>,

    /// Output prefix
    #[arg(long, default_value = "output")]
    out: String,

    /// Enable streaming TTS (no-ref only)
    #[arg(long, default_value = "false")]
    stream: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();

    let texts = load_texts(&args)?;

    println!("Loading model...");
    let mut generator = VoxCPMGenerator::new(args.model.to_str().unwrap(), None, None)?;
    println!("✓ Model loaded");

    // --------------------------------------------------
    // 🔹 Prompt cache initialization (ONCE)
    // --------------------------------------------------
    let use_ref = match (&args.ref_wav, &args.ref_text) {
        (Some(wav), Some(text)) => {
            println!("Initializing prompt cache...");
            generator.build_prompt_cache(text.clone(), wav.to_string_lossy().to_string())?;
            println!("✓ Prompt cache ready");
            true
        }
        _ => {
            println!("Running in no-ref TTS mode");
            false
        }
    };

    if args.stream {
        // --------------------------------------------------
        // 🔹 Inference loop (NO cache init here)
        // --------------------------------------------------
        let mut total_time_cost = 0.0f64;
        let mut total_audio_time = 0.0f64;
        for (i, text) in texts.iter().enumerate() {
            println!("▶ Generating (streaming) {}/{}", i + 1, texts.len());
            let start = Instant::now();
            let mut first_chunk_time = None;

            let stream: Box<dyn Iterator<Item = Result<Tensor>> + '_> = if use_ref {
                Box::new(generator.generate_stream_use_prompt_cache(
                    text.clone(),
                    5,    // min_len
                    500,  // max_len
                    15,   // timesteps
                    2.5,  // cfg
                    true, // use cache
                    6.0,  // retry threshold
                )?)
            } else {
                Box::new(generator.generate_stream_simple(text.clone())?)
            };

            let mut chunks = Vec::new();
            for (_chunk_idx, chunk_res) in stream.enumerate() {
                let chunk = chunk_res?;
                if first_chunk_time.is_none() {
                    first_chunk_time = Some(start.elapsed().as_secs_f64());
                    println!(
                        "  ↳ First chunk generated in {:.2}s",
                        first_chunk_time.unwrap()
                    );
                }
                chunks.push(chunk);
                print!(".");
                use std::io::Write;
                std::io::stdout().flush()?;
            }
            println!();

            if chunks.is_empty() {
                println!("! No audio generated for text: {}", text);
                continue;
            }

            let tensor = Tensor::cat(&chunks, 1)?;
            let wav = generator.to_wav(&tensor)?;
            let text_prefix = substring_by_char_count(text, 0, 4);

            let time_cost = start.elapsed().as_secs_f64();
            let audio_len = wav.len() as f64 / generator.sample_rate() as f64 / 2.0;

            total_audio_time += audio_len;
            total_time_cost += time_cost;

            let rtf = time_cost / audio_len;
            println!(
                "✓ Streaming finished for text {}... , using {:.2} seconds, output {:.2} seconds, rtf {:.2}\n",
                text_prefix, time_cost, audio_len, rtf,
            );
            let output = format!("{}_{}_stream.wav", args.out, i + 1);
            std::fs::write(&output, wav)?;
            println!("✓ Saved {}\n", output);
        }
        println!(
            "Average time cost: {:.2} seconds",
            total_time_cost / texts.len() as f64
        );
        println!(
            "Average audio time: {:.2} seconds",
            total_audio_time / texts.len() as f64
        );
        println!("Average rtf: {:.2}", total_time_cost / total_audio_time);
    } else {
        // --------------------------------------------------
        // 🔹 Inference loop (NO cache init here)
        // --------------------------------------------------
        let mut total_time_cost = 0.0f64;
        let mut total_audio_time = 0.0f64;
        for (i, text) in texts.iter().enumerate() {
            println!("▶ Generating {}/{}", i + 1, texts.len());
            let start = Instant::now();
            let tensor = if use_ref {
                generator.generate_use_prompt_cache(
                    text.clone(),
                    5,    // min_len
                    500,  // max_len
                    15,   // timesteps
                    2.5,  // cfg
                    true, // use cache
                    6.0,  // retry threshold
                )?
            } else {
                generator.generate_simple(text.clone())?
            };

            let wav = generator.to_wav(&tensor)?;
            let text_prefix = substring_by_char_count(text, 0, 4);

            let time_cost = start.elapsed().as_secs_f64();
            let audio_len = wav.len() as f64 / generator.sample_rate() as f64 / 2.0;

            total_audio_time += audio_len;
            total_time_cost += time_cost;

            let rtf = time_cost / audio_len;
            println!(
            "✓ Generated for text {}... , using {:.2} seconds, output {:.2} seconds, rtf {:.2}\n",
            text_prefix, time_cost, audio_len, rtf,
        );
            let output = format!("{}_{}.wav", args.out, i + 1);
            std::fs::write(&output, wav)?;
            println!("✓ Saved {}\n", output);
        }
        println!(
            "Average time cost: {:.2} seconds",
            total_time_cost / texts.len() as f64
        );
        println!(
            "Average audio time: {:.2} seconds",
            total_audio_time / texts.len() as f64
        );
        println!("Average rtf: {:.2}", total_time_cost / total_audio_time);
    }

    Ok(())
}

/* ---------------- helpers ---------------- */

fn load_texts(args: &Args) -> Result<Vec<String>> {
    match (&args.text, &args.text_file) {
        (Some(t), None) => Ok(vec![t.clone()]),
        (None, Some(file)) => read_lines(file),
        (None, None) => bail!("Either --text or --text-file must be provided"),
        _ => unreachable!(),
    }
}

fn read_lines(path: &PathBuf) -> Result<Vec<String>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);

    let lines: Vec<String> = reader
        .lines()
        .filter_map(Result::ok)
        .map(|l| l.trim().to_string())
        .filter(|l| !l.is_empty())
        .collect();

    if lines.is_empty() {
        bail!("Input text file is empty");
    }

    Ok(lines)
}

fn substring_by_char_count(s: &str, start_char: usize, char_count: usize) -> &str {
    let mut byte_start = 0;
    let mut byte_end = 0;

    for (i, char) in s.chars().enumerate() {
        if i == start_char {
            byte_start = byte_end;
        }
        byte_end += char.len_utf8();
        if i == start_char + char_count - 1 {
            // adjust for count vs index
            break;
        }
    }

    &s[byte_start..byte_end]
}
