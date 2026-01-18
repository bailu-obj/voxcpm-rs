//! # VoxCPM:  Standalone Text-to-Speech Library
//!
//! A pure Rust implementation of the VoxCPM TTS model.

pub mod audio_vae;
pub mod common;
pub mod config;
pub mod generate;
pub mod minicpm4;
pub mod models;
pub mod position_embed;
pub mod tokenizer;
pub mod utils;

pub use config::{AudioVaeConfig, VoxCPMConfig};
pub use generate::VoxCPMGenerator;

pub const VERSION: &str = env!("CARGO_PKG_VERSION");
