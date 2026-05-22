// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Kind Computers, LLC.

#![allow(clippy::items_after_test_module)]

mod archive;
mod batch;
mod cleanup;
mod engine;
mod gui;
mod hrtf;
mod native_diagnostics;
mod openmpt;
mod player;
mod remaster;
mod render;
mod simd;
mod upsample;

use std::path::PathBuf;
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex};

use clap::{ArgAction, Args, Parser, Subcommand, ValueEnum};
use gui::Quinlight;
use remaster::{CleanupEngineVersion, CleanupMode, CleanupSettings, UpscaleMode};

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
enum UpscaleModeArg {
    Cpu,
    Gpu,
    Hybrid,
}

impl From<UpscaleModeArg> for UpscaleMode {
    fn from(value: UpscaleModeArg) -> Self {
        match value {
            UpscaleModeArg::Cpu => UpscaleMode::CpuOnly,
            UpscaleModeArg::Gpu => UpscaleMode::GpuOnly,
            UpscaleModeArg::Hybrid => UpscaleMode::Hybrid,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, ValueEnum)]
enum CleanupModeArg {
    #[default]
    Off,
    #[value(name = "declick-ar")]
    DeclickAr,
    #[value(name = "declick-median")]
    DeclickMedian,
    #[value(name = "decrackle")]
    Decrackle,
}

impl From<CleanupModeArg> for CleanupMode {
    fn from(value: CleanupModeArg) -> Self {
        match value {
            CleanupModeArg::Off => CleanupMode::Off,
            CleanupModeArg::DeclickAr => CleanupMode::DeclickAr,
            CleanupModeArg::DeclickMedian => CleanupMode::DeclickMedian,
            CleanupModeArg::Decrackle => CleanupMode::Decrackle,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, ValueEnum)]
enum CleanupEngineArg {
    V1,
    #[default]
    #[value(name = "v2-1")]
    V21,
}

impl From<CleanupEngineArg> for CleanupEngineVersion {
    fn from(value: CleanupEngineArg) -> Self {
        match value {
            CleanupEngineArg::V1 => CleanupEngineVersion::V1,
            CleanupEngineArg::V21 => CleanupEngineVersion::V21,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
enum ConvertEngineArg {
    #[value(name = "audiosr")]
    AudioSr,
    #[value(name = "lavasr")]
    LavaSr,
    #[value(name = "flowhigh")]
    FlowHigh,
    #[value(name = "apbwe")]
    ApBwe,
}

impl ConvertEngineArg {
    fn engine_name(self) -> &'static str {
        match self {
            Self::AudioSr => "AudioSR",
            Self::LavaSr => "LavaSR",
            Self::FlowHigh => "FLowHigh",
            Self::ApBwe => "AP-BWE",
        }
    }
}

#[derive(Args, Clone, Copy, Debug, Eq, PartialEq)]
struct UpscaleCliArgs {
    /// Quinlight worker mode: cpu, gpu, or hybrid (default: auto-detect — gpu if a GPU is present, else cpu)
    #[arg(long, value_enum)]
    upscale_mode: Option<UpscaleModeArg>,

    /// Run all AI engines concurrently (memory-capped) instead of one at a time.
    /// Faster on hosts with ample RAM/VRAM, but can trigger PyTorch subprocess
    /// hangs on some systems. Default: engines run serially.
    #[arg(long)]
    full_parallel: bool,
}

fn default_mode_for_vendor(vendor: engine::GpuVendor) -> UpscaleMode {
    match vendor {
        engine::GpuVendor::Nvidia | engine::GpuVendor::Amd | engine::GpuVendor::Intel => {
            UpscaleMode::GpuOnly
        }
        engine::GpuVendor::None => UpscaleMode::CpuOnly,
    }
}

fn resolve_upscale_mode(args: UpscaleCliArgs) -> UpscaleMode {
    resolve_with_vendor(args, engine::detect_gpu())
}

fn resolve_with_vendor(args: UpscaleCliArgs, vendor: engine::GpuVendor) -> UpscaleMode {
    match args.upscale_mode {
        Some(explicit) => {
            let mode: UpscaleMode = explicit.into();
            let no_gpu = vendor == engine::GpuVendor::None;
            if mode == UpscaleMode::Hybrid && no_gpu {
                // Hybrid = 2 workers per engine (1 GPU + 1 CPU). With no GPU,
                // both workers fall through to CPU, doubling CPU contention
                // for zero benefit. Downgrade to CpuOnly so the user gets 1
                // CPU worker per engine, not 2.
                eprintln!(
                    "quinlight: --upscale-mode {explicit:?} requested but no GPU detected; \
                     dropping to --upscale-mode cpu to avoid redundant CPU workers."
                );
                return UpscaleMode::CpuOnly;
            }
            if mode == UpscaleMode::GpuOnly && no_gpu {
                // GpuOnly = 1 worker per engine, device resolves to "cpu" via
                // gpu_device_string() when vendor is None. Behaviorally
                // equivalent to CpuOnly but we keep the user's requested mode
                // name for visibility in logs.
                eprintln!(
                    "quinlight: --upscale-mode {explicit:?} requested but no GPU detected; workers will run on CPU."
                );
            }
            mode
        }
        None => default_mode_for_vendor(vendor),
    }
}

fn application_window_settings() -> iced::window::Settings {
    iced::window::Settings {
        size: gui::initial_window_size(),
        min_size: Some(gui::minimum_window_size()),
        icon: Some(gui::icon::create_icon()),
        platform_specific: iced::window::settings::PlatformSpecific {
            application_id: "quinlight-audio".into(),
            ..Default::default()
        },
        exit_on_close_request: false,
        ..Default::default()
    }
}

#[derive(Parser)]
#[command(
    name = "quinlight-audio",
    about = "Tracker music player and remastering tool presented by Kind Computers"
)]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,

    #[command(flatten)]
    upscale: UpscaleCliArgs,

    /// Install .desktop file and icon, then exit
    #[arg(long)]
    install_icon: bool,

    /// Force OpenGL backend (workaround for Wayland Vulkan issues)
    #[arg(long)]
    gl: bool,

    /// Disable sample cache reads and writes (for debugging)
    #[arg(long)]
    no_cache: bool,

    /// Force the GUI playback DAC sample rate in Hz. If unset, auto-negotiates
    /// from [96000, 88200, 48000, 44100]. Useful when 96 kHz crashes Bluetooth
    /// headphones — pass `--playback-rate 48000` to cap output at 48 kHz.
    #[arg(long, value_parser = clap::value_parser!(u32).range(8000..=384000))]
    playback_rate: Option<u32>,
}

#[derive(Subcommand)]
enum Commands {
    /// Batch render modules in a directory to FLAC/M4A (with Quinlight remastering by default)
    Convert {
        /// Input file or directory containing module files
        input: PathBuf,

        /// Output directory (default: same as input)
        #[arg(short, long)]
        output: Option<PathBuf>,

        /// Output format(s): flac, m4a, or both
        #[arg(long, default_value = "flac", num_args = 1..)]
        format: Vec<String>,

        /// Stereo separation percentage (0-200, default: 66)
        #[arg(long)]
        stereo_separation: Option<i32>,

        /// Enable OpenMPT AGC during rendering
        #[arg(long, action = ArgAction::SetTrue, conflicts_with = "no_agc")]
        agc: bool,

        /// Disable OpenMPT AGC during rendering
        #[arg(long = "no-agc", action = ArgAction::SetTrue)]
        no_agc: bool,

        /// Recurse into subdirectories
        #[arg(short, long)]
        recursive: bool,

        /// Skip AI remastering (just render originals)
        #[arg(long, conflicts_with = "engine")]
        no_remaster: bool,

        /// Restrict Quinlight remastering to the selected engine(s)
        #[arg(long, value_enum, conflicts_with = "no_remaster")]
        engine: Vec<ConvertEngineArg>,

        /// Reference cleanup preset to apply before AI remastering
        #[arg(long, value_enum, default_value_t = CleanupModeArg::Off)]
        cleanup_preset: CleanupModeArg,

        /// Cleanup engine version to use for reference preparation
        #[arg(long, value_enum, default_value_t = CleanupEngineArg::V21)]
        cleanup_engine: CleanupEngineArg,

        /// AudioSR DDIM steps (quality/speed tradeoff: 25=fast, 50=default, 100=max)
        #[arg(long, default_value = "50")]
        ddim_steps: u32,

        /// Override output sample rate in Hz (default: 96000, capped at 96000 for AAC)
        #[arg(long)]
        sample_rate: Option<u32>,

        /// Disable HRTF headphone spatialization in rendered output
        #[arg(long = "no-hrtf", action = ArgAction::SetTrue)]
        no_hrtf: bool,

        /// HRTF wet/dry mix percentage (0=dry, 100=full wet, default: 33)
        #[arg(long, default_value = "33")]
        hrtf_mix: i32,

        #[command(flatten)]
        upscale: UpscaleCliArgs,
    },
    /// AI-upsample one or more standalone FLAC files through all detected
    /// engines and write the Quinlight consensus result as 48 kHz / 32-bit
    /// FLAC. Passing many files in one invocation amortizes model-load cost
    /// — each engine's Python subprocess loads its model once per batch
    /// chunk instead of once per file.
    Upsample {
        /// Input FLAC file(s) (< 48 kHz each — engines decline already-48 kHz inputs).
        /// Pass multiple to batch them through a shared model load.
        #[arg(required = true, num_args = 1..)]
        inputs: Vec<PathBuf>,

        /// Output path. For a single input, this may be a file path or a directory.
        /// For multiple inputs, it must be a directory. Defaults to
        /// {input_stem}-Quinlight-Audio-Remastered-48Khz.flac next to each input.
        #[arg(short, long)]
        output: Option<PathBuf>,

        /// Restrict Quinlight remastering to the selected engine(s)
        #[arg(long, value_enum)]
        engine: Vec<ConvertEngineArg>,

        /// AudioSR DDIM steps (quality/speed tradeoff: 25=fast, 50=default, 100=max)
        #[arg(long, default_value = "50")]
        ddim_steps: u32,

        /// Minimum per-engine spectral-correlation score required for that
        /// engine's output to contribute to Quinlight's consensus. Engines
        /// below the floor are dropped for that sample; if fewer than 2
        /// engines remain usable, Quinlight keeps the original sample.
        /// Range 0.0 – 1.0. Defaults to the built-in floor (0.9).
        #[arg(long)]
        threshold: Option<f64>,

        #[command(flatten)]
        upscale: UpscaleCliArgs,
    },
    /// Render a module directly to FLAC or M4A audio
    Render {
        /// Input module file
        input: PathBuf,

        /// Output file path
        #[arg(short, long)]
        output: PathBuf,

        /// Output format: flac or m4a
        #[arg(long, default_value = "flac")]
        format: String,

        /// Stereo separation percentage (0-200, default: 66)
        #[arg(long)]
        stereo_separation: Option<i32>,

        /// Enable OpenMPT AGC during rendering
        #[arg(long, action = ArgAction::SetTrue, conflicts_with = "no_agc")]
        agc: bool,

        /// Disable OpenMPT AGC during rendering
        #[arg(long = "no-agc", action = ArgAction::SetTrue)]
        no_agc: bool,

        /// File within archive to process (when input is an archive)
        #[arg(long)]
        file: Option<String>,

        /// Output sample rate in Hz (default: 96000, capped at 96000 for AAC)
        #[arg(long)]
        sample_rate: Option<u32>,

        /// Disable HRTF headphone spatialization in rendered output
        #[arg(long = "no-hrtf", action = ArgAction::SetTrue)]
        no_hrtf: bool,

        /// HRTF wet/dry mix percentage (0=dry, 100=full wet, default: 33)
        #[arg(long, default_value = "33")]
        hrtf_mix: i32,
    },
    /// Print tracker-module metadata as JSON (title, artist, sample/instrument names)
    ProbeMetadata {
        /// Input module file
        input: PathBuf,
    },
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    if native_diagnostics::run_symbolizer_helper_from_env() {
        return Ok(());
    }

    let cli = Cli::parse();

    if cli.no_cache {
        remaster::set_no_cache(true);
    }

    if cli.gl || std::env::var("QUINLIGHT_AUDIO_GL_RETRY").as_deref() == Ok("1") {
        // SAFETY: called before any threads are spawned
        unsafe { std::env::set_var("WGPU_BACKEND", "gl") };
    }

    // Register Ctrl-C handler for graceful shutdown
    let shutdown_flag = Arc::new(AtomicBool::new(false));
    signal_hook::flag::register(signal_hook::consts::SIGINT, shutdown_flag.clone())?;

    if cli.install_icon {
        install_desktop_icon();
        return Ok(());
    }

    native_diagnostics::install_from_env();
    native_diagnostics::maybe_trigger_test_signal_from_env();

    match cli.command {
        Some(Commands::Convert {
            input,
            output,
            format,
            stereo_separation,
            agc,
            no_agc,
            recursive,
            no_remaster,
            engine,
            cleanup_preset,
            cleanup_engine,
            ddim_steps,
            sample_rate,
            no_hrtf,
            hrtf_mix,
            upscale,
        }) => {
            let mode = resolve_upscale_mode(upscale);
            let agc_enabled = agc || !no_agc;
            let hrtf_mix = if no_hrtf { 0 } else { hrtf_mix };
            let engine_names: Vec<String> = engine
                .into_iter()
                .map(|engine| engine.engine_name().to_string())
                .collect();
            match batch::run_convert(
                &input,
                output.as_deref(),
                &format,
                stereo_separation,
                agc_enabled,
                recursive,
                !no_remaster,
                &engine_names,
                mode,
                upscale.full_parallel,
                CleanupSettings::new(cleanup_preset.into(), cleanup_engine.into()),
                ddim_steps,
                sample_rate,
                &shutdown_flag,
                hrtf_mix,
            ) {
                Ok(()) => {}
                Err(e) => {
                    eprintln!("Error: {e}");
                    std::process::exit(match e {
                        batch::ConvertError::Fatal(_) => 1,
                        batch::ConvertError::QualityGate(_) => 2,
                    });
                }
            }
        }
        Some(Commands::Upsample {
            inputs,
            output,
            engine,
            ddim_steps,
            threshold,
            upscale,
        }) => {
            let mode = resolve_upscale_mode(upscale);
            let engine_names: Vec<String> = engine
                .into_iter()
                .map(|e| e.engine_name().to_string())
                .collect();
            if let Some(t) = threshold {
                if !(0.0..=1.0).contains(&t) {
                    eprintln!("Error: --threshold must be in [0.0, 1.0] (got {t})");
                    std::process::exit(2);
                }
                remaster::set_quinlight_usable_score_floor(t);
                eprintln!(
                    "Quinlight Audio: usable-score floor overridden to {t:.2} via --threshold"
                );
            }
            match upsample::run_upsample_batch(
                &inputs,
                output.as_deref(),
                mode,
                upscale.full_parallel,
                &engine_names,
                ddim_steps,
                &shutdown_flag,
            ) {
                Ok(()) => {}
                Err(e) => {
                    eprintln!("Error: {e}");
                    std::process::exit(match e {
                        upsample::UpsampleError::Fatal(_) => 1,
                        upsample::UpsampleError::QualityGate(_) => 2,
                    });
                }
            }
        }
        Some(Commands::Render {
            input,
            output,
            format,
            stereo_separation,
            agc,
            no_agc,
            file,
            sample_rate,
            no_hrtf,
            hrtf_mix,
        }) => {
            let agc_enabled = agc || !no_agc;
            let hrtf_mix = if no_hrtf { 0 } else { hrtf_mix };
            let stereo_separation = stereo_separation.unwrap_or_else(|| {
                openmpt::effective_stereo_separation(
                    &input,
                    openmpt::DEFAULT_STEREO_SEPARATION_PERCENT,
                )
            });
            let rate = sample_rate.unwrap_or(96_000);
            let rate = if format == "m4a" && rate > 96_000 {
                eprintln!("AAC max sample rate is 96 kHz; capping from {rate} Hz");
                96_000
            } else {
                rate
            };
            batch::run_render(
                &input,
                &output,
                &format,
                stereo_separation,
                agc_enabled,
                file.as_deref(),
                rate,
                hrtf_mix,
            )?;
        }
        Some(Commands::ProbeMetadata { input }) => {
            probe_metadata(&input)?;
        }
        None => {
            let upscale_mode = resolve_upscale_mode(cli.upscale);
            let playback_rate = cli.playback_rate;
            let panic_info_store: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));
            let store_clone = panic_info_store.clone();
            let default_hook = std::panic::take_hook();
            std::panic::set_hook(Box::new(move |info| {
                *store_clone.lock().unwrap() = Some(format!("{info}"));
            }));

            // Start engine detection early so it overlaps with Iced window creation
            let detect_handle = std::thread::spawn(remaster::RemasterEngine::detect);

            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                iced::application(Quinlight::title, Quinlight::update, Quinlight::view)
                    .antialiasing(true)
                    .subscription(Quinlight::subscription)
                    .theme(Quinlight::theme)
                    .window(application_window_settings())
                    .run_with(move || {
                        Quinlight::new(upscale_mode, detect_handle, shutdown_flag, playback_rate)
                    })
            }));

            std::panic::set_hook(default_hook);

            match result {
                Ok(Ok(())) => {}
                Ok(Err(e)) => return Err(e.into()),
                Err(panic_payload) => {
                    if is_wgpu_panic(&panic_payload) && can_retry_with_gl() {
                        eprintln!(
                            "quinlight: Vulkan backend crashed (Wayland DMA-BUF issue). \
                             Restarting with OpenGL backend..."
                        );
                        retry_with_gl_backend();
                    }
                    if let Some(info) = panic_info_store.lock().unwrap().take() {
                        eprintln!("{info}");
                    }
                    std::panic::resume_unwind(panic_payload);
                }
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::CommandFactory;

    #[test]
    fn help_text_mentions_upscale_mode_and_not_ensemble() {
        let help = Cli::command().render_long_help().to_string();
        assert!(help.contains("--upscale-mode"));
        assert!(!help.contains("--ensemble"));
        assert!(!help.contains("--gpu-upscale"));
        assert!(!help.contains("--hybrid-upscale"));
    }

    #[test]
    fn playback_rate_flag_parses() {
        let cli = Cli::try_parse_from(["quinlight-audio", "--playback-rate", "48000"])
            .expect("--playback-rate 48000 should parse");
        assert_eq!(cli.playback_rate, Some(48_000));
    }

    #[test]
    fn playback_rate_defaults_to_none() {
        let cli = Cli::try_parse_from(["quinlight-audio"]).expect("root command should parse");
        assert_eq!(cli.playback_rate, None);
    }

    #[test]
    fn playback_rate_flag_rejects_out_of_range() {
        assert!(Cli::try_parse_from(["quinlight-audio", "--playback-rate", "0"]).is_err());
        assert!(Cli::try_parse_from(["quinlight-audio", "--playback-rate", "500000"]).is_err());
    }

    #[test]
    fn remaster_subcommand_is_removed() {
        let subcommands: Vec<_> = Cli::command()
            .get_subcommands()
            .map(|subcommand| subcommand.get_name().to_string())
            .collect();
        assert!(!subcommands.iter().any(|name| name == "remaster"));
    }

    #[test]
    fn removed_remaster_subcommand_is_rejected() {
        let parsed = Cli::try_parse_from(["quinlight-audio", "remaster", "song.it"]);
        assert!(parsed.is_err());
    }

    #[test]
    fn convert_rejects_ensemble_flag() {
        let parsed = Cli::try_parse_from(["quinlight-audio", "convert", "mods", "--ensemble"]);
        assert!(parsed.is_err());
    }

    #[test]
    fn convert_help_mentions_upscale_mode_and_hides_legacy_flags() {
        let mut cmd = Cli::command();
        let help = cmd
            .find_subcommand_mut("convert")
            .expect("convert subcommand should exist")
            .render_long_help()
            .to_string();
        assert!(help.contains("--upscale-mode"));
        assert!(help.contains("--cleanup-preset"));
        assert!(help.contains("--engine"));
        assert!(!help.contains("--reference-only"));
        assert!(!help.contains("--gpu"));
        assert!(!help.contains("--hybrid"));
    }

    #[test]
    fn cli_defaults_stereo_separation_and_upscale_unset() {
        let cli = Cli::try_parse_from(["quinlight-audio"]).expect("root command should parse");
        assert_eq!(cli.upscale.upscale_mode, None);
        assert_eq!(
            batch::default_render_interpolation_filter(),
            openmpt::DEFAULT_INTERPOLATION_FILTER_LENGTH,
            "Render and convert should share the OpenMPT interpolation default",
        );

        let convert = Cli::try_parse_from(["quinlight-audio", "convert", "mods"])
            .expect("convert command should parse");
        match convert.command {
            Some(Commands::Convert {
                stereo_separation,
                cleanup_preset,
                cleanup_engine,
                upscale,
                ..
            }) => {
                assert_eq!(stereo_separation, None);
                assert_eq!(cleanup_preset, CleanupModeArg::Off);
                assert_eq!(cleanup_engine, CleanupEngineArg::V21);
                assert_eq!(upscale.upscale_mode, None);
            }
            _ => panic!("expected convert command"),
        }

        let render =
            Cli::try_parse_from(["quinlight-audio", "render", "song.it", "-o", "song.flac"])
                .expect("render command should parse");
        match render.command {
            Some(Commands::Render {
                stereo_separation,
                sample_rate,
                ..
            }) => {
                assert_eq!(stereo_separation, None);
                assert_eq!(sample_rate, None);
            }
            _ => panic!("expected render command"),
        }
    }

    #[test]
    fn default_mode_for_vendor_maps_correctly() {
        assert_eq!(
            default_mode_for_vendor(engine::GpuVendor::Nvidia),
            UpscaleMode::GpuOnly,
        );
        assert_eq!(
            default_mode_for_vendor(engine::GpuVendor::Amd),
            UpscaleMode::GpuOnly,
        );
        assert_eq!(
            default_mode_for_vendor(engine::GpuVendor::Intel),
            UpscaleMode::GpuOnly,
        );
        assert_eq!(
            default_mode_for_vendor(engine::GpuVendor::None),
            UpscaleMode::CpuOnly,
        );
    }

    #[test]
    fn upscale_mode_parses_for_gui_launch() {
        let gpu = Cli::try_parse_from(["quinlight-audio", "--upscale-mode", "gpu"])
            .expect("root gpu mode should parse");
        assert_eq!(
            resolve_with_vendor(gpu.upscale, engine::GpuVendor::Nvidia),
            UpscaleMode::GpuOnly,
        );

        let hybrid = Cli::try_parse_from(["quinlight-audio", "--upscale-mode", "hybrid"])
            .expect("root hybrid mode should parse");
        assert_eq!(
            resolve_with_vendor(hybrid.upscale, engine::GpuVendor::Nvidia),
            UpscaleMode::Hybrid,
        );
    }

    #[test]
    fn upscale_mode_parses_for_convert() {
        let gpu = Cli::try_parse_from([
            "quinlight-audio",
            "convert",
            "mods",
            "--upscale-mode",
            "gpu",
        ])
        .expect("convert gpu mode should parse");
        match gpu.command {
            Some(Commands::Convert { upscale, .. }) => {
                assert_eq!(
                    resolve_with_vendor(upscale, engine::GpuVendor::Nvidia),
                    UpscaleMode::GpuOnly,
                );
            }
            _ => panic!("expected convert command"),
        }

        let hybrid = Cli::try_parse_from([
            "quinlight-audio",
            "convert",
            "mods",
            "--upscale-mode",
            "hybrid",
        ])
        .expect("convert hybrid mode should parse");
        match hybrid.command {
            Some(Commands::Convert { upscale, .. }) => {
                assert_eq!(
                    resolve_with_vendor(upscale, engine::GpuVendor::Nvidia),
                    UpscaleMode::Hybrid,
                );
            }
            _ => panic!("expected convert command"),
        }
    }

    #[test]
    fn resolve_with_vendor_downgrades_hybrid_when_no_gpu() {
        let args = UpscaleCliArgs {
            upscale_mode: Some(UpscaleModeArg::Hybrid),
            full_parallel: false,
        };
        assert_eq!(
            resolve_with_vendor(args, engine::GpuVendor::None),
            UpscaleMode::CpuOnly,
        );
    }

    #[test]
    fn resolve_with_vendor_keeps_hybrid_for_each_gpu_vendor() {
        let args = UpscaleCliArgs {
            upscale_mode: Some(UpscaleModeArg::Hybrid),
            full_parallel: false,
        };
        for vendor in [
            engine::GpuVendor::Nvidia,
            engine::GpuVendor::Amd,
            engine::GpuVendor::Intel,
        ] {
            assert_eq!(resolve_with_vendor(args, vendor), UpscaleMode::Hybrid);
        }
    }

    #[test]
    fn resolve_with_vendor_keeps_gpu_only_when_no_gpu() {
        // GpuOnly uses 1 worker per engine; gpu_device_string() falls through
        // to "cpu" when vendor is None, so the mode is behaviorally harmless.
        // We warn but don't rewrite it.
        let args = UpscaleCliArgs {
            upscale_mode: Some(UpscaleModeArg::Gpu),
            full_parallel: false,
        };
        assert_eq!(
            resolve_with_vendor(args, engine::GpuVendor::None),
            UpscaleMode::GpuOnly,
        );
    }

    #[test]
    fn resolve_with_vendor_none_args_falls_back_to_default() {
        let args = UpscaleCliArgs {
            upscale_mode: None,
            full_parallel: false,
        };
        assert_eq!(
            resolve_with_vendor(args, engine::GpuVendor::None),
            UpscaleMode::CpuOnly,
        );
        assert_eq!(
            resolve_with_vendor(args, engine::GpuVendor::Nvidia),
            UpscaleMode::GpuOnly,
        );
    }

    #[test]
    fn application_window_settings_disable_auto_exit_on_close_request() {
        let settings = application_window_settings();

        assert!(!settings.exit_on_close_request);
        assert_eq!(settings.size, gui::initial_window_size());
        assert_eq!(settings.min_size, Some(gui::minimum_window_size()));
    }

    #[test]
    fn cleanup_preset_parses_for_convert() {
        let off = Cli::try_parse_from([
            "quinlight-audio",
            "convert",
            "mods",
            "--cleanup-preset",
            "off",
        ])
        .expect("convert off cleanup should parse");
        match off.command {
            Some(Commands::Convert { cleanup_preset, .. }) => {
                assert_eq!(cleanup_preset, CleanupModeArg::Off);
            }
            _ => panic!("expected convert command"),
        }

        let click_ar = Cli::try_parse_from([
            "quinlight-audio",
            "convert",
            "mods",
            "--cleanup-preset",
            "declick-ar",
        ])
        .expect("convert click ar cleanup should parse");
        match click_ar.command {
            Some(Commands::Convert { cleanup_preset, .. }) => {
                assert_eq!(cleanup_preset, CleanupModeArg::DeclickAr);
            }
            _ => panic!("expected convert command"),
        }

        let click_median = Cli::try_parse_from([
            "quinlight-audio",
            "convert",
            "mods",
            "--cleanup-preset",
            "declick-median",
        ])
        .expect("convert click median cleanup should parse");
        match click_median.command {
            Some(Commands::Convert { cleanup_preset, .. }) => {
                assert_eq!(cleanup_preset, CleanupModeArg::DeclickMedian);
            }
            _ => panic!("expected convert command"),
        }

        let crackle = Cli::try_parse_from([
            "quinlight-audio",
            "convert",
            "mods",
            "--cleanup-preset",
            "decrackle",
        ])
        .expect("convert crackle cleanup should parse");
        match crackle.command {
            Some(Commands::Convert { cleanup_preset, .. }) => {
                assert_eq!(cleanup_preset, CleanupModeArg::Decrackle);
            }
            _ => panic!("expected convert command"),
        }
    }

    #[test]
    fn cleanup_engine_parses_for_convert() {
        let v1 = Cli::try_parse_from([
            "quinlight-audio",
            "convert",
            "mods",
            "--cleanup-engine",
            "v1",
        ])
        .expect("convert v1 cleanup engine should parse");
        match v1.command {
            Some(Commands::Convert { cleanup_engine, .. }) => {
                assert_eq!(cleanup_engine, CleanupEngineArg::V1);
            }
            _ => panic!("expected convert command"),
        }

        let v21 = Cli::try_parse_from([
            "quinlight-audio",
            "convert",
            "mods",
            "--cleanup-engine",
            "v2-1",
        ])
        .expect("convert v2.1 cleanup engine should parse");
        match v21.command {
            Some(Commands::Convert { cleanup_engine, .. }) => {
                assert_eq!(cleanup_engine, CleanupEngineArg::V21);
            }
            _ => panic!("expected convert command"),
        }
    }

    #[test]
    fn convert_engine_selection_parses_repeatably() {
        let parsed = Cli::try_parse_from([
            "quinlight-audio",
            "convert",
            "mods",
            "--engine",
            "audiosr",
            "--engine",
            "lavasr",
            "--engine",
            "apbwe",
        ])
        .expect("repeatable engine selection should parse");

        match parsed.command {
            Some(Commands::Convert {
                engine,
                no_remaster,
                ..
            }) => {
                assert_eq!(
                    engine,
                    vec![
                        ConvertEngineArg::AudioSr,
                        ConvertEngineArg::LavaSr,
                        ConvertEngineArg::ApBwe,
                    ]
                );
                assert!(!no_remaster);
            }
            _ => panic!("expected convert command"),
        }
    }

    #[test]
    fn convert_engine_apbwe_maps_to_canonical_name() {
        assert_eq!(ConvertEngineArg::ApBwe.engine_name(), "AP-BWE");
    }

    #[test]
    fn convert_reference_only_is_rejected() {
        let parsed =
            Cli::try_parse_from(["quinlight-audio", "convert", "mods", "--reference-only"]);
        assert!(parsed.is_err());
    }

    #[test]
    fn convert_engine_still_conflicts_with_no_remaster() {
        let engine_with_no_remaster = Cli::try_parse_from([
            "quinlight-audio",
            "convert",
            "mods",
            "--engine",
            "audiosr",
            "--no-remaster",
        ]);
        assert!(engine_with_no_remaster.is_err());
    }

    #[test]
    fn retired_cleanup_presets_are_rejected() {
        let light = Cli::try_parse_from([
            "quinlight-audio",
            "convert",
            "mods",
            "--cleanup-preset",
            "light",
        ]);
        assert!(light.is_err());

        let archival = Cli::try_parse_from([
            "quinlight-audio",
            "convert",
            "mods",
            "--cleanup-preset",
            "archival",
        ]);
        assert!(archival.is_err());
    }

    #[test]
    fn legacy_cleanup_preset_names_are_rejected() {
        let click_ar = Cli::try_parse_from([
            "quinlight-audio",
            "convert",
            "mods",
            "--cleanup-preset",
            "click::remove_click_ar",
        ]);
        assert!(click_ar.is_err());

        let click_median = Cli::try_parse_from([
            "quinlight-audio",
            "convert",
            "mods",
            "--cleanup-preset",
            "click::remove_click_median",
        ]);
        assert!(click_median.is_err());

        let crackle = Cli::try_parse_from([
            "quinlight-audio",
            "convert",
            "mods",
            "--cleanup-preset",
            "crackle",
        ]);
        assert!(crackle.is_err());
    }

    #[test]
    fn invalid_upscale_mode_is_rejected() {
        let root = Cli::try_parse_from(["quinlight-audio", "--upscale-mode", "banana"]);
        assert!(root.is_err());

        let convert = Cli::try_parse_from([
            "quinlight-audio",
            "convert",
            "mods",
            "--upscale-mode",
            "banana",
        ]);
        assert!(convert.is_err());
    }

    #[test]
    fn render_rejects_upscale_mode() {
        let parsed = Cli::try_parse_from([
            "quinlight-audio",
            "render",
            "song.it",
            "-o",
            "song.flac",
            "--upscale-mode",
            "gpu",
        ]);
        assert!(parsed.is_err());
    }

    #[test]
    fn render_rejects_cleanup_preset() {
        let parsed = Cli::try_parse_from([
            "quinlight-audio",
            "render",
            "song.it",
            "-o",
            "song.flac",
            "--cleanup-preset",
            "off",
        ]);
        assert!(parsed.is_err());
    }

    #[test]
    fn render_rejects_cleanup_engine() {
        let parsed = Cli::try_parse_from([
            "quinlight-audio",
            "render",
            "song.it",
            "-o",
            "song.flac",
            "--cleanup-engine",
            "v1",
        ]);
        assert!(parsed.is_err());
    }

    #[test]
    fn legacy_upscale_flags_are_rejected() {
        assert!(Cli::try_parse_from(["quinlight-audio", "--gpu-upscale"]).is_err());
        assert!(Cli::try_parse_from(["quinlight-audio", "--hybrid-upscale"]).is_err());
        assert!(Cli::try_parse_from(["quinlight-audio", "convert", "mods", "--gpu"]).is_err());
        assert!(Cli::try_parse_from(["quinlight-audio", "convert", "mods", "--hybrid"]).is_err());
    }
}

fn probe_metadata(path: &std::path::Path) -> Result<(), Box<dyn std::error::Error>> {
    let data = std::fs::read(path)?;
    let module = openmpt::Module::from_memory(&data)
        .map_err(|e| format!("libopenmpt failed to load {}: {e}", path.display()))?;

    let meta = module.metadata();
    let num_samples = module.num_samples();
    let num_instruments = module.num_instruments();

    let collect_names = |count: i32, get: &dyn Fn(i32) -> String| -> Vec<String> {
        (1..=count)
            .map(get)
            .filter(|s| !s.trim().is_empty())
            .map(|s| s.trim_end().to_string())
            .collect()
    };
    let sample_names = collect_names(num_samples, &|i| module.sample_name(i));
    let instrument_names = collect_names(num_instruments, &|i| module.instrument_name(i));

    let out = serde_json::json!({
        "title": meta.title.trim(),
        "artist": meta.artist.trim(),
        "tracker": meta.tracker.trim(),
        "type_long": meta.type_long.trim(),
        "date": meta.date.trim(),
        "message": meta.message.trim(),
        "sample_names": sample_names,
        "instrument_names": instrument_names,
    });
    println!("{out}");
    Ok(())
}

fn is_wgpu_panic(payload: &Box<dyn std::any::Any + Send>) -> bool {
    let msg = if let Some(s) = payload.downcast_ref::<&str>() {
        *s
    } else if let Some(s) = payload.downcast_ref::<String>() {
        s.as_str()
    } else {
        return false;
    };
    msg.contains("Fallback system failed to choose present mode") || msg.contains("present_modes")
}

fn can_retry_with_gl() -> bool {
    std::env::var("QUINLIGHT_AUDIO_GL_RETRY").as_deref() != Ok("1")
}

fn retry_with_gl_backend() -> ! {
    use std::os::unix::process::CommandExt;
    let exe = std::env::current_exe().expect("Failed to get current executable path");
    let args: Vec<String> = std::env::args().skip(1).collect();
    let err = std::process::Command::new(exe)
        .args(&args)
        .env("QUINLIGHT_AUDIO_GL_RETRY", "1")
        .exec();
    panic!("Failed to re-exec with GL backend: {err}");
}

fn install_desktop_icon() {
    let Some(data_dir) = dirs::data_dir() else {
        eprintln!("Warning: Could not determine XDG data directory, skipping icon install");
        return;
    };

    let icon_dir = data_dir.join("icons/hicolor/256x256/apps");
    if let Err(e) = std::fs::create_dir_all(&icon_dir) {
        eprintln!("Warning: Failed to create icon directory: {e}");
        return;
    }
    let icon_path = icon_dir.join("quinlight-audio.png");
    if let Err(e) = gui::icon::save_icon_png(&icon_path) {
        eprintln!("Warning: Failed to save icon: {e}");
    } else {
        eprintln!("Installed icon: {}", icon_path.display());
    }

    let apps_dir = data_dir.join("applications");
    if let Err(e) = std::fs::create_dir_all(&apps_dir) {
        eprintln!("Warning: Failed to create applications directory: {e}");
        return;
    }
    let desktop_path = apps_dir.join("quinlight-audio.desktop");

    let exe = std::env::current_exe()
        .map(|p| p.display().to_string())
        .unwrap_or_else(|_| "quinlight-audio".into());

    match std::fs::write(
        &desktop_path,
        format!(
            "[Desktop Entry]\n\
             Name=Quinlight Audio\n\
             Comment=Tracker music player and remastering tool presented by Kind Computers\n\
             Exec={exe}\n\
             Icon=quinlight-audio\n\
             Type=Application\n\
             Categories=Audio;AudioVideo;\n\
             StartupWMClass=quinlight-audio\n"
        ),
    ) {
        Ok(()) => eprintln!("Installed desktop entry: {}", desktop_path.display()),
        Err(e) => eprintln!("Warning: Failed to write .desktop file: {e}"),
    }
}
