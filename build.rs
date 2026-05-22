// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Kind Computers, LLC.

use std::collections::hash_map::DefaultHasher;
use std::env;
use std::fs;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::process::Command;

const DIAGNOSTICS_ENV: &str = "QUINLIGHT_AUDIO_NATIVE_DIAGNOSTICS";
const BUILD_SIGNATURE_FILE: &str = ".openmpt-native-build-signature";
const OPENMPT_NATIVEFLOAT_CPPFLAGS: &str =
    "-DMPT_COMPILER_QUIRK_FLOAT_PREFER64=1 -DMPT_COMPILER_QUIRK_FLOAT_PREFER32=0";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum NativeDiagnosticsMode {
    Off,
    Symbols,
    Asan,
    Ubsan,
}

impl NativeDiagnosticsMode {
    fn from_env() -> Self {
        let raw = env::var(DIAGNOSTICS_ENV).unwrap_or_else(|_| "off".to_string());
        match raw.trim().to_ascii_lowercase().as_str() {
            "" | "off" => Self::Off,
            "symbols" => Self::Symbols,
            "asan" => Self::Asan,
            "ubsan" => Self::Ubsan,
            other => panic!(
                "invalid {DIAGNOSTICS_ENV} value '{other}'; expected off, symbols, asan, or ubsan"
            ),
        }
    }

    fn make_args(self) -> &'static [&'static str] {
        match self {
            Self::Off => &[],
            Self::Symbols => &["OPTIMIZE=debug", "CHECKED=1"],
            Self::Asan => &[
                "OPTIMIZE=debug",
                "CHECKED=1",
                "CHECKED_ADDRESS=1",
                "CC=clang",
                "CXX=clang++",
            ],
            Self::Ubsan => &[
                "OPTIMIZE=debug",
                "CHECKED=1",
                "CHECKED_UNDEFINED=1",
                "CC=clang",
                "CXX=clang++",
            ],
        }
    }

    fn linker_args(self, target_os: &str, target_env: &str) -> &'static [&'static str] {
        match self {
            Self::Off => &[],
            Self::Symbols => {
                if target_os == "linux" && target_env == "gnu" {
                    &["-Wl,--export-dynamic"]
                } else {
                    &[]
                }
            }
            Self::Asan => {
                if target_os == "linux" && target_env == "gnu" {
                    &["-fsanitize=address", "-Wl,--export-dynamic"]
                } else {
                    &["-fsanitize=address"]
                }
            }
            Self::Ubsan => {
                if target_os == "linux" && target_env == "gnu" {
                    &["-fsanitize=undefined", "-Wl,--export-dynamic"]
                } else {
                    &["-fsanitize=undefined"]
                }
            }
        }
    }

    fn build_signature(self, openmpt_dir: &Path) -> String {
        let cc = match self {
            Self::Asan | Self::Ubsan => "clang".to_string(),
            _ => env::var("CC").unwrap_or_default(),
        };
        let cxx = match self {
            Self::Asan | Self::Ubsan => "clang++".to_string(),
            _ => env::var("CXX").unwrap_or_default(),
        };
        let cc_version = compiler_version(if cc.is_empty() { "cc" } else { &cc });
        let header_hash = hash_header_files(openmpt_dir);
        format!(
            "mode={self:?};cc={cc};cxx={cxx};cc_ver={cc_version};headers={header_hash:016x};nativefloat={OPENMPT_NATIVEFLOAT_CPPFLAGS}"
        )
    }
}

fn clang_runtime_arch(target_arch: &str) -> &str {
    match target_arch {
        "x86_64" => "x86_64",
        "aarch64" => "aarch64",
        "arm" => "armhf",
        "i686" => "i386",
        other => panic!("unsupported sanitizer runtime target arch '{other}'"),
    }
}

fn clang_runtime_path(library_name: &str) -> String {
    let output = Command::new("clang")
        .arg(format!("-print-file-name={library_name}"))
        .output()
        .unwrap_or_else(|err| {
            panic!("failed to query clang runtime path for {library_name}: {err}")
        });
    if !output.status.success() {
        panic!("clang failed to locate runtime library {library_name}");
    }
    let path = String::from_utf8(output.stdout)
        .unwrap_or_else(|err| panic!("clang returned invalid utf-8 for {library_name}: {err}"));
    let path = path.trim();
    if path.is_empty() || path == library_name {
        panic!("clang runtime library {library_name} was not found");
    }
    path.to_string()
}

/// Query the compiler's version string for build-signature invalidation.
///
/// When the system compiler is upgraded (e.g. GCC 13 → 15), the version
/// changes, the signature mismatches, and `make clean` runs automatically —
/// preventing stale `.d`/`.o` files from referencing old include paths.
fn compiler_version(cc: &str) -> String {
    Command::new(cc)
        .arg("--version")
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .and_then(|s| s.lines().next().map(str::to_string))
        .unwrap_or_default()
}

/// Hash the contents of all C/C++ header files under `dir`.
///
/// When any header changes (struct layout, constants, enums), the hash changes,
/// which invalidates the build signature and triggers `make clean`.  This prevents
/// ODR violations from stale object files compiled against old header versions.
fn hash_header_files(dir: &Path) -> u64 {
    let mut paths = Vec::new();
    collect_headers(dir, &mut paths);
    paths.sort();
    let mut hasher = DefaultHasher::new();
    for path in &paths {
        path.hash(&mut hasher);
        if let Ok(contents) = fs::read(path) {
            contents.hash(&mut hasher);
        }
    }
    hasher.finish()
}

fn collect_headers(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            if path
                .file_name()
                .is_some_and(|n| n == "bin" || n == "include")
            {
                continue;
            }
            collect_headers(&path, out);
        } else if let Some(ext) = path.extension().and_then(|e| e.to_str())
            && matches!(ext, "h" | "hpp")
        {
            out.push(path);
        }
    }
}

fn watch_dir(dir: &Path) {
    if let Ok(entries) = fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                if path.file_name().is_some_and(|n| n == "bin") {
                    continue;
                }
                watch_dir(&path);
            } else if let Some(ext) = path.extension().and_then(|e| e.to_str())
                && matches!(ext, "cpp" | "c" | "h" | "hpp")
            {
                println!("cargo:rerun-if-changed={}", path.display());
            }
        }
    }
}

fn make_command(openmpt_dir: &Path, nproc: &str) -> Command {
    let mut command = Command::new("make");
    command.current_dir(openmpt_dir).arg(format!("-j{nproc}"));
    let cppflags = env::var("CPPFLAGS").unwrap_or_default();
    let cppflags = if cppflags.trim().is_empty() {
        OPENMPT_NATIVEFLOAT_CPPFLAGS.to_string()
    } else {
        format!("{cppflags} {OPENMPT_NATIVEFLOAT_CPPFLAGS}")
    };
    command.env("CPPFLAGS", cppflags);
    command
}

fn run_make_clean(openmpt_dir: &Path, nproc: &str) {
    let status = make_command(openmpt_dir, nproc)
        .arg("clean")
        .status()
        .expect("Failed to run make clean for libopenmpt");
    if !status.success() {
        panic!("libopenmpt clean failed");
    }
}

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let openmpt_dir = manifest_dir.join("openmpt");
    let target_dir = manifest_dir.join("target");
    let signature_path = target_dir.join(BUILD_SIGNATURE_FILE);
    let diagnostics_mode = NativeDiagnosticsMode::from_env();
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let target_env = env::var("CARGO_CFG_TARGET_ENV").unwrap_or_default();
    let target_arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();

    watch_dir(&openmpt_dir);

    for env_var in [
        DIAGNOSTICS_ENV,
        "CC",
        "CXX",
        "CPPFLAGS",
        "ASAN_OPTIONS",
        "UBSAN_OPTIONS",
        "ASAN_SYMBOLIZER_PATH",
        "LLVM_SYMBOLIZER_PATH",
    ] {
        println!("cargo:rerun-if-env-changed={env_var}");
    }

    let nproc = std::thread::available_parallelism()
        .map(|n| n.get().to_string())
        .unwrap_or_else(|_| "4".to_string());

    let signature = diagnostics_mode.build_signature(&openmpt_dir);
    let previous_signature = fs::read_to_string(&signature_path).ok();
    let library_path = openmpt_dir.join("bin").join("libopenmpt.a");
    let needs_clean = library_path.exists()
        && previous_signature
            .as_deref()
            .map(|previous| previous.trim() != signature)
            .unwrap_or(true);
    if needs_clean {
        run_make_clean(&openmpt_dir, &nproc);
    }

    let mut build = make_command(&openmpt_dir, &nproc);
    build
        .arg("STATIC_LIB=1")
        .arg("SHARED_LIB=0")
        .arg("OPENMPT123=0")
        .arg("EXAMPLES=0")
        .arg("TEST=0")
        .arg("NO_ZLIB=1")
        .arg("NO_MPG123=1")
        .arg("NO_OGG=1")
        .arg("NO_VORBIS=1")
        .arg("NO_VORBISFILE=1")
        .arg("bin/libopenmpt.a");
    for arg in diagnostics_mode.make_args() {
        build.arg(arg);
    }

    let status = build.status().expect("Failed to run make for libopenmpt");
    if !status.success() {
        panic!("libopenmpt build failed");
    }

    fs::create_dir_all(&target_dir).expect("Failed to create target directory");
    fs::write(&signature_path, format!("{signature}\n"))
        .expect("Failed to write OpenMPT build signature");

    println!(
        "cargo:rustc-link-search=native={}",
        openmpt_dir.join("bin").display()
    );
    println!("cargo:rustc-link-lib=static=openmpt");
    println!("cargo:rustc-link-lib=dylib=stdc++");
    for arg in diagnostics_mode.linker_args(&target_os, &target_env) {
        println!("cargo:rustc-link-arg={arg}");
    }
    if diagnostics_mode == NativeDiagnosticsMode::Ubsan
        && target_os == "linux"
        && target_env == "gnu"
    {
        let clang_arch = clang_runtime_arch(&target_arch);
        let runtime_cxx =
            clang_runtime_path(&format!("libclang_rt.ubsan_standalone_cxx-{clang_arch}.a"));
        println!("cargo:rustc-link-arg={runtime_cxx}");
    }
}
