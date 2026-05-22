// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Kind Computers, LLC.

use std::fmt;

mod shared;
mod v1;
mod v21;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub enum CleanupMode {
    #[default]
    Off,
    DeclickAr,
    DeclickMedian,
    Decrackle,
}

impl CleanupMode {
    pub const ALL: [Self; 4] = [
        Self::Off,
        Self::DeclickAr,
        Self::DeclickMedian,
        Self::Decrackle,
    ];
}

impl fmt::Display for CleanupMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Off => write!(f, "Off"),
            Self::DeclickAr => write!(f, "Declick (AR)"),
            Self::DeclickMedian => write!(f, "Declick (Median)"),
            Self::Decrackle => write!(f, "Decrackle"),
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub enum CleanupEngineVersion {
    V1,
    #[default]
    V21,
}

impl CleanupEngineVersion {
    pub const ALL: [Self; 2] = [Self::V1, Self::V21];
}

impl fmt::Display for CleanupEngineVersion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::V1 => write!(f, "Quinlight Audio V1"),
            Self::V21 => write!(f, "Quinlight Audio V2.1"),
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct CleanupSettings {
    pub mode: CleanupMode,
    pub engine_version: CleanupEngineVersion,
}

impl CleanupSettings {
    pub const fn new(mode: CleanupMode, engine_version: CleanupEngineVersion) -> Self {
        Self {
            mode,
            engine_version,
        }
    }

    pub const fn off() -> Self {
        Self::new(CleanupMode::Off, CleanupEngineVersion::V21)
    }

    /// Every active (non-Off) mode crossed with every engine version.
    pub const ALL_ACTIVE: [Self; 6] = [
        Self::new(CleanupMode::DeclickAr, CleanupEngineVersion::V1),
        Self::new(CleanupMode::DeclickMedian, CleanupEngineVersion::V1),
        Self::new(CleanupMode::Decrackle, CleanupEngineVersion::V1),
        Self::new(CleanupMode::DeclickAr, CleanupEngineVersion::V21),
        Self::new(CleanupMode::DeclickMedian, CleanupEngineVersion::V21),
        Self::new(CleanupMode::Decrackle, CleanupEngineVersion::V21),
    ];

    pub(crate) fn hash_tag(self) -> u8 {
        match self.mode {
            CleanupMode::Off => 0,
            CleanupMode::DeclickAr => match self.engine_version {
                CleanupEngineVersion::V1 => 1,
                CleanupEngineVersion::V21 => 4,
            },
            CleanupMode::DeclickMedian => match self.engine_version {
                CleanupEngineVersion::V1 => 2,
                CleanupEngineVersion::V21 => 5,
            },
            CleanupMode::Decrackle => match self.engine_version {
                CleanupEngineVersion::V1 => 3,
                CleanupEngineVersion::V21 => 6,
            },
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RetiredCleanupPreset {
    Light,
    Archival,
}

impl RetiredCleanupPreset {
    pub(crate) fn hash_tag(self) -> u8 {
        match self {
            Self::Light => 1,
            Self::Archival => 2,
        }
    }
}

pub fn apply_cleanup(
    data: &[f64],
    sample_rate: u32,
    channels: usize,
    settings: CleanupSettings,
) -> Result<Vec<f64>, String> {
    if channels == 0 || data.is_empty() {
        return Ok(Vec::new());
    }
    if settings.mode == CleanupMode::Off {
        return Ok(data.to_vec());
    }
    match settings.engine_version {
        CleanupEngineVersion::V1 => apply_cleanup_v1(data, sample_rate, channels, settings.mode),
        CleanupEngineVersion::V21 => apply_cleanup_v21(data, sample_rate, channels, settings.mode),
    }
}

pub(crate) fn apply_retired_cleanup_preset(
    data: &[f64],
    sample_rate: u32,
    channels: usize,
    preset: RetiredCleanupPreset,
) -> Result<Vec<f64>, String> {
    apply_retired_cleanup_preset_v1(data, sample_rate, channels, preset)
}

#[allow(dead_code)]
pub(crate) fn apply_cleanup_v1(
    data: &[f64],
    sample_rate: u32,
    channels: usize,
    mode: CleanupMode,
) -> Result<Vec<f64>, String> {
    v1::apply_cleanup_v1(data, sample_rate, channels, mode)
}

#[allow(dead_code)]
pub(crate) fn apply_cleanup_v21(
    data: &[f64],
    sample_rate: u32,
    channels: usize,
    mode: CleanupMode,
) -> Result<Vec<f64>, String> {
    v21::apply_cleanup_v21(data, sample_rate, channels, mode)
}

fn apply_retired_cleanup_preset_v1(
    data: &[f64],
    sample_rate: u32,
    channels: usize,
    preset: RetiredCleanupPreset,
) -> Result<Vec<f64>, String> {
    v1::apply_retired_cleanup_preset_v1(data, sample_rate, channels, preset)
}
