// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Kind Computers, LLC.

use iced::Theme;
use iced::color;

// ── Designer palette ──────────────────────────────────────────────
// Five-stop ramp shared between panel chrome and the spectrogram heatmap:
//   PANEL_BG → PANEL_SURFACE → PANEL_LABEL → ACCENT_GREEN → ACCENT_AMBER
// (silence → mid-low → mid → loud → clipping/danger)
pub const PANEL_BG: iced::Color = color!(0x2b, 0x2b, 0x40);
pub const PANEL_SURFACE: iced::Color = color!(0x40, 0x40, 0x6b);
pub const PANEL_LABEL: iced::Color = color!(0x5c, 0x5c, 0x8a);
pub const CONTENT_TEXT: iced::Color = color!(0xe0, 0xe0, 0xff);
pub const ACCENT_GREEN: iced::Color = color!(0x00, 0xe8, 0x9c);
pub const ACCENT_AMBER: iced::Color = color!(0xff, 0xd3, 0x3c);

pub fn quinlight_theme() -> Theme {
    Theme::custom(
        "Quinlight Audio".into(),
        iced::theme::Palette {
            background: PANEL_BG,
            text: CONTENT_TEXT,
            primary: ACCENT_AMBER,
            success: ACCENT_GREEN,
            danger: ACCENT_AMBER,
        },
    )
}

/// VU meter color: panel gray (quiet) → green (loud) → amber (clipping).
/// Mirrors the spectrogram ramp so the two visualizations read as one system.
pub fn vu_color(value: f32) -> iced::Color {
    let v = value.clamp(0.0, 1.0);
    if v < 0.6 {
        // PANEL_LABEL → ACCENT_GREEN
        let t = v / 0.6;
        let from = PANEL_LABEL;
        let to = ACCENT_GREEN;
        iced::Color::from_rgba(
            from.r + (to.r - from.r) * t,
            from.g + (to.g - from.g) * t,
            from.b + (to.b - from.b) * t,
            0.85,
        )
    } else {
        // ACCENT_GREEN → ACCENT_AMBER
        let t = (v - 0.6) / 0.4;
        let from = ACCENT_GREEN;
        let to = ACCENT_AMBER;
        iced::Color::from_rgba(
            from.r + (to.r - from.r) * t,
            from.g + (to.g - from.g) * t,
            from.b + (to.b - from.b) * t,
            0.85,
        )
    }
}
