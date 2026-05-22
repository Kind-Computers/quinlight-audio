// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Kind Computers, LLC.

#![allow(
    clippy::arc_with_non_send_sync,
    clippy::collapsible_if,
    clippy::manual_is_multiple_of,
    clippy::redundant_closure,
    clippy::type_complexity
)]

pub mod icon;
mod scope_shader;
pub mod theme;
mod vinyl_shader;
pub mod widgets;

use iced::widget::{
    Column, Shader, button, canvas, checkbox, column, container, horizontal_rule, horizontal_space,
    mouse_area, opaque, pick_list, progress_bar, row, scrollable, slider, stack, text,
};
use iced::{Element, Length, Subscription, Task, Theme};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

// ── Background canvas: rotating vinyl record + twinkling stars ──

struct BgStar {
    x: f32,
    y: f32,
    brightness: f32,
    twinkle_speed: f32,
    phase: f32,
    size: f32,
}

/// Info displayed on the vinyl record label.
#[derive(Clone, Default)]
struct RecordLabel {
    title: String,
    artist: String,
    format_badge: String, // "S3M | 22ch | 44 samples | 314 KB"
    tracker: String,
}

struct BackgroundState {
    stars: Vec<BgStar>,
    time: f64,
    angle: f64, // cumulative rotation in radians
    frame: u32,
    playing: bool,
    label: RecordLabel,
    shuffled_splashes: Vec<String>,
    scroller_text: String,
    scroller_offset: f64,
    cache: canvas::Cache,
}

const SPLASH_MESSAGES: &[&str] = &[
    "This AI was not available in 1994!",
    "Amiga 500 not included",
    "Greetings to all sceners!",
    "Future Crew sends their regards",
    "Your sample rate is showing...",
    "Now with 0% AI hallucinations",
    "Also try Schism Tracker!",
    "Ripped from Hornet Archive",
    "MOD, S3M, XM, IT, and vibes",
    "48kHz or bust",
    "This is not a SoundCloud player",
    "Press F5 to compile... wait, different app",
    "Amiga forever!",
    "Tracking since before it was cool",
    "Purple Motion approved*  (*not yet)",
    "Dedicated to the memory of Necros",
    "What would Skaven do?",
    "Hugi would have reviewed this",
    "Works on my Amiga",
    "The sample rate is a lie",
    "Now loading... just kidding",
    "Remastered with exquisite taste",
    "More channels than your TV",
    "Fewer bugs than a demo compo",
    "Requires mass storage device",
    "Not tested on real hardware",
    "Compliant with Adlib standards",
    "Now with extra bits per sample",
    "The BBS called, they want their MODs back",
    "64-bit ought to be enough for anybody",
    "No GUS, no glory",
    "Imagine this with an OPL3",
    "Certified by the SID Chip Appreciation Society",
    "Warning: may contain tracker nostalgia",
    "Runs on pure demoscene energy",
    "Insert disk 2 to continue...",
    "DMA transfer complete!",
    "IRQ 7 acknowledged",
    "MCP says: greetings, programs",
    "All your samples are belong to us",
    "Don't cross the streams... or the channels",
    "Handle with care: contains high-frequency content!",
    "One does not simply resample to 48kHz",
    "In a world of MP3s, be a MOD file",
    "Keep calm and track on",
];

const PRESENTED_BY_KIND_COMPUTERS: &str = "Presented by Kind Computers";

impl BackgroundState {
    fn new() -> Self {
        let mut seed: u64 = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0xDEAD_BEEF_CAFE);
        let mut rng = || -> f32 {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((seed >> 33) as f32) / (u32::MAX as f32 / 2.0)
        };

        let stars: Vec<BgStar> = (0..120)
            .map(|_| {
                let size_roll = rng();
                BgStar {
                    x: rng(),
                    y: rng(),
                    brightness: 0.2 + rng() * 0.6,
                    twinkle_speed: 0.5 + rng() * 2.5,
                    phase: rng() * std::f32::consts::TAU,
                    size: if size_roll > 0.85 { 2.0 } else { 1.0 },
                }
            })
            .collect();

        // Shuffle splash messages using Fisher-Yates with the LCG
        let mut splashes: Vec<String> = SPLASH_MESSAGES.iter().map(|s| s.to_string()).collect();
        for i in (1..splashes.len()).rev() {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let j = (seed >> 33) as usize % (i + 1);
            splashes.swap(i, j);
        }

        // Build default scroller interleaving "QUINLIGHT AUDIO" info with shuffled messages
        let mut scroller_parts: Vec<String> = Vec::new();
        let info_parts = [
            "QUINLIGHT AUDIO",
            PRESENTED_BY_KIND_COMPUTERS,
            "Greetings to all sceners!",
        ];
        let mut splash_iter = splashes.iter().cycle();
        for info in &info_parts {
            scroller_parts.push(info.to_string());
            if let Some(msg) = splash_iter.next() {
                scroller_parts.push(msg.clone());
            }
        }
        let scroller_text = format!("    {}  ///  ", scroller_parts.join("  ///  "));

        Self {
            stars,
            time: 0.0,
            angle: 0.0,
            frame: 0,
            playing: false,
            label: RecordLabel::default(),
            shuffled_splashes: splashes,
            scroller_text,
            scroller_offset: 0.0,
            cache: canvas::Cache::default(),
        }
    }

    fn tick(&mut self, playing: bool, dt: f32) {
        let dt = dt as f64;
        self.time += dt;
        self.playing = playing;
        // Frame advances every tick so the vinyl shader's Halton jitter never
        // freezes on a single phase — otherwise paused frames bake one spatial
        // sampling pattern into the output and any residual sparkle sticks.
        self.frame = self.frame.wrapping_add(1);
        self.angle += 0.4924 * dt; // 1.5 * 3.1337 RPM
        // Advance scroller at ~60 pixels/sec
        self.scroller_offset += 15.0 * dt;
        let text_width = self.scroller_text.len() as f64 * 8.5; // approx char width at size 14
        if text_width > 0.0 && self.scroller_offset > text_width {
            self.scroller_offset -= text_width;
        }
        self.cache.clear();
    }

    fn set_label(&mut self, info: &crate::openmpt::ModuleInfo) {
        let format_upper = info.format_type.to_uppercase();
        let sample_count = info.samples.len();
        let size_str = if info.file_size_bytes >= 1_048_576 {
            format!("{:.1} MB", info.file_size_bytes as f64 / 1_048_576.0)
        } else {
            format!("{} KB", info.file_size_bytes / 1024)
        };
        self.label = RecordLabel {
            title: if info.title.is_empty() {
                "Untitled".into()
            } else {
                info.title.clone()
            },
            artist: info.artist.clone(),
            format_badge: format!(
                "{format_upper} | {}ch | {} samples | {size_str}",
                info.num_channels, sample_count
            ),
            tracker: info.tracker.clone(),
        };
        // Update scroller text — interleave module info with shuffled splash messages
        let mut info_parts: Vec<String> = Vec::new();
        info_parts.push(format!("    {}", self.label.title));
        if !self.label.artist.is_empty() {
            info_parts.push(self.label.artist.clone());
        }
        info_parts.push(self.label.format_badge.clone());
        info_parts.push(PRESENTED_BY_KIND_COMPUTERS.into());
        info_parts.push("Greetings to all sceners!".into());

        let mut scroller_parts: Vec<String> = Vec::new();
        let mut splash_iter = self.shuffled_splashes.iter().cycle();
        for part in &info_parts {
            scroller_parts.push(part.clone());
            if let Some(msg) = splash_iter.next() {
                scroller_parts.push(msg.clone());
            }
        }
        self.scroller_text = format!("    {}  ///  ", scroller_parts.join("  ///  "));
        self.scroller_offset = 0.0;
    }
}

/// Cardinal offsets for a 1px soft-glow halo (anti-aliasing approximation).
const GLOW_OFFSETS: [(f32, f32); 4] = [(0.0, -1.0), (0.0, 1.0), (-1.0, 0.0), (1.0, 0.0)];

/// Draw a soft 1px glow halo behind a text element.
fn draw_text_with_glow(frame: &mut canvas::Frame, text: canvas::Text, glow_strength: f32) {
    let glow_alpha = text.color.a * glow_strength;
    for &(dx, dy) in &GLOW_OFFSETS {
        let mut glow = text.clone();
        glow.position = iced::Point::new(text.position.x + dx, text.position.y + dy);
        glow.color.a = glow_alpha;
        frame.fill_text(glow);
    }
    frame.fill_text(text);
}

/// Draw a soft 1px glow halo behind a small rectangle.
fn draw_rect_with_glow(
    frame: &mut canvas::Frame,
    pos: iced::Point,
    size: iced::Size,
    color: iced::Color,
    glow_strength: f32,
) {
    let glow_color = iced::Color {
        a: color.a * glow_strength,
        ..color
    };
    for &(dx, dy) in &GLOW_OFFSETS {
        frame.fill_rectangle(iced::Point::new(pos.x + dx, pos.y + dy), size, glow_color);
    }
    frame.fill_rectangle(pos, size, color);
}

struct WaveformProgram<'a> {
    data: Vec<f64>,
    channels: i32,
    normal_loop: crate::openmpt::SampleLoopRegion,
    sustain_loop: crate::openmpt::SampleLoopRegion,
    cache: &'a canvas::Cache,
}

impl<Message> canvas::Program<Message> for WaveformProgram<'_> {
    type State = ();

    fn draw(
        &self,
        _state: &(),
        renderer: &iced::Renderer,
        _theme: &Theme,
        bounds: iced::Rectangle,
        _cursor: iced::mouse::Cursor,
    ) -> Vec<canvas::Geometry> {
        let geometry = self.cache.draw(renderer, bounds.size(), |frame| {
            self.draw_into(frame, bounds);
        });
        vec![geometry]
    }
}

impl WaveformProgram<'_> {
    fn draw_into(&self, frame: &mut canvas::Frame, bounds: iced::Rectangle) {
        let w = bounds.width;
        let h = bounds.height;

        frame.fill_rectangle(iced::Point::ORIGIN, iced::Size::new(w, h), theme::PANEL_BG);

        let channels = self.channels.max(1) as usize;
        let frames = self.data.len() / channels;
        if frames == 0 || w <= 0.0 {
            return;
        }

        let y_mid = h * 0.5;
        let max_amp = h * 0.45;

        let grid = theme::PANEL_SURFACE;
        frame.fill_rectangle(
            iced::Point::new(0.0, y_mid - 0.5),
            iced::Size::new(w, 1.0),
            grid,
        );

        let columns = w.ceil().max(1.0) as usize;
        let waveform_color = theme::ACCENT_GREEN;

        for col in 0..columns {
            let start_f = (col as u64 * frames as u64 / columns as u64) as usize;
            let mut end_f = ((col + 1) as u64 * frames as u64 / columns as u64) as usize;
            if end_f <= start_f {
                end_f = start_f + 1;
            }
            let end_f = end_f.min(frames);
            if start_f >= end_f {
                continue;
            }
            let mut min_v: f64 = 0.0;
            let mut max_v: f64 = 0.0;
            for f in start_f..end_f {
                let base = f * channels;
                if base + channels > self.data.len() {
                    break;
                }
                let mut sum = 0.0;
                for ch in 0..channels {
                    sum += self.data[base + ch];
                }
                let v = sum / channels as f64;
                if v < min_v {
                    min_v = v;
                }
                if v > max_v {
                    max_v = v;
                }
            }
            let min_y = y_mid - (min_v as f32).clamp(-1.0, 1.0) * max_amp;
            let max_y = y_mid - (max_v as f32).clamp(-1.0, 1.0) * max_amp;
            let top = max_y.min(min_y);
            let bot = max_y.max(min_y);
            frame.fill_rectangle(
                iced::Point::new(col as f32, top),
                iced::Size::new(1.0, (bot - top).max(1.0)),
                waveform_color,
            );
        }

        let frames_f = frames as f32;
        let draw_dotted_line = |frame: &mut canvas::Frame, x: f32, color: iced::Color| {
            let dash_len = 3.0;
            let gap_len = 3.0;
            let mut y = 0.0;
            while y < h {
                let next = (y + dash_len).min(h);
                frame.fill_rectangle(
                    iced::Point::new(x, y),
                    iced::Size::new(1.0, next - y),
                    color,
                );
                y = next + gap_len;
            }
        };

        let frame_to_x = |f: i64| -> f32 { (f as f32 / frames_f).clamp(0.0, 1.0) * w };

        let normal_color = theme::ACCENT_AMBER;
        if self.normal_loop.has_loop() {
            draw_dotted_line(
                &mut *frame,
                frame_to_x(self.normal_loop.start_frames),
                normal_color,
            );
            draw_dotted_line(
                &mut *frame,
                frame_to_x(self.normal_loop.end_frames),
                normal_color,
            );
        }

        let sustain_color = theme::ACCENT_GREEN;
        if self.sustain_loop.has_loop() {
            draw_dotted_line(
                &mut *frame,
                frame_to_x(self.sustain_loop.start_frames),
                sustain_color,
            );
            draw_dotted_line(
                &mut *frame,
                frame_to_x(self.sustain_loop.end_frames),
                sustain_color,
            );
        }
    }
}

impl<Message> canvas::Program<Message> for BackgroundState {
    type State = ();

    fn draw(
        &self,
        _state: &(),
        renderer: &iced::Renderer,
        _theme: &Theme,
        bounds: iced::Rectangle,
        _cursor: iced::mouse::Cursor,
    ) -> Vec<canvas::Geometry> {
        let bg = self.cache.draw(renderer, bounds.size(), |frame| {
            let w = bounds.width;
            let h = bounds.height;
            let cx = w / 2.0;
            let cy = h / 2.0;

            // Vinyl disc + dark background rendered by GPU shader (vinyl_shader.rs).
            // Canvas only draws overlay elements: label text, stars, glow, scroller.
            let record_radius = (h * 0.42).min(w * 0.35);
            let label_radius = record_radius * 0.35;

            // ── Label text (rotates with the record) ──
            if !self.label.title.is_empty() {
                frame.with_save(|frame| {
                    frame.translate(iced::Vector::new(cx, cy));
                    frame.rotate(self.angle as f32);

                    // Title
                    draw_text_with_glow(
                        frame,
                        canvas::Text {
                            content: self.label.title.clone(),
                            position: iced::Point::new(0.0, -label_radius * 0.45),
                            color: theme::CONTENT_TEXT,
                            size: iced::Pixels(13.0),
                            font: iced::Font {
                                weight: iced::font::Weight::Bold,
                                ..Default::default()
                            },
                            horizontal_alignment: iced::alignment::Horizontal::Center,
                            vertical_alignment: iced::alignment::Vertical::Center,
                            ..Default::default()
                        },
                        0.10,
                    );

                    // Artist
                    if !self.label.artist.is_empty() {
                        draw_text_with_glow(
                            frame,
                            canvas::Text {
                                content: self.label.artist.clone(),
                                position: iced::Point::new(0.0, -label_radius * 0.2),
                                color: theme::CONTENT_TEXT,
                                size: iced::Pixels(11.0),
                                horizontal_alignment: iced::alignment::Horizontal::Center,
                                vertical_alignment: iced::alignment::Vertical::Center,
                                ..Default::default()
                            },
                            0.10,
                        );
                    }

                    // Format badge
                    draw_text_with_glow(
                        frame,
                        canvas::Text {
                            content: self.label.format_badge.clone(),
                            position: iced::Point::new(0.0, label_radius * 0.15),
                            color: theme::CONTENT_TEXT,
                            size: iced::Pixels(9.0),
                            horizontal_alignment: iced::alignment::Horizontal::Center,
                            vertical_alignment: iced::alignment::Vertical::Center,
                            ..Default::default()
                        },
                        0.10,
                    );

                    // Tracker
                    if !self.label.tracker.is_empty() {
                        draw_text_with_glow(
                            frame,
                            canvas::Text {
                                content: self.label.tracker.clone(),
                                position: iced::Point::new(0.0, label_radius * 0.4),
                                color: theme::CONTENT_TEXT,
                                size: iced::Pixels(8.0),
                                horizontal_alignment: iced::alignment::Horizontal::Center,
                                vertical_alignment: iced::alignment::Vertical::Center,
                                ..Default::default()
                            },
                            0.10,
                        );
                    }
                });
            }

            // ── Twinkling stars (avoid the record area) ──
            let exclusion_radius = record_radius + 10.0;
            for star in &self.stars {
                let twinkle = (self.time as f32 * star.twinkle_speed + star.phase).sin();
                let alpha = (star.brightness + twinkle * 0.3).clamp(0.0, 1.0);
                if alpha < 0.05 {
                    continue;
                }

                let sx = star.x * w;
                let sy = star.y * h;
                let dx = sx - cx;
                let dy = sy - cy;
                if dx * dx + dy * dy < exclusion_radius * exclusion_radius {
                    continue;
                }
                let sz = star.size;

                let (r, g, b) = if star.brightness > 0.5 {
                    (0.7, 0.9, 1.0)
                } else {
                    (0.9, 0.9, 0.95)
                };

                draw_rect_with_glow(
                    frame,
                    iced::Point::new(sx, sy),
                    iced::Size::new(sz, sz),
                    iced::Color::from_rgba(r, g, b, alpha * 0.8),
                    0.12,
                );
            }

            // ── Quinlight glow behind the record (subtle) ──
            let nova_size = 200u32;
            let rgba = icon::create_icon_rgba(nova_size);
            let dim = 0.15_f32;
            let ox = (w - nova_size as f32) / 2.0;
            let oy = (h - nova_size as f32) / 2.0;
            for py in 0..nova_size {
                for px in 0..nova_size {
                    let idx = ((py * nova_size + px) * 4) as usize;
                    let a = rgba[idx + 3] as f32 / 255.0 * dim;
                    if a < 0.01 {
                        continue;
                    }
                    let npx = ox + px as f32;
                    let npy = oy + py as f32;
                    // Only draw outside the record to create a glow halo
                    let ddx = npx - cx;
                    let ddy = npy - cy;
                    if ddx * ddx + ddy * ddy < record_radius * record_radius {
                        continue;
                    }
                    let r = rgba[idx] as f32 / 255.0 * dim;
                    let g = rgba[idx + 1] as f32 / 255.0 * dim;
                    let b = rgba[idx + 2] as f32 / 255.0 * dim;
                    frame.fill_rectangle(
                        iced::Point::new(npx, npy),
                        iced::Size::new(1.0, 1.0),
                        iced::Color::from_rgba(r.min(1.0), g.min(1.0), b.min(1.0), a.min(1.0)),
                    );
                }
            }

            // ── Sine-wave copper-bar scroller at the bottom ──
            if !self.scroller_text.is_empty() {
                let base_y = h - 30.0;
                let char_w = 8.5_f32;
                let chars: Vec<char> = self.scroller_text.chars().collect();
                let text_width = chars.len() as f32 * char_w;
                if text_width > 0.0 {
                    // Draw characters that are visible on screen
                    let scroller_offset_f32 = self.scroller_offset as f32;
                    let time_f32 = self.time as f32;
                    let start_char = (scroller_offset_f32 / char_w) as usize;
                    let visible_chars = (w / char_w) as usize + 2;
                    for i in 0..visible_chars {
                        let ci = (start_char + i) % chars.len();
                        let x = i as f32 * char_w - (scroller_offset_f32 % char_w);
                        let sine_y = (x * 0.04 + time_f32 * 2.0).sin() * 4.0;
                        // Copper-bar gradient: cycle through warm colors
                        let t =
                            ((ci as f32 * 0.15 + time_f32 * 1.5).sin() * 0.5 + 0.5).clamp(0.0, 1.0);
                        let r = 0.25 + t * 0.3;
                        let g = 0.12 + t * 0.3;
                        let b = 0.03 + t * 0.08;
                        // Use translate for sub-pixel precision (bypasses glyph snapping)
                        frame.with_save(|frame| {
                            frame.translate(iced::Vector::new(x, base_y + sine_y));
                            draw_text_with_glow(
                                frame,
                                canvas::Text {
                                    content: chars[ci].to_string(),
                                    position: iced::Point::ORIGIN,
                                    color: iced::Color::from_rgba(r, g, b, 0.8),
                                    size: iced::Pixels(14.0),
                                    font: iced::Font::MONOSPACE,
                                    ..Default::default()
                                },
                                0.07,
                            );
                        });
                    }
                }
            }
        });

        vec![bg]
    }
}

const OSCILLOSCOPE_HEIGHT: f32 = 72.0;
const SPECTROGRAM_HEIGHT: f32 = 150.0;
pub const INITIAL_WINDOW_WIDTH: f32 = 1280.0;
pub const INITIAL_WINDOW_HEIGHT: f32 = 720.0;

const PANEL_MIN_PATTERN_VIEWER_HEIGHT: f32 = 60.0;
const PANEL_MIN_VU_METERS_HEIGHT: f32 = 40.0;
const PANEL_MIN_SAMPLE_PANEL_HEIGHT: f32 = 80.0;

const CONTENT_PADDING_Y: f32 = 16.0;
const ROOT_COLUMN_SPACING: f32 = 4.0;
const TOP_COLUMN_SPACING: f32 = 44.0;
const BOTTOM_COLUMN_SPACING: f32 = 4.0;
const TOP_RULES_HEIGHT: f32 = 2.0;
const BOTTOM_RULE_HEIGHT: f32 = 1.0;
const DRAG_HANDLES_HEIGHT: f32 = 12.0;
const HEADER_BUDGET_HEIGHT: f32 = 28.0;
const TRANSPORT_BUDGET_HEIGHT: f32 = 76.0;
const INFO_PANEL_BUDGET_HEIGHT: f32 = 18.0;
const REMASTER_ROW_BUDGET_HEIGHT: f32 = 28.0;
const REMASTER_BASE_BUDGET_HEIGHT: f32 = REMASTER_ROW_BUDGET_HEIGHT * 2.0 + BOTTOM_COLUMN_SPACING;
const REMASTER_PROGRESS_BUDGET_HEIGHT: f32 = 12.0;
const REMASTER_LOG_BUDGET_HEIGHT: f32 = 124.0;
const REMASTER_ERROR_BUDGET_HEIGHT: f32 = 154.0;
const REMASTER_NOTE_BUDGET_HEIGHT: f32 = 22.0;

#[derive(Debug, Clone, Copy, PartialEq)]
struct PanelHeights {
    pattern_viewer: f32,
    vu_meters: f32,
    sample_panel: f32,
}

impl PanelHeights {
    const fn new(pattern_viewer: f32, vu_meters: f32, sample_panel: f32) -> Self {
        Self {
            pattern_viewer,
            vu_meters,
            sample_panel,
        }
    }

    const fn minimums() -> Self {
        Self::new(
            PANEL_MIN_PATTERN_VIEWER_HEIGHT,
            PANEL_MIN_VU_METERS_HEIGHT,
            PANEL_MIN_SAMPLE_PANEL_HEIGHT,
        )
    }

    fn total(self) -> f32 {
        self.pattern_viewer + self.vu_meters + self.sample_panel
    }

    fn at_least(self, minimums: Self) -> Self {
        Self::new(
            self.pattern_viewer.max(minimums.pattern_viewer),
            self.vu_meters.max(minimums.vu_meters),
            self.sample_panel.max(minimums.sample_panel),
        )
    }
}

pub fn initial_window_size() -> iced::Size {
    iced::Size::new(INITIAL_WINDOW_WIDTH, INITIAL_WINDOW_HEIGHT)
}

pub fn minimum_window_size() -> iced::Size {
    iced::Size::new(1.0, minimum_window_height())
}

fn remaster_panel_budget_height(
    has_error: bool,
    has_progress: bool,
    has_log: bool,
    has_note: bool,
) -> f32 {
    if has_error {
        REMASTER_ROW_BUDGET_HEIGHT + BOTTOM_COLUMN_SPACING + REMASTER_ERROR_BUDGET_HEIGHT
    } else {
        let mut height = REMASTER_BASE_BUDGET_HEIGHT;
        if has_progress {
            height += REMASTER_PROGRESS_BUDGET_HEIGHT;
        }
        if has_log {
            height += REMASTER_LOG_BUDGET_HEIGHT;
        }
        if has_note {
            height += REMASTER_NOTE_BUDGET_HEIGHT;
        }
        height
    }
}

fn worst_case_remaster_panel_height() -> f32 {
    remaster_panel_budget_height(false, true, true, true)
        .max(remaster_panel_budget_height(true, false, false, false))
}

fn fixed_chrome_height(remaster_height: f32) -> f32 {
    CONTENT_PADDING_Y
        + ROOT_COLUMN_SPACING
        + TOP_COLUMN_SPACING
        + BOTTOM_COLUMN_SPACING
        + TOP_RULES_HEIGHT
        + BOTTOM_RULE_HEIGHT
        + DRAG_HANDLES_HEIGHT
        + HEADER_BUDGET_HEIGHT
        + TRANSPORT_BUDGET_HEIGHT
        + OSCILLOSCOPE_HEIGHT
        + SPECTROGRAM_HEIGHT
        + INFO_PANEL_BUDGET_HEIGHT
        + remaster_height
}

fn minimum_window_height() -> f32 {
    fixed_chrome_height(worst_case_remaster_panel_height()) + PanelHeights::minimums().total()
}

fn available_panel_height(viewport_height: f32, remaster_height: f32) -> f32 {
    (viewport_height - fixed_chrome_height(remaster_height)).max(PanelHeights::minimums().total())
}

fn resolve_panel_heights(preferred: PanelHeights, available: f32) -> PanelHeights {
    let minimums = PanelHeights::minimums();
    let preferred = preferred.at_least(minimums);
    let minimum_total = minimums.total();
    let preferred_total = preferred.total();

    if available <= minimum_total {
        return minimums;
    }
    if preferred_total <= available {
        return preferred;
    }

    let available_slack = available - minimum_total;
    let preferred_slack = preferred_total - minimum_total;
    if preferred_slack <= 0.0 {
        return minimums;
    }

    let scale = (available_slack / preferred_slack).clamp(0.0, 1.0);
    PanelHeights::new(
        minimums.pattern_viewer + (preferred.pattern_viewer - minimums.pattern_viewer) * scale,
        minimums.vu_meters + (preferred.vu_meters - minimums.vu_meters) * scale,
        minimums.sample_panel + (preferred.sample_panel - minimums.sample_panel) * scale,
    )
}

/// Color a pattern string based on highlight codes from libopenmpt.
fn highlight_pattern_text<'a>(fmt: &str, hl: &str, is_current: bool) -> Element<'a, Message> {
    // If no highlight data or empty, just return plain text
    if hl.is_empty() || fmt.is_empty() {
        return text(fmt.to_string())
            .size(10)
            .font(iced::Font::MONOSPACE)
            .color(theme::PANEL_SURFACE)
            .into();
    }

    let dim = if is_current { 1.0_f32 } else { 0.55 };
    let mut items: Vec<Element<'a, Message>> = Vec::new();
    let fmt_chars: Vec<char> = fmt.chars().collect();
    let hl_chars: Vec<char> = hl.chars().collect();

    // Group consecutive chars with same highlight code
    let mut i = 0;
    while i < fmt_chars.len() {
        let code = hl_chars.get(i).copied().unwrap_or(' ');
        let mut end = i + 1;
        while end < fmt_chars.len() && hl_chars.get(end).copied().unwrap_or(' ') == code {
            end += 1;
        }
        let chunk: String = fmt_chars[i..end].iter().collect();
        let c = match code {
            'n' => iced::Color::from_rgba(1.0 * dim, 1.0 * dim, 1.0 * dim, 0.75), // note: white
            'm' => iced::Color::from_rgba(1.0 * dim, 0.7 * dim, 0.3 * dim, 0.75), // special note: orange
            'i' => iced::Color::from_rgba(1.0 * dim, 0.9 * dim, 0.4 * dim, 0.75), // instrument: yellow
            'u' => iced::Color::from_rgba(0.4 * dim, 1.0 * dim, 0.4 * dim, 0.75), // vol effect: green
            'v' => iced::Color::from_rgba(0.3 * dim, 0.8 * dim, 0.3 * dim, 0.75), // vol param: dim green
            'e' => iced::Color::from_rgba(0.3 * dim, 0.9 * dim, 1.0 * dim, 0.75), // effect: cyan
            'f' => iced::Color::from_rgba(0.2 * dim, 0.7 * dim, 0.8 * dim, 0.75), // effect param: dim cyan
            _ => iced::Color::from_rgba(0.25 * dim, 0.25 * dim, 0.3 * dim, 0.75), // empty: dark grey
        };
        items.push(
            text(chunk)
                .size(10)
                .font(iced::Font::MONOSPACE)
                .color(c)
                .into(),
        );
        i = end;
    }

    iced::widget::Row::with_children(items).spacing(0).into()
}

/// Map keyboard key to MIDI note number for keyjazz.
/// Lower row = octave 4 (C-4 = note 60), upper row = octave 5.
fn key_to_note(key: &str) -> Option<i32> {
    match key {
        // Lower row: C-4 through B-4
        "z" => Some(60), // C-4
        "s" => Some(61), // C#4
        "x" => Some(62), // D-4
        "d" => Some(63), // D#4
        "c" => Some(64), // E-4
        "v" => Some(65), // F-4
        "g" => Some(66), // F#4
        "b" => Some(67), // G-4
        "h" => Some(68), // G#4
        "n" => Some(69), // A-4
        "j" => Some(70), // A#4
        "m" => Some(71), // B-4
        // Upper row: C-5 through B-5
        "q" => Some(72), // C-5
        "2" => Some(73), // C#5
        "w" => Some(74), // D-5
        "3" => Some(75), // D#5
        "e" => Some(76), // E-5
        "r" => Some(77), // F-5
        "5" => Some(78), // F#5
        "t" => Some(79), // G-5
        "6" => Some(80), // G#5
        "y" => Some(81), // A-5
        "7" => Some(82), // A#5
        "u" => Some(83), // B-5
        _ => None,
    }
}

use crate::openmpt::{Module, ModuleInfo, SampleFormat};
use crate::player::{PlaybackStatus, Player, PlayerCommand, PlayerState};
use crate::remaster::{
    CleanupEngineVersion, CleanupMode, CleanupSettings, RemasterEngine, RemasterOutput,
    RemasterStatus, SampleResult, UpscaleMode,
};

#[derive(Debug, Clone)]
struct PendingModuleLoadContext {
    loaded_path: Option<PathBuf>,
    stereo_separation: i32,
}

#[derive(Debug)]
struct PreparedModuleLoadPackage {
    request_id: u64,
    context: PendingModuleLoadContext,
    prepared: crate::player::PreparedModuleLoad,
}

#[derive(Debug)]
struct PendingGrooveRender {
    request_id: u64,
    file_data: Vec<u8>,
    stereo_separation: i32,
    interpolation_filter: i32,
}

#[derive(Clone)]
pub(crate) struct PreparedModuleLoadHandle(Arc<Mutex<Option<PreparedModuleLoadPackage>>>);

impl PreparedModuleLoadHandle {
    fn new(package: PreparedModuleLoadPackage) -> Self {
        Self(Arc::new(Mutex::new(Some(package))))
    }

    fn take(&self) -> Option<PreparedModuleLoadPackage> {
        self.0.lock().unwrap().take()
    }
}

impl std::fmt::Debug for PreparedModuleLoadHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("PreparedModuleLoadHandle").finish()
    }
}

#[derive(Debug, Clone)]
pub(crate) struct ExtractedArchiveModule {
    entry_path: String,
    data: Vec<u8>,
}

#[derive(Debug, Clone)]
pub(crate) enum PreparedModuleLoadOutcome {
    Success(PreparedModuleLoadHandle),
    Failure { request_id: u64, error: String },
}

#[derive(Debug, Clone)]
pub enum Message {
    // File
    OpenFileDialog,
    FileSelected(Option<PathBuf>),

    // Transport
    Play,
    Pause,
    Stop,
    SeekChanged(f64),
    SeekReleased,

    // State updates
    Tick,
    WindowResized(iced::Size),

    // Remaster
    StartRemaster,
    CancelRemaster,
    RemasterUpdate(RemasterStatus),
    ConfirmRemasterInterrupt,
    DismissRemasterInterrupt,
    ShowInstallDialog,
    ShowInstallMissing,
    DismissInstallDialog,
    CopyInstallCommand,
    CopyRemasterError,

    // Render
    RenderFlac,
    RenderFlacComplete(Result<PathBuf, String>),
    RenderAac,
    RenderAacComplete(Result<PathBuf, String>),

    // Audio error
    DismissAudioError,

    // Settings
    SetInterpolation(InterpolationChoice),
    SetStereoSeparation(i32),
    SetAgcEnabled(bool),
    SetHrtfEnabled(bool),
    SetHrtfMix(i32),
    SetCleanupMode(CleanupMode),
    SetCleanupEngineVersion(CleanupEngineVersion),
    ToggleEngine(String, bool),
    SetDdimSteps(u32),

    // Cache
    ClearSongCache,

    // Per-sample mode cycling
    CycleSampleMode(i32),
    CycleAllSampleModes,
    ToggleSampleMute(i32),
    SoloSample(i32),

    // Archive browsing
    ArchiveContents(Result<Vec<crate::archive::ArchiveEntry>, String>),
    ArchiveSelect(usize),
    ArchiveExtracted(Result<ExtractedArchiveModule, String>),
    PreparedModuleLoadReady(PreparedModuleLoadOutcome),
    DismissArchivePicker,

    // Vinyl groove rendering
    GrooveDataReady { request_id: u64, data: Vec<f32> },

    // Window close with fade-out
    CloseRequested(iced::window::Id),
    ResolvedOldestWindow(Option<iced::window::Id>),
    FadeOutDone,
    FinalClose,

    // Keyboard shortcuts
    KeyPressed(iced::keyboard::Key, iced::keyboard::Modifiers),

    // Keyjazz
    SelectKeyjazzSample(i32),

    // Blind test
    StartBlindTest,
    PlayBlindA,
    PlayBlindB,
    GuessBlind(bool), // true = user thinks A is AI
    NextBlindTest,
    DismissBlindTest,

    // Drag and drop
    FileDropped(PathBuf),

    // Panel resizing
    DragStart(DragPanel),
    DragMove(f32),
    DragEnd,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DragPanel {
    PatternViewer,
    VuMeters,
    SampleList,
}

#[derive(Debug, Clone, PartialEq)]
enum SampleMode {
    Engine(String), // Named engine output: "AudioSR", "LavaSR", "FLowHigh"
    Reference48k,   // Pure SINC-resampled 48 kHz reference (no AI processing)
    Original,       // Raw original rate
}

#[derive(Debug, Clone, PartialEq)]
enum RemasterInterruptAction {
    Close(iced::window::Id),
    LoadFile(PathBuf),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RemasterPrimaryAction {
    Install,
    Start,
    Cancel,
    Cancelling,
    Complete,
    Disabled,
}

const QUINLIGHT_ENGINE_NAME: &str = "Quinlight Audio";
const QUINLIGHT_ORIGINAL_TAG: &str = "Quinlight Audio (Original)";
const CLEANUP_MODES: &[CleanupMode] = &CleanupMode::ALL;
const CLEANUP_ENGINES: &[CleanupEngineVersion] = &CleanupEngineVersion::ALL;

fn enabled_engine_names(
    enable_audiosr: bool,
    enable_lavasr: bool,
    enable_flowhigh: bool,
    enable_apbwe: bool,
) -> Vec<String> {
    let mut enabled = Vec::new();
    if enable_lavasr {
        enabled.push("LavaSR".to_string());
    }
    if enable_flowhigh {
        enabled.push("FLowHigh".to_string());
    }
    if enable_apbwe {
        enabled.push("AP-BWE".to_string());
    }
    if enable_audiosr {
        enabled.push("AudioSR".to_string());
    }
    enabled
}

fn open_button_tooltip() -> &'static str {
    "Open a module file or supported archive."
}

fn play_pause_button_tooltip(is_playing: bool) -> &'static str {
    if is_playing {
        "Pause playback."
    } else {
        "Start playback."
    }
}

fn stop_button_tooltip() -> &'static str {
    "Stop playback and rewind to the start."
}

fn render_button_tooltip(format: &str, enabled: bool) -> String {
    let article = if format.starts_with('A') || format.starts_with('M') {
        "an"
    } else {
        "a"
    };
    if enabled {
        format!("Render the current module to {article} {format} file.")
    } else {
        format!("Load a module to render {article} {format} file.")
    }
}

fn blind_test_button_tooltip(enabled: bool) -> &'static str {
    if enabled {
        "Start an A/B blind test for remastered samples."
    } else {
        "Remaster at least one sample to unlock blind test."
    }
}

fn agc_checkbox_tooltip(enabled: bool) -> &'static str {
    if enabled {
        "Disable automatic gain control during playback."
    } else {
        "Enable automatic gain control during playback."
    }
}

fn cycle_all_samples_tooltip() -> &'static str {
    "Cycle all remastered samples to the next playback mode."
}

fn sample_mode_name_for_slot(slot: &SampleSlot, mode: &SampleMode) -> String {
    match mode {
        SampleMode::Engine(name) => name.clone(),
        SampleMode::Reference48k => "Reference 48k (SINC)".into(),
        SampleMode::Original if slot.quinlight_original_fallback => QUINLIGHT_ORIGINAL_TAG.into(),
        SampleMode::Original => "original audio".into(),
    }
}

fn sample_mode_button_label(slot: &SampleSlot) -> String {
    match &slot.mode {
        SampleMode::Engine(name) => name.clone(),
        SampleMode::Reference48k => "Ref 48k".into(),
        SampleMode::Original if slot.quinlight_original_fallback => QUINLIGHT_ORIGINAL_TAG.into(),
        SampleMode::Original => "Original".to_string(),
    }
}

fn sample_mode_button_color(slot: &SampleSlot) -> iced::Color {
    match &slot.mode {
        // Live engine output → accent green (same family as VU/oscilloscope).
        SampleMode::Engine(_) => theme::ACCENT_GREEN,
        // Pure SINC reference → neutral panel-label (not a "signal", a control).
        SampleMode::Reference48k => theme::PANEL_LABEL,
        // Quinlight fell back to the un-upscaled sample → neutral.
        SampleMode::Original if slot.quinlight_original_fallback => theme::PANEL_LABEL,
        // Raw original sample → amber (attention: you're hearing the un-upscaled signal).
        SampleMode::Original => theme::ACCENT_AMBER,
    }
}

fn sample_mode_tooltip(slot: &SampleSlot) -> String {
    format!(
        "Cycle sample {} from {} to {}.",
        slot.index + 1,
        sample_mode_name_for_slot(slot, &slot.mode),
        sample_mode_name_for_slot(slot, &slot.next_mode())
    )
}

fn sample_identity(index: i32, name: &str, uppercase: bool) -> String {
    let prefix = if uppercase { "Sample" } else { "sample" };
    let name = name.trim();

    if name.is_empty() {
        format!("{prefix} {}", index + 1)
    } else {
        format!("{prefix} {} \"{name}\"", index + 1)
    }
}

fn sample_row_tooltip(sample: &crate::openmpt::SampleInfo, is_keyjazz: bool) -> String {
    if is_keyjazz {
        format!(
            "{} is selected for keyjazz preview.",
            sample_identity(sample.index, &sample.name, true)
        )
    } else {
        format!(
            "Select {} for keyjazz preview.",
            sample_identity(sample.index, &sample.name, false)
        )
    }
}

fn remaster_primary_tooltip(action: RemasterPrimaryAction) -> &'static str {
    match action {
        RemasterPrimaryAction::Install => "Show AI engine install instructions.",
        RemasterPrimaryAction::Start => "Start remastering the loaded module.",
        RemasterPrimaryAction::Cancel => "Cancel the current remaster run.",
        RemasterPrimaryAction::Cancelling => "Remaster cancellation is in progress.",
        RemasterPrimaryAction::Complete => "Remastering finished for the current module.",
        RemasterPrimaryAction::Disabled => "Load a module to start remastering.",
    }
}

fn clear_song_cache_tooltip(enabled: bool) -> &'static str {
    if enabled {
        "Clear cached remaster data for this song."
    } else {
        "Load a module to clear its cached remaster data."
    }
}

fn engine_checkbox_tooltip(name: &str, enabled: bool) -> String {
    if enabled {
        format!("Disable {name} for the next remaster run.")
    } else {
        format!("Enable {name} for the next remaster run.")
    }
}

fn install_missing_button_tooltip(names: &[&str]) -> String {
    format!("Show install instructions for {}.", names.join(", "))
}

fn blind_test_play_tooltip(label: &str) -> String {
    format!("Play candidate {label} for this blind test.")
}

fn blind_test_guess_tooltip(label: &str) -> String {
    format!("Guess that {label} is the AI-remastered version.")
}

fn remaster_interrupt_confirm_tooltip(action: &RemasterInterruptAction) -> String {
    match action {
        RemasterInterruptAction::Close(_) => {
            "Cancel the current remaster and exit Quinlight.".into()
        }
        RemasterInterruptAction::LoadFile(path) => {
            let name = path
                .file_name()
                .and_then(|name| name.to_str())
                .unwrap_or("the selected file");
            format!("Cancel the current remaster and load {name}.")
        }
    }
}

fn archive_entry_tooltip(filename: &str) -> String {
    format!("Load \"{filename}\" from the archive.")
}

/// Per-engine remastered audio data.
#[derive(Debug, Clone)]
struct EngineResult {
    name: String,
    data: Vec<f64>,
    length_frames: i64,
    channels: i32,
    sample_rate_hz: i32,
    discovered_loops: Option<crate::remaster::DiscoveredLoopInfo>,
}

/// Per-sample state for original/remastered toggling.
struct SampleSlot {
    index: i32,
    original: Vec<f64>,
    original_rate: i32,
    original_channels: i32,
    original_length_frames: i64,
    #[allow(dead_code)] // Read only in tests (expected_quinlight_mix)
    loop_info: crate::openmpt::SampleLoopInfo,
    engine_results: Vec<EngineResult>,
    quinlight_result: Option<EngineResult>,
    quinlight_original_fallback: bool,
    mode: SampleMode,
    failed: bool,
    original_effects: Vec<crate::remaster::SavedEffectParam>,
    /// How many selected engines have completed this sample.
    engines_done: i32,
    /// Total engines participating in this run.
    engines_total: i32,
    /// Whether this sample is muted (replaced with silence for debugging).
    muted: bool,
    /// SINC-resampled 48 kHz copy of `original`, computed lazily on first
    /// entry to `SampleMode::Reference48k`.
    reference_48k: Option<Vec<f64>>,
}

impl SampleSlot {
    fn has_engine_results(&self) -> bool {
        self.quinlight_result.is_some() || !self.engine_results.is_empty()
    }

    fn available_modes(&self) -> Vec<SampleMode> {
        let mut modes = Vec::new();
        if let Some(sr) = &self.quinlight_result {
            modes.push(SampleMode::Engine(sr.name.clone()));
        }
        modes.extend(
            self.engine_results
                .iter()
                .map(|er| SampleMode::Engine(er.name.clone())),
        );
        modes.push(SampleMode::Reference48k);
        modes.push(SampleMode::Original);
        modes
    }

    fn next_mode(&self) -> SampleMode {
        let modes = self.available_modes();
        let current_idx = modes
            .iter()
            .position(|mode| *mode == self.mode)
            .unwrap_or(0);
        let next_idx = (current_idx + 1) % modes.len();
        modes[next_idx].clone()
    }

    fn engine_result(&self, name: &str) -> Option<&EngineResult> {
        if name.starts_with(QUINLIGHT_ENGINE_NAME) {
            return self.quinlight_result.as_ref();
        }
        self.engine_results.iter().find(|er| er.name == name)
    }

    fn preferred_blind_test_result(&self) -> Option<&EngineResult> {
        self.quinlight_result
            .as_ref()
            .or_else(|| self.engine_results.first())
    }

    fn blind_test_data(&self) -> Option<Vec<f64>> {
        let engine = self.preferred_blind_test_result()?;
        if engine.channels == 2 {
            Some(
                engine
                    .data
                    .chunks(2)
                    .map(|chunk| (chunk[0] + chunk[1]) * 0.5)
                    .collect(),
            )
        } else {
            Some(engine.data.clone())
        }
    }

    fn blind_test_original_data(&self) -> Result<Vec<f64>, String> {
        let original_rate = self.original_rate.max(1) as u32;
        let original_channels = self.original_channels.max(1) as usize;
        let preview = crate::remaster::resample_audio(
            &self.original,
            original_rate,
            48_000,
            original_channels,
            crate::remaster::ResampleBoundaryMode::OneShot,
        )?;

        Ok(if original_channels == 2 {
            preview
                .chunks(2)
                .map(|chunk| (chunk[0] + chunk[1]) * 0.5)
                .collect()
        } else {
            preview
        })
    }

    fn apply_candidate_result(
        &mut self,
        result: &SampleResult,
        saved_effects: Vec<crate::remaster::SavedEffectParam>,
    ) -> Option<SampleMode> {
        if result.data.is_empty() {
            return None;
        }

        if self.original_effects.is_empty() {
            self.original_effects = saved_effects;
        }
        self.quinlight_original_fallback = false;
        let next = EngineResult {
            name: result.engine_name.clone(),
            data: result.data.clone(),
            length_frames: result.length_frames,
            channels: result.channels,
            sample_rate_hz: result.sample_rate_hz,
            discovered_loops: result.discovered_loops.clone(),
        };
        if let Some(existing) = self
            .engine_results
            .iter_mut()
            .find(|er| er.name == next.name)
        {
            *existing = next;
        } else {
            self.engine_results.push(next);
            self.engine_results.sort_by(|a, b| {
                crate::engine::engine_preference_rank(&a.name)
                    .cmp(&crate::engine::engine_preference_rank(&b.name))
                    .then_with(|| a.name.cmp(&b.name))
            });
        }
        self.failed = false;
        Some(if self.quinlight_result.is_none() {
            SampleMode::Engine(result.engine_name.clone())
        } else {
            self.mode.clone()
        })
    }

    fn apply_final_result(
        &mut self,
        result: &SampleResult,
        saved_effects: Vec<crate::remaster::SavedEffectParam>,
    ) -> Option<SampleMode> {
        if result.data.is_empty() {
            return None;
        }

        if self.original_effects.is_empty() {
            self.original_effects = saved_effects;
        }
        if crate::remaster::is_no_consensus_result(&result.engine_name) {
            self.quinlight_result = None;
            self.quinlight_original_fallback = true;
            self.failed = false;
            return Some(SampleMode::Original);
        }
        self.quinlight_original_fallback = false;
        self.quinlight_result = Some(EngineResult {
            name: result.engine_name.clone(),
            data: result.data.clone(),
            length_frames: result.length_frames,
            channels: result.channels,
            sample_rate_hz: result.sample_rate_hz,
            discovered_loops: result.discovered_loops.clone(),
        });
        self.failed = false;
        Some(SampleMode::Engine(result.engine_name.clone()))
    }

    fn refresh_failed_state(&mut self) {
        if self.has_engine_results() {
            self.failed = false;
        } else if self.engines_total > 0 && self.engines_done >= self.engines_total {
            self.failed = true;
        }
    }
}

fn mute_sample(module: &mut Module, slot: &SampleSlot) {
    let frames = slot.original_length_frames.max(1) as usize;
    let channels = slot.original_channels.max(1) as usize;
    let silence = vec![0.0f64; frames * channels];
    let _ = module.replace_sample_data(
        slot.index,
        &silence,
        frames as i64,
        channels as i32,
        slot.original_rate,
    );
}

fn apply_sample_mode_to_slot(
    module: &mut Module,
    slot: &mut SampleSlot,
    target_mode: SampleMode,
    is_playing: bool,
) -> Result<(), String> {
    match target_mode.clone() {
        SampleMode::Engine(name) => {
            let Some(result) = slot.engine_result(&name) else {
                return Err(format!(
                    "Sample {} has no engine result for mode {name}",
                    slot.index + 1
                ));
            };
            if let Some(ref discovered) = result.discovered_loops {
                // New pipeline: replace data without loop scaling, then set discovered loop points
                if !module.replace_sample_data_raw(
                    slot.index,
                    &result.data,
                    result.length_frames,
                    result.channels,
                    result.sample_rate_hz,
                ) {
                    return Err(format!("Failed to replace sample {}", slot.index + 1));
                }
                let loop_info = crate::openmpt::SampleLoopInfo {
                    normal: discovered.normal,
                    sustain: discovered.sustain,
                };
                if !module.set_sample_loop_points(slot.index, &loop_info) {
                    return Err(format!(
                        "Failed to set loop points for sample {}",
                        slot.index + 1
                    ));
                }
                if !slot.original_effects.is_empty() {
                    crate::remaster::restore_effect_params(module, &slot.original_effects);
                }
                if result.sample_rate_hz != slot.original_rate {
                    crate::remaster::patch_sample_offsets(
                        module,
                        slot.index,
                        slot.original_rate,
                        result.sample_rate_hz,
                    );
                }
            } else {
                // Fallback: use proportional loop scaling
                crate::remaster::apply_sample_replacement(
                    module,
                    slot.index,
                    &result.data,
                    result.length_frames,
                    result.channels,
                    result.sample_rate_hz,
                    slot.original_rate,
                    &slot.original_effects,
                )?;
            }
        }
        SampleMode::Reference48k => {
            if slot.reference_48k.is_none() {
                let data = crate::remaster::resample_audio(
                    &slot.original,
                    slot.original_rate.max(1) as u32,
                    48_000,
                    slot.original_channels.max(1) as usize,
                    crate::remaster::ResampleBoundaryMode::OneShot,
                )?;
                slot.reference_48k = Some(data);
            }
            let data = slot
                .reference_48k
                .as_ref()
                .expect("reference_48k was just populated");
            let length_frames = (data.len() as i64) / (slot.original_channels.max(1) as i64);
            crate::remaster::apply_sample_replacement(
                module,
                slot.index,
                data,
                length_frames,
                slot.original_channels,
                48_000,
                slot.original_rate,
                &slot.original_effects,
            )?;
        }
        SampleMode::Original => {
            crate::remaster::apply_sample_replacement(
                module,
                slot.index,
                &slot.original,
                slot.original_length_frames,
                slot.original_channels,
                slot.original_rate,
                slot.original_rate,
                &slot.original_effects,
            )?;
        }
    }

    slot.mode = target_mode;

    // Drop any stale per-channel state that was copied from the sample before
    // its loop points / rate were updated. Runs on every live swap because
    // Engine↔Engine swaps at 48 kHz still change the discovered-loop points
    // that the mixer reads via chn.nLoopStart/End.
    if is_playing {
        module.refresh_channels_for_sample(slot.index);
    }

    Ok(())
}

/// State for the A/B blind test mode.
struct BlindTest {
    sample_name: String,
    original_data: Vec<f64>,
    ai_data: Vec<f64>,
    ai_is_b: bool, // true = AI is button B
    revealed: bool,
    last_correct: bool,
    correct: u32,
    total: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InterpolationChoice {
    None,
    Linear,
    Cubic,
    CatmullRom,
    Sinc16,
    Aniso64,
}

impl std::fmt::Display for InterpolationChoice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::None => write!(f, "Nearest"),
            Self::Linear => write!(f, "Trilinear"),
            Self::Cubic => write!(f, "Cubic"),
            Self::CatmullRom => write!(f, "Catmull-Rom"),
            Self::Sinc16 => write!(f, "Polyphase"),
            Self::Aniso64 => write!(f, "Aniso-64"),
        }
    }
}

impl InterpolationChoice {
    fn to_filter_length(self) -> i32 {
        match self {
            Self::None => 1,
            Self::Linear => 2,
            Self::Cubic => 4,
            Self::CatmullRom => 5,
            Self::Sinc16 => 16,
            Self::Aniso64 => 64,
        }
    }
}

const INTERPOLATION_CHOICES: &[InterpolationChoice] = &[
    InterpolationChoice::None,
    InterpolationChoice::Linear,
    InterpolationChoice::Cubic,
    InterpolationChoice::CatmullRom,
    InterpolationChoice::Sinc16,
    InterpolationChoice::Aniso64,
];

fn render_save_name(stem: &str, extension: &str, render_rate_hz: u32) -> String {
    let rate = crate::batch::format_rate_khz(render_rate_hz);
    format!("{stem}-Quinlight-Audio-Remastered-{rate}.{extension}")
}

fn default_render_rate(device_rate: u32) -> u32 {
    if device_rate > 0 { device_rate } else { 48_000 }
}

fn default_render_save_name(orig_path: &Path, ext: &str, device_rate: u32) -> String {
    render_save_name(
        &orig_path.file_stem().unwrap_or_default().to_string_lossy(),
        ext,
        default_render_rate(device_rate),
    )
}

pub struct Quinlight {
    player: Arc<Player>,
    player_state: PlayerState,
    module_info: Option<ModuleInfo>,
    loaded_path: Option<PathBuf>,

    // Seek state
    seek_position: f64,
    is_seeking: bool,

    // Remaster
    remaster_engine: RemasterEngine,
    remaster_status: RemasterStatus,
    remaster_notice: Option<String>,
    remaster_log: Vec<String>,
    remaster_rx: Option<crossbeam_channel::Receiver<RemasterStatus>>,
    remaster_result_rx: Option<crossbeam_channel::Receiver<RemasterOutput>>,
    remaster_cancel_flag: Option<Arc<AtomicBool>>,
    remaster_interrupt_action: Option<RemasterInterruptAction>,
    pending_post_cancel_action: Option<RemasterInterruptAction>,
    restore_originals_after_cancel: bool,
    show_install_dialog: bool,
    /// Custom title for the install dialog (e.g. "Install Missing Engines (...)"
    /// when only some engines are missing). None means the default "no engines
    /// installed" title.
    install_dialog_title: Option<String>,
    audio_error: Option<String>,
    show_audio_error_dialog: bool,

    // Settings
    interpolation: InterpolationChoice,
    stereo_separation: i32,
    agc_enabled: bool,
    hrtf_enabled: bool,
    hrtf_mix: i32,
    cleanup_settings: CleanupSettings,

    // Per-sample remaster state
    sample_slots: Vec<SampleSlot>,
    original_linear_slides: bool,

    // Upscale threading mode
    upscale_mode: UpscaleMode,
    // AudioSR DDIM steps (quality/speed: 10=fast, 100=full)
    ddim_steps: u32,
    attract_countdown: f32,
    attract_phase: f32,
    last_tick: std::time::Instant,
    drain_tick_counter: u8,
    // Ctrl-C shutdown flag from signal handler
    shutdown_flag: Arc<AtomicBool>,
    // Per-engine enable flags for GUI-only Quinlight engine selection.
    enable_audiosr: bool,
    enable_lavasr: bool,
    enable_flowhigh: bool,
    enable_apbwe: bool,

    // Archive browsing
    archive_path: Option<PathBuf>,
    archive_entries: Vec<crate::archive::ArchiveEntry>,
    show_archive_picker: bool,

    // Fade-out on close
    closing: bool,
    closing_window: Option<iced::window::Id>,
    shutdown_close_resolution_pending: bool,

    // File size for format badge
    file_size_bytes: u64,
    // Pattern viewer
    show_pattern_viewer: bool,
    /// Cached pattern data: (pattern_index, rows_of_channels: Vec<row: Vec<(formatted, highlight)>>)
    pattern_cache: Option<(i32, Vec<Vec<(String, String)>>)>,

    // Oscilloscope + spectrogram
    waveform: Vec<f64>,
    oscilloscope_gain: f64,
    spectrogram: scope_shader::SpectrogramState,
    last_spectro_push: std::time::Instant,

    // Vinyl groove waveform (rendered from the full song)
    groove_data: Vec<f32>,

    // Keyjazz
    keyjazz_sample: Option<i32>,
    keyjazz_data: Option<Vec<f64>>,

    // Waveform viewer (replaces oscilloscope when Some). The cache persists
    // the rendered geometry so long samples don't rebuild on every tick —
    // invalidated whenever the viewer's input changes (sample selection,
    // mode cycle, module reload).
    selected_waveform_sample: Option<i32>,
    waveform_cache: canvas::Cache,

    // Render progress (FLAC/AAC export)
    render_progress_rx: Option<crossbeam_channel::Receiver<f32>>,
    render_progress_pct: f32,
    render_format_label: &'static str,

    // Blind test
    blind_test: Option<BlindTest>,

    // Tracks which module load the audio thread has processed
    last_loaded_generation: u64,
    active_module_load_request: u64,

    // Viewport and preferred resizable panel heights (pixels)
    viewport_size: iced::Size,
    preferred_panel_heights: PanelHeights,

    // Drag state for panel resizing
    drag_panel: Option<DragPanel>,
    drag_start_y: f32,

    // Animated background
    background: BackgroundState,
}

impl Quinlight {
    pub fn new(
        upscale_mode: UpscaleMode,
        detect_handle: std::thread::JoinHandle<RemasterEngine>,
        shutdown_flag: Arc<AtomicBool>,
        playback_rate: Option<u32>,
    ) -> (Self, Task<Message>) {
        let (player, audio_error) = match Player::new(playback_rate) {
            Ok(p) => (p, None),
            Err(e) => {
                eprintln!("Warning: Audio initialization failed: {e}");
                (Player::dummy(), Some(e))
            }
        };
        // Join the detection thread — it's been running in parallel with Iced init
        let remaster = detect_handle
            .join()
            .unwrap_or_else(|_| RemasterEngine::empty());

        let status = if remaster.is_available() {
            RemasterStatus::Ready
        } else {
            RemasterStatus::Unavailable
        };

        (
            Quinlight {
                player: Arc::new(player),
                player_state: PlayerState::default(),
                module_info: None,
                loaded_path: None,
                seek_position: 0.0,
                is_seeking: false,
                remaster_engine: remaster,
                remaster_status: status.clone(),
                remaster_notice: None,
                remaster_log: Vec::new(),
                remaster_rx: None,
                remaster_result_rx: None,
                remaster_cancel_flag: None,
                remaster_interrupt_action: None,
                pending_post_cancel_action: None,
                restore_originals_after_cancel: false,
                show_install_dialog: matches!(status, RemasterStatus::Unavailable),
                install_dialog_title: None,
                show_audio_error_dialog: audio_error.is_some(),
                audio_error,
                interpolation: InterpolationChoice::Aniso64,
                stereo_separation: crate::openmpt::DEFAULT_STEREO_SEPARATION_PERCENT,
                agc_enabled: crate::openmpt::DEFAULT_AGC_ENABLED,
                hrtf_enabled: true,
                hrtf_mix: 33,
                cleanup_settings: CleanupSettings::default(),
                sample_slots: Vec::new(),
                original_linear_slides: false,
                upscale_mode,
                ddim_steps: 50,
                attract_countdown: 8.0,
                attract_phase: 0.0,
                last_tick: std::time::Instant::now(),
                drain_tick_counter: 0,
                shutdown_flag,
                enable_audiosr: true,
                enable_lavasr: true,
                enable_flowhigh: true,
                enable_apbwe: true,
                archive_path: None,
                archive_entries: Vec::new(),
                show_archive_picker: false,
                closing: false,
                closing_window: None,
                shutdown_close_resolution_pending: false,
                file_size_bytes: 0,
                show_pattern_viewer: true,
                pattern_cache: None,
                waveform: Vec::new(),
                oscilloscope_gain: 1.0,
                spectrogram: scope_shader::SpectrogramState::new(),
                last_spectro_push: std::time::Instant::now(),
                groove_data: Vec::new(),
                keyjazz_sample: None,
                keyjazz_data: None,
                selected_waveform_sample: None,
                waveform_cache: canvas::Cache::default(),
                render_progress_rx: None,
                render_progress_pct: 0.0,
                render_format_label: "",
                blind_test: None,
                last_loaded_generation: 0,
                active_module_load_request: 0,
                viewport_size: initial_window_size(),
                preferred_panel_heights: PanelHeights::new(200.0, 120.0, 300.0),
                drag_panel: None,
                drag_start_y: 0.0,
                background: BackgroundState::new(),
            },
            Task::none(),
        )
    }

    pub fn title(&self) -> String {
        match &self.module_info {
            Some(info) if !info.title.is_empty() => {
                format!(
                    "Quinlight Audio \u{2014} {PRESENTED_BY_KIND_COMPUTERS}: {}",
                    info.title
                )
            }
            _ => format!("Quinlight Audio \u{2014} {PRESENTED_BY_KIND_COMPUTERS}"),
        }
    }

    fn idle_remaster_status(&self) -> RemasterStatus {
        if self.remaster_engine.is_available() {
            RemasterStatus::Ready
        } else {
            RemasterStatus::Unavailable
        }
    }

    fn remaster_is_active(&self) -> bool {
        matches!(
            self.remaster_status,
            RemasterStatus::Processing { .. } | RemasterStatus::Cancelling
        )
    }

    fn remaster_primary_action(&self) -> RemasterPrimaryAction {
        if matches!(self.remaster_status, RemasterStatus::Unavailable) {
            RemasterPrimaryAction::Install
        } else if matches!(self.remaster_status, RemasterStatus::Processing { .. }) {
            RemasterPrimaryAction::Cancel
        } else if matches!(self.remaster_status, RemasterStatus::Cancelling) {
            RemasterPrimaryAction::Cancelling
        } else if matches!(self.remaster_status, RemasterStatus::Complete)
            && self.module_info.is_some()
        {
            RemasterPrimaryAction::Complete
        } else if matches!(
            self.remaster_status,
            RemasterStatus::Ready | RemasterStatus::Cancelled | RemasterStatus::Failed(_)
        ) && self.module_info.is_some()
        {
            RemasterPrimaryAction::Start
        } else {
            RemasterPrimaryAction::Disabled
        }
    }

    fn current_remaster_panel_budget_height(&self) -> f32 {
        remaster_panel_budget_height(
            matches!(self.remaster_status, RemasterStatus::Failed(_)),
            matches!(self.remaster_status, RemasterStatus::Processing { .. }),
            !self.remaster_log.is_empty(),
            self.remaster_notice.is_some(),
        )
    }

    fn resolved_panel_heights(&self) -> PanelHeights {
        resolve_panel_heights(
            self.preferred_panel_heights,
            available_panel_height(
                self.viewport_size.height,
                self.current_remaster_panel_budget_height(),
            ),
        )
    }

    fn sync_linear_slides(&mut self) {
        self.player.with_module(|m| {
            m.set_linear_slides(self.original_linear_slides);
        });
    }

    fn apply_sample_mode_by_slot_index(
        &mut self,
        slot_idx: usize,
        target_mode: SampleMode,
    ) -> Result<(), String> {
        let Some(slot) = self.sample_slots.get(slot_idx) else {
            return Err(format!("Sample slot {slot_idx} is out of range"));
        };
        let slot_index = slot.index;
        let is_playing = self.player_state.status == PlaybackStatus::Playing;
        let player = Arc::clone(&self.player);
        let result = player
            .with_module(|module| {
                apply_sample_mode_to_slot(
                    module,
                    &mut self.sample_slots[slot_idx],
                    target_mode,
                    is_playing,
                )
            })
            .ok_or_else(|| format!("No module loaded while applying sample {}", slot_index + 1))?;
        result.map_err(|err| format!("Sample {}: {err}", slot_index + 1))
    }

    fn reset_loaded_module_state(&mut self) {
        self.module_info = None;
        self.sample_slots.clear();
        self.pattern_cache = None;
        self.keyjazz_sample = None;
        self.keyjazz_data = None;
        self.selected_waveform_sample = None;
        self.waveform_cache.clear();
        self.blind_test = None;
        self.waveform.clear();
        self.spectrogram.clear();
        self.groove_data.clear();
        self.original_linear_slides = false;
        self.remaster_notice = None;
        self.remaster_status = self.idle_remaster_status();
    }

    fn begin_prepared_module_load<F>(
        &mut self,
        context: PendingModuleLoadContext,
        loader: F,
    ) -> Task<Message>
    where
        F: FnOnce() -> Result<crate::player::PreparedModuleLoad, String> + Send + 'static,
    {
        self.active_module_load_request = self.active_module_load_request.wrapping_add(1);
        let request_id = self.active_module_load_request;
        self.loaded_path = None;
        self.file_size_bytes = 0;
        self.reset_loaded_module_state();

        Task::perform(
            async move {
                match loader() {
                    Ok(prepared) => PreparedModuleLoadOutcome::Success(
                        PreparedModuleLoadHandle::new(PreparedModuleLoadPackage {
                            request_id,
                            context,
                            prepared,
                        }),
                    ),
                    Err(error) => PreparedModuleLoadOutcome::Failure { request_id, error },
                }
            },
            Message::PreparedModuleLoadReady,
        )
    }

    fn begin_module_load_from_path(&mut self, path: PathBuf) -> Task<Message> {
        let stereo_separation = crate::openmpt::effective_stereo_separation(
            &path,
            crate::openmpt::DEFAULT_STEREO_SEPARATION_PERCENT,
        );
        let context = PendingModuleLoadContext {
            loaded_path: Some(path.clone()),
            stereo_separation,
        };
        self.begin_prepared_module_load(context, move || {
            crate::player::prepare_module_load_from_path(&path)
        })
    }

    fn begin_module_load_from_archive_bytes(
        &mut self,
        entry_path: String,
        data: Vec<u8>,
    ) -> Task<Message> {
        let stereo_separation = crate::openmpt::effective_stereo_separation(
            Path::new(&entry_path),
            crate::openmpt::DEFAULT_STEREO_SEPARATION_PERCENT,
        );
        let context = PendingModuleLoadContext {
            loaded_path: Some(PathBuf::from(&entry_path)),
            stereo_separation,
        };
        self.begin_prepared_module_load(context, move || {
            crate::player::prepare_module_load_from_bytes(data)
        })
    }

    fn apply_prepared_module_load(
        &mut self,
        handle: PreparedModuleLoadHandle,
    ) -> Option<PendingGrooveRender> {
        let package = handle.take()?;
        if package.request_id != self.active_module_load_request {
            return None;
        }

        let PreparedModuleLoadPackage {
            request_id,
            context,
            prepared,
        } = package;
        let interpolation_filter = self.interpolation.to_filter_length();
        let pending_groove_render = PendingGrooveRender {
            request_id,
            file_data: prepared.clone_file_data(),
            stereo_separation: context.stereo_separation,
            interpolation_filter: InterpolationChoice::Linear.to_filter_length(), // Trilinear — fast enough for groove viz
        };
        self.loaded_path = context.loaded_path;
        self.file_size_bytes = prepared.file_size_bytes();
        self.stereo_separation = context.stereo_separation;
        self.player.install_prepared_load_with_settings(
            prepared,
            self.stereo_separation,
            interpolation_filter,
            self.agc_enabled,
        );
        Some(pending_groove_render)
    }

    fn spawn_groove_render_task(pending: PendingGrooveRender) -> Task<Message> {
        let PendingGrooveRender {
            request_id,
            file_data,
            stereo_separation,
            interpolation_filter,
        } = pending;
        Task::perform(
            async move {
                (
                    request_id,
                    vinyl_shader::render_groove_data(
                        &file_data,
                        stereo_separation,
                        interpolation_filter,
                    ),
                )
            },
            |(request_id, result)| match result {
                Ok(data) => Message::GrooveDataReady { request_id, data },
                Err(e) => {
                    eprintln!("Groove render failed: {e}");
                    Message::GrooveDataReady {
                        request_id,
                        data: Vec::new(),
                    }
                }
            },
        )
    }

    fn has_pending_close_request(&self) -> bool {
        matches!(
            self.remaster_interrupt_action.as_ref(),
            Some(RemasterInterruptAction::Close(_))
        ) || matches!(
            self.pending_post_cancel_action.as_ref(),
            Some(RemasterInterruptAction::Close(_))
        )
    }

    fn resolve_oldest_window_for_close(&mut self) -> Task<Message> {
        self.shutdown_close_resolution_pending = true;
        iced::window::get_oldest().map(Message::ResolvedOldestWindow)
    }

    fn begin_close_sequence(&mut self, id: iced::window::Id) -> Task<Message> {
        self.closing = true;
        self.closing_window = Some(id);
        self.player.set_volume(0.0);
        Task::perform(
            async {
                // Let a few silent audio buffers play before closing
                std::thread::sleep(std::time::Duration::from_millis(150));
            },
            |_| Message::FadeOutDone,
        )
    }

    fn begin_remaster_cancellation(
        &mut self,
        pending_action: Option<RemasterInterruptAction>,
        restore_originals: bool,
    ) {
        let Some(cancel_flag) = &self.remaster_cancel_flag else {
            return;
        };

        cancel_flag.store(true, Ordering::Relaxed);
        self.remaster_rx = None;
        self.remaster_result_rx = None;
        self.remaster_interrupt_action = None;
        self.pending_post_cancel_action = pending_action;
        self.restore_originals_after_cancel |= restore_originals;
        self.remaster_notice = None;
        self.remaster_status = RemasterStatus::Cancelling;
    }

    fn restore_original_samples_after_cancel(&mut self) {
        let slots = std::mem::take(&mut self.sample_slots);
        if slots.is_empty() {
            self.blind_test = None;
            return;
        }

        self.player.with_module(|m| {
            for slot in &slots {
                m.replace_sample_data(
                    slot.index,
                    &slot.original,
                    slot.original_length_frames,
                    slot.original_channels,
                    slot.original_rate,
                );
            }
            for slot in &slots {
                if !slot.original_effects.is_empty() {
                    crate::remaster::restore_effect_params(m, &slot.original_effects);
                }
            }
        });
        self.module_info = self.player.with_module(|m| m.info());
        self.sync_linear_slides();
        self.blind_test = None;
    }

    fn finalize_cancelled_remaster(&mut self) -> Task<Message> {
        let pending_action = self.pending_post_cancel_action.take();
        let restore_originals = std::mem::take(&mut self.restore_originals_after_cancel);

        self.remaster_cancel_flag = None;
        self.remaster_rx = None;
        self.remaster_result_rx = None;
        self.remaster_interrupt_action = None;

        if restore_originals {
            self.restore_original_samples_after_cancel();
            self.remaster_notice = Some("Remaster cancelled".into());
        }

        self.remaster_status = self.idle_remaster_status();

        match pending_action {
            Some(RemasterInterruptAction::Close(id)) => self.begin_close_sequence(id),
            Some(RemasterInterruptAction::LoadFile(path)) => {
                self.update(Message::FileSelected(Some(path)))
            }
            None => Task::none(),
        }
    }

    fn take_remaster_statuses(&self) -> Vec<RemasterStatus> {
        let mut statuses = Vec::new();
        if let Some(rx) = &self.remaster_rx {
            while let Ok(status) = rx.try_recv() {
                statuses.push(status);
            }
        }
        statuses
    }

    fn take_remaster_results(&self) -> Vec<RemasterOutput> {
        let mut results = Vec::new();
        if let Some(rx) = &self.remaster_result_rx {
            while let Ok(result) = rx.try_recv() {
                results.push(result);
            }
        }
        results
    }

    fn apply_remaster_status(&mut self, status: RemasterStatus) {
        match &status {
            RemasterStatus::Log(line) => {
                self.remaster_log.push(line.clone());
            }
            RemasterStatus::EngineProgress {
                sample_index,
                engines_done,
                engines_total,
            } => {
                if let Some(slot) = self
                    .sample_slots
                    .iter_mut()
                    .find(|slot| slot.index == *sample_index)
                {
                    slot.engines_done = *engines_done;
                    slot.engines_total = *engines_total;
                }
            }
            _ => {
                self.remaster_status = status;
            }
        }
    }

    fn apply_sample_result(&mut self, output: RemasterOutput) -> bool {
        let (result, is_final) = match output {
            RemasterOutput::Candidate(result) => (result, false),
            RemasterOutput::Final(result) => (result, true),
        };
        let Some(slot_idx) = self
            .sample_slots
            .iter()
            .position(|slot| slot.index == result.index)
        else {
            return false;
        };

        if result.data.is_empty() {
            return false;
        }

        let original_rate = self.sample_slots[slot_idx].original_rate;
        let needs_effect_snapshot = self.sample_slots[slot_idx].original_effects.is_empty();
        let saved_effects = if needs_effect_snapshot {
            self.player
                .with_module(|m| crate::remaster::save_effect_params(m, result.index))
                .unwrap_or_default()
        } else {
            Vec::new()
        };
        let target_mode = {
            let slot = &mut self.sample_slots[slot_idx];
            if is_final {
                slot.apply_final_result(&result, saved_effects)
            } else {
                slot.apply_candidate_result(&result, saved_effects)
            }
        };
        let Some(target_mode) = target_mode else {
            return false;
        };

        self.apply_sample_mode_by_slot_index(slot_idx, target_mode)
            .map(|_| true)
            .unwrap_or_else(|err| {
                eprintln!(
                    "Failed to apply remaster result for sample {} at {}Hz: {err}",
                    result.index + 1,
                    original_rate
                );
                false
            })
    }

    fn drain_remaster_updates(&mut self) {
        let mut updated_preview = false;
        let results = self.take_remaster_results();
        // Uncomment for debugging remaster result delivery:
        // if !results.is_empty() {
        //     eprintln!(
        //         "[drain] received {} remaster result(s), {} sample slots",
        //         results.len(),
        //         self.sample_slots.len()
        //     );
        // }
        for result in results {
            // let (idx, engine, len, is_final) = match &result {
            //     RemasterOutput::Candidate(r) => (r.index, &r.engine_name, r.data.len(), false),
            //     RemasterOutput::Final(r) => (r.index, &r.engine_name, r.data.len(), true),
            // };
            // eprintln!(
            //     "[drain] sample #{} engine={} data_len={} final={} slot_match={}",
            //     idx + 1, engine, len, is_final,
            //     self.sample_slots.iter().any(|s| s.index == idx)
            // );
            let applied = self.apply_sample_result(result);
            // eprintln!("[drain]   → applied={applied}");
            updated_preview |= applied;
        }
        for status in self.take_remaster_statuses() {
            self.apply_remaster_status(status);
        }
        for slot in &mut self.sample_slots {
            slot.refresh_failed_state();
        }

        if updated_preview {
            self.module_info = self.player.with_module(|m| m.info());
            self.sync_linear_slides();
        }
    }

    pub fn update(&mut self, message: Message) -> Task<Message> {
        match message {
            Message::OpenFileDialog => {
                return Task::perform(
                    async {
                        rfd::AsyncFileDialog::new()
                            .add_filter(
                                "Modules & archives",
                                &[
                                    "mod", "s3m", "xm", "it", "mptm", "stm", "nst", "m15", "stk",
                                    "wow", "ult", "669", "mtm", "med", "far", "mdl", "ams", "dsm",
                                    "amf", "okt", "dmf", "ptm", "psm", "mt2", "dbm", "digi", "imf",
                                    "j2b", "gdm", "umx", "plm", "mo3", "xpk", "ppm", "mmcmp",
                                    "zip", "7z", "rar", "tar", "gz", "tgz", "bz2", "xz", "lha",
                                    "lzh",
                                ],
                            )
                            .add_filter("All files", &["*"])
                            .pick_file()
                            .await
                            .map(|f| f.path().to_path_buf())
                    },
                    Message::FileSelected,
                );
            }
            Message::FileSelected(Some(path)) => {
                if matches!(self.remaster_status, RemasterStatus::Processing { .. }) {
                    self.remaster_interrupt_action = Some(RemasterInterruptAction::LoadFile(path));
                    return Task::none();
                }
                if matches!(self.remaster_status, RemasterStatus::Cancelling) {
                    self.pending_post_cancel_action = Some(RemasterInterruptAction::LoadFile(path));
                    return Task::none();
                }
                if crate::archive::is_archive(&path) {
                    self.archive_path = Some(path.clone());
                    return Task::perform(
                        async move { crate::archive::list_modules_in_archive(&path) },
                        Message::ArchiveContents,
                    );
                }
                return self.begin_module_load_from_path(path);
            }
            Message::FileSelected(None) => {}
            Message::Play => self.player.send(PlayerCommand::Play),
            Message::Pause => self.player.send(PlayerCommand::Pause),
            Message::Stop => self.player.send(PlayerCommand::Stop),
            Message::SeekChanged(pos) => {
                self.is_seeking = true;
                self.seek_position = pos;
            }
            Message::SeekReleased => {
                self.is_seeking = false;
                self.player.send(PlayerCommand::Seek(self.seek_position));
            }
            Message::Tick => {
                let now = std::time::Instant::now();
                let dt = now.duration_since(self.last_tick).as_secs_f32().min(0.05);
                self.last_tick = now;

                // Auto-grow audio buffer on underrun
                if let Some(new_frames) = self.player.check_and_grow_buffer() {
                    self.remaster_notice = Some(format!(
                        "Audio buffer underrun — increased to {new_frames} frames"
                    ));
                }

                // Check Ctrl-C shutdown flag
                if self.shutdown_flag.load(Ordering::Relaxed)
                    && !self.closing
                    && !self.shutdown_close_resolution_pending
                    && !self.has_pending_close_request()
                {
                    return self.resolve_oldest_window_for_close();
                }
                self.background
                    .tick(self.player_state.status == PlaybackStatus::Playing, dt);

                // Attract-mode pulse on the Remaster button
                if matches!(self.remaster_primary_action(), RemasterPrimaryAction::Start) {
                    if self.attract_phase > 0.0 {
                        self.attract_phase += dt / 0.4; // 0.4s pulse duration
                        if self.attract_phase >= 1.0 {
                            self.attract_phase = 0.0;
                            self.attract_countdown =
                                5.0 + (self.background.time * 7.3).fract() as f32 * 15.0;
                        }
                    } else {
                        self.attract_countdown -= dt;
                        if self.attract_countdown <= 0.0 {
                            self.attract_phase = 0.001;
                        }
                    }
                } else {
                    self.attract_phase = 0.0;
                }

                // Skip inner-mutex operations while render export holds the lock,
                // so the GUI thread never blocks on the mutex during synthesis.
                let rendering = self.render_progress_rx.is_some();
                if !rendering {
                    self.player.refresh_visual_state();
                }
                self.player_state = self.player.state();
                self.waveform = self.player.waveform();
                // Push spectrogram at 250 Hz — one column per frame on 240 Hz displays
                const SPECTRO_INTERVAL_MS: u64 = 4;
                if self.player_state.status == PlaybackStatus::Playing
                    && !self.waveform.is_empty()
                    && now.duration_since(self.last_spectro_push).as_millis()
                        >= SPECTRO_INTERVAL_MS as u128
                {
                    self.spectrogram.push_fft(&self.waveform);
                    self.last_spectro_push = now;
                }
                if let Some(new_cols) = self.spectrogram.poll_desired_cols(now) {
                    self.spectrogram.resize(new_cols);
                }
                {
                    let peak = self.waveform.iter().map(|s| s.abs()).fold(0.0f64, f64::max);
                    let target_gain = (if peak > 1e-6 { 0.85 / peak } else { 1.0 }).min(20.0);
                    if target_gain < self.oscilloscope_gain {
                        // Instant attack: prevent transient clipping
                        self.oscilloscope_gain = target_gain;
                    } else {
                        // Slewed release (~400ms), dt-scaled
                        let release = (10.0 * dt as f64).min(1.0);
                        self.oscilloscope_gain += (target_gain - self.oscilloscope_gain) * release;
                    }
                }
                if !self.is_seeking {
                    let dur = self.player_state.duration_seconds;
                    self.seek_position = if dur > 0.0 {
                        self.player_state.position_seconds % dur
                    } else {
                        0.0
                    };
                }
                // Load module info once we have a playing module
                if !rendering
                    && self.module_info.is_none()
                    && self.player_state.status != PlaybackStatus::Stopped
                    && self.player_state.load_generation != self.last_loaded_generation
                {
                    if let Some((mut info, linear_slides)) = self
                        .player
                        .with_module(|m| (m.info(), m.linear_slides_enabled()))
                    {
                        info.file_size_bytes = self.file_size_bytes;
                        self.background.set_label(&info);
                        self.module_info = Some(info);
                        self.original_linear_slides = linear_slides;
                        self.last_loaded_generation = self.player_state.load_generation;
                    }
                }
                // Poll render export progress (fast, no mutex)
                if let Some(rx) = &self.render_progress_rx {
                    while let Ok(pct) = rx.try_recv() {
                        self.render_progress_pct = pct;
                    }
                }

                // Throttle expensive module-lock operations to ~15 Hz to avoid
                // contention with the audio callback (which also locks inner)
                self.drain_tick_counter = self.drain_tick_counter.wrapping_add(1);
                if !rendering && self.drain_tick_counter % 8 == 0 {
                    // Refresh pattern cache when pattern changes
                    if self.show_pattern_viewer
                        && self.player_state.status == PlaybackStatus::Playing
                    {
                        let pat = self.player_state.current_pattern;
                        let need_refresh = match &self.pattern_cache {
                            Some((cached_pat, _)) => *cached_pat != pat,
                            None => true,
                        };
                        if need_refresh {
                            if let Some(info) = &self.module_info {
                                let num_ch = info.num_channels;
                                self.pattern_cache = self.player.with_module(|m| {
                                    let num_rows = m.pattern_num_rows(pat);
                                    let mut rows = Vec::with_capacity(num_rows as usize);
                                    for r in 0..num_rows {
                                        let mut channels = Vec::with_capacity(num_ch as usize);
                                        for ch in 0..num_ch {
                                            let fmt = m.format_pattern_row_channel(pat, r, ch);
                                            let hl = m.highlight_pattern_row_channel(pat, r, ch);
                                            channels.push((fmt, hl));
                                        }
                                        rows.push(channels);
                                    }
                                    (pat, rows)
                                });
                            }
                        }
                    }
                    self.drain_remaster_updates();
                }
            }
            Message::ResolvedOldestWindow(window_id) => {
                self.shutdown_close_resolution_pending = false;
                if let Some(id) = window_id {
                    return self.update(Message::CloseRequested(id));
                }
            }
            Message::WindowResized(size) => {
                self.viewport_size = size;
            }
            Message::CancelRemaster => {
                if matches!(self.remaster_status, RemasterStatus::Processing { .. }) {
                    self.begin_remaster_cancellation(None, true);
                }
            }
            Message::StartRemaster => {
                if self.remaster_is_active() {
                    return Task::none();
                }
                self.remaster_notice = None;
                self.remaster_log.clear();
                // Fast: just read raw sample data under brief module lock
                let raw_samples = match self
                    .player
                    .with_module(|m| crate::remaster::read_raw_samples(m))
                {
                    Some(samples) if !samples.is_empty() => samples,
                    _ => {
                        self.remaster_status = RemasterStatus::Failed(
                            "No module loaded or no eligible samples".into(),
                        );
                        return Task::none();
                    }
                };
                self.original_linear_slides = self
                    .player
                    .with_module(|m| m.linear_slides_enabled())
                    .unwrap_or(false);

                let enabled_engines = enabled_engine_names(
                    self.enable_audiosr,
                    self.enable_lavasr,
                    self.enable_flowhigh,
                    self.enable_apbwe,
                );
                if enabled_engines.is_empty() {
                    self.remaster_status =
                        RemasterStatus::Failed("Enable at least one AI engine.".into());
                    return Task::none();
                }
                let cleanup_settings = self.cleanup_settings;

                // Build sample slots for original/engine mode toggling.
                self.sample_slots = raw_samples
                    .iter()
                    .map(|o| SampleSlot {
                        index: o.index,
                        original: o.data.clone(),
                        original_rate: o.rate,
                        original_channels: o.channels,
                        original_length_frames: o.source_length_frames,
                        loop_info: o.loop_info,
                        engine_results: Vec::new(),
                        quinlight_result: None,
                        quinlight_original_fallback: false,
                        mode: SampleMode::Original,
                        failed: false,
                        original_effects: Vec::new(),
                        engines_done: 0,
                        engines_total: self
                            .remaster_engine
                            .eligible_enabled_engine_count_for_rate(o.rate, &enabled_engines),
                        muted: false,
                        reference_48k: None,
                    })
                    .collect();

                self.remaster_status = RemasterStatus::Processing {
                    current: 0,
                    total: 0,
                    sample_name: "Starting...".into(),
                };

                let cancel_flag = Arc::new(AtomicBool::new(false));
                let (progress_tx, progress_rx) = crossbeam_channel::unbounded();
                let (result_tx, result_rx) = crossbeam_channel::unbounded();
                self.remaster_cancel_flag = Some(cancel_flag.clone());
                self.remaster_rx = Some(progress_rx);
                self.remaster_result_rx = Some(result_rx);
                self.remaster_interrupt_action = None;
                self.pending_post_cancel_action = None;
                self.restore_originals_after_cancel = false;
                self.remaster_notice = None;

                let mode = self.upscale_mode;
                let ddim_steps = self.ddim_steps;
                let min_dur = self.remaster_engine.min_duration_secs();
                return Task::perform(
                    async move {
                        // Engine detection + heavy processing runs off GUI thread
                        let engine = RemasterEngine::detect_with_fallback(&enabled_engines);
                        let work_dir = match tempfile::tempdir() {
                            Ok(d) => d,
                            Err(e) => return RemasterStatus::Failed(e.to_string()),
                        };
                        let jobs = match crate::remaster::extract_sample_jobs(
                            &raw_samples,
                            work_dir.path(),
                            min_dur,
                            cleanup_settings,
                            cancel_flag.as_ref(),
                        ) {
                            Ok(j) => j,
                            Err(e) if crate::remaster::is_cancelled_error(&e) => {
                                return RemasterStatus::Cancelled;
                            }
                            Err(e) => return RemasterStatus::Failed(e),
                        };
                        match engine.remaster_samples(
                            jobs,
                            work_dir,
                            &progress_tx,
                            &result_tx,
                            mode,
                            cancel_flag.as_ref(),
                            ddim_steps,
                            true,
                            false,
                        ) {
                            Ok(()) if cancel_flag.load(Ordering::Relaxed) => {
                                RemasterStatus::Cancelled
                            }
                            Ok(()) => RemasterStatus::Complete,
                            Err(e) if crate::remaster::is_cancelled_error(&e) => {
                                RemasterStatus::Cancelled
                            }
                            Err(e) => RemasterStatus::Failed(e),
                        }
                    },
                    Message::RemasterUpdate,
                );
            }
            Message::RemasterUpdate(status) => {
                self.drain_remaster_updates();
                self.remaster_notice = None;
                if matches!(status, RemasterStatus::Cancelled) {
                    return self.finalize_cancelled_remaster();
                }
                self.apply_remaster_status(status);
                for slot in &mut self.sample_slots {
                    slot.refresh_failed_state();
                }
                self.remaster_cancel_flag = None;
                self.pending_post_cancel_action = None;
                self.remaster_interrupt_action = None;
                self.restore_originals_after_cancel = false;
                self.remaster_rx = None;
                self.remaster_result_rx = None;
            }
            Message::ConfirmRemasterInterrupt => {
                if let Some(action) = self.remaster_interrupt_action.take() {
                    if matches!(self.remaster_status, RemasterStatus::Cancelling) {
                        self.pending_post_cancel_action = Some(action);
                    } else {
                        self.begin_remaster_cancellation(Some(action), false);
                    }
                }
            }
            Message::DismissRemasterInterrupt => {
                self.remaster_interrupt_action = None;
            }
            Message::ShowInstallDialog => {
                self.install_dialog_title = None;
                self.show_install_dialog = true;
            }
            Message::ShowInstallMissing => {
                if let Some(names) = self.remaster_engine.missing_engine_names() {
                    self.install_dialog_title =
                        Some(format!("Install Missing Engines ({})", names.join(", ")));
                    self.show_install_dialog = true;
                }
            }
            Message::RenderFlac => {
                return self.start_render_export("flac", "FLAC");
            }
            Message::RenderFlacComplete(result) => {
                self.finish_render_export(result);
            }
            Message::RenderAac => {
                return self.start_render_export("m4a", "M4A");
            }
            Message::RenderAacComplete(result) => {
                self.finish_render_export(result);
            }
            Message::DismissInstallDialog => {
                self.show_install_dialog = false;
                self.install_dialog_title = None;
            }
            Message::CopyInstallCommand => {
                return iced::clipboard::write(crate::remaster::INSTALL_COMMAND.to_string());
            }
            Message::CopyRemasterError => {
                if let RemasterStatus::Failed(e) = &self.remaster_status {
                    return iced::clipboard::write(e.clone());
                }
            }
            Message::DismissAudioError => {
                self.show_audio_error_dialog = false;
            }
            Message::SetInterpolation(choice) => {
                self.interpolation = choice;
                self.player
                    .send(PlayerCommand::SetInterpolation(choice.to_filter_length()));
            }
            Message::SetStereoSeparation(percent) => {
                self.stereo_separation = percent;
                self.player
                    .send(PlayerCommand::SetStereoSeparation(percent));
            }
            Message::SetAgcEnabled(enabled) => {
                self.agc_enabled = enabled;
                self.player.send(PlayerCommand::SetAgcEnabled(enabled));
            }
            Message::SetHrtfEnabled(enabled) => {
                self.hrtf_enabled = enabled;
                self.player.send(PlayerCommand::SetHrtfEnabled(enabled));
            }
            Message::SetHrtfMix(percent) => {
                self.hrtf_mix = percent;
                self.player.send(PlayerCommand::SetHrtfMix(percent));
            }
            Message::SetCleanupMode(mode) => {
                self.cleanup_settings.mode = mode;
            }
            Message::SetCleanupEngineVersion(engine_version) => {
                self.cleanup_settings.engine_version = engine_version;
            }
            Message::ToggleEngine(ref name, enabled) => match name.as_str() {
                "AudioSR" => self.enable_audiosr = enabled,
                "LavaSR" => self.enable_lavasr = enabled,
                "FLowHigh" => self.enable_flowhigh = enabled,
                "AP-BWE" => self.enable_apbwe = enabled,
                _ => {}
            },
            Message::SetDdimSteps(steps) => {
                self.ddim_steps = steps;
            }
            Message::ClearSongCache => {
                let deleted = self
                    .player
                    .with_module(crate::remaster::clear_cache_for_module)
                    .unwrap_or(0);
                self.remaster_notice = Some(if deleted > 0 {
                    format!(
                        "Cleared {deleted} cached sample{}",
                        if deleted == 1 { "" } else { "s" }
                    )
                } else {
                    "No cached samples found for this song".into()
                });
            }
            Message::CycleSampleMode(index) => {
                if let Some(slot_idx) = self.sample_slots.iter().position(|s| s.index == index) {
                    let new_mode = self.sample_slots[slot_idx].next_mode();
                    if let Err(err) = self.apply_sample_mode_by_slot_index(slot_idx, new_mode) {
                        eprintln!(
                            "Failed to cycle sample mode for sample {}: {err}",
                            index + 1
                        );
                    }
                    self.module_info = self.player.with_module(|m| m.info());
                    self.sync_linear_slides();
                    if self.selected_waveform_sample == Some(index) {
                        self.waveform_cache.clear();
                    }
                }
            }
            Message::CycleAllSampleModes => {
                // Determine the global target mode from the first slot with results
                let target_mode = {
                    let first = self.sample_slots.iter().find(|s| s.has_engine_results());
                    first.map(SampleSlot::next_mode)
                };
                if let Some(target) = target_mode {
                    let slot_count = self.sample_slots.len();
                    for slot_idx in 0..slot_count {
                        let can_use = {
                            let slot = &self.sample_slots[slot_idx];
                            if !slot.has_engine_results() {
                                false
                            } else {
                                match &target {
                                    SampleMode::Engine(name) => slot.engine_result(name).is_some(),
                                    _ => true,
                                }
                            }
                        };
                        if !can_use {
                            continue;
                        }
                        if let Err(err) =
                            self.apply_sample_mode_by_slot_index(slot_idx, target.clone())
                        {
                            let sample_index = self.sample_slots[slot_idx].index;
                            eprintln!(
                                "Failed to cycle all sample modes for sample {}: {err}",
                                sample_index + 1
                            );
                        }
                    }
                    self.module_info = self.player.with_module(|m| m.info());
                    self.sync_linear_slides();
                    self.waveform_cache.clear();
                }
            }
            Message::ToggleSampleMute(index) => {
                if let Some(slot_idx) = self.sample_slots.iter().position(|s| s.index == index) {
                    let muted = !self.sample_slots[slot_idx].muted;
                    self.sample_slots[slot_idx].muted = muted;
                    if muted {
                        let _ = self.player.with_module(|module| {
                            mute_sample(module, &self.sample_slots[slot_idx])
                        });
                    } else {
                        let mode = self.sample_slots[slot_idx].mode.clone();
                        if let Err(err) = self.apply_sample_mode_by_slot_index(slot_idx, mode) {
                            eprintln!("Failed to unmute sample {}: {err}", index + 1);
                        }
                    }
                }
            }
            Message::SoloSample(index) => {
                let already_soloed = self.sample_slots.iter().all(|sl| {
                    if sl.index == index {
                        !sl.muted
                    } else {
                        sl.muted
                    }
                });
                let slot_count = self.sample_slots.len();
                if already_soloed {
                    // Unsolo: unmute everything.
                    for slot_idx in 0..slot_count {
                        if self.sample_slots[slot_idx].muted {
                            self.sample_slots[slot_idx].muted = false;
                            let mode = self.sample_slots[slot_idx].mode.clone();
                            if let Err(err) = self.apply_sample_mode_by_slot_index(slot_idx, mode) {
                                let si = self.sample_slots[slot_idx].index;
                                eprintln!("Failed to unmute sample {}: {err}", si + 1);
                            }
                        }
                    }
                } else {
                    for slot_idx in 0..slot_count {
                        let is_target = self.sample_slots[slot_idx].index == index;
                        let was_muted = self.sample_slots[slot_idx].muted;
                        self.sample_slots[slot_idx].muted = !is_target;
                        if is_target && was_muted {
                            let mode = self.sample_slots[slot_idx].mode.clone();
                            if let Err(err) = self.apply_sample_mode_by_slot_index(slot_idx, mode) {
                                eprintln!("Failed to unmute sample {}: {err}", index + 1);
                            }
                        } else if !is_target && !was_muted {
                            let _ = self.player.with_module(|module| {
                                mute_sample(module, &self.sample_slots[slot_idx])
                            });
                        }
                    }
                }
            }
            Message::ArchiveContents(Ok(entries)) => {
                let Some(archive) = self.archive_path.clone() else {
                    self.remaster_status = RemasterStatus::Failed("No archive path set".into());
                    return Task::none();
                };
                if entries.is_empty() {
                    self.remaster_status =
                        RemasterStatus::Failed("No module files found in archive".into());
                } else if entries.len() == 1 {
                    // Single module — extract directly, skip picker
                    let entry_path = entries[0].path.clone();
                    return Task::perform(
                        async move {
                            crate::archive::extract_from_archive(&archive, &entry_path)
                                .map(|data| ExtractedArchiveModule { entry_path, data })
                        },
                        Message::ArchiveExtracted,
                    );
                } else {
                    self.archive_entries = entries;
                    self.show_archive_picker = true;
                }
            }
            Message::ArchiveContents(Err(e)) => {
                self.remaster_status = RemasterStatus::Failed(e);
            }
            Message::ArchiveSelect(index) => {
                self.show_archive_picker = false;
                let Some(archive) = self.archive_path.clone() else {
                    return Task::none();
                };
                let Some(entry) = self.archive_entries.get(index) else {
                    return Task::none();
                };
                let entry_path = entry.path.clone();
                return Task::perform(
                    async move {
                        crate::archive::extract_from_archive(&archive, &entry_path)
                            .map(|data| ExtractedArchiveModule { entry_path, data })
                    },
                    Message::ArchiveExtracted,
                );
            }
            Message::ArchiveExtracted(Ok(extracted)) => {
                return self
                    .begin_module_load_from_archive_bytes(extracted.entry_path, extracted.data);
            }
            Message::ArchiveExtracted(Err(e)) => {
                self.remaster_status = RemasterStatus::Failed(e);
            }
            Message::PreparedModuleLoadReady(PreparedModuleLoadOutcome::Success(handle)) => {
                if let Some(pending) = self.apply_prepared_module_load(handle) {
                    return Self::spawn_groove_render_task(pending);
                }
            }
            Message::GrooveDataReady { request_id, data } => {
                if request_id == self.active_module_load_request {
                    self.groove_data = data;
                }
            }
            Message::PreparedModuleLoadReady(PreparedModuleLoadOutcome::Failure {
                request_id,
                error,
            }) => {
                if request_id == self.active_module_load_request {
                    self.remaster_status = RemasterStatus::Failed(error);
                }
            }
            Message::DismissArchivePicker => {
                self.show_archive_picker = false;
            }
            Message::CloseRequested(id) => {
                if self.closing {
                    return Task::none();
                }
                if matches!(self.remaster_status, RemasterStatus::Processing { .. }) {
                    self.remaster_interrupt_action = Some(RemasterInterruptAction::Close(id));
                    return Task::none();
                }
                if matches!(self.remaster_status, RemasterStatus::Cancelling) {
                    self.pending_post_cancel_action = Some(RemasterInterruptAction::Close(id));
                    return Task::none();
                }
                return self.begin_close_sequence(id);
            }
            Message::FadeOutDone => {
                if let Some(_id) = self.closing_window {
                    // Extra 100ms so the audio callback flushes
                    // silent samples before the stream is dropped
                    return Task::perform(
                        async {
                            std::thread::sleep(std::time::Duration::from_millis(100));
                        },
                        |_| Message::FinalClose,
                    );
                }
            }
            Message::FinalClose => {
                if let Some(id) = self.closing_window {
                    return iced::window::close(id);
                }
            }
            Message::SelectKeyjazzSample(index) => {
                self.keyjazz_sample = Some(index);
                self.selected_waveform_sample = if self.selected_waveform_sample == Some(index) {
                    None
                } else {
                    Some(index)
                };
                self.waveform_cache.clear();
                // Extract and cache sample data (convert stereo to mono)
                self.keyjazz_data = self
                    .player
                    .with_module(|m| {
                        let data = m.read_sample_data(index)?;
                        let channels = m.sample_channels(index);
                        if channels == 2 {
                            // Average L+R to mono
                            let mono: Vec<f64> =
                                data.chunks(2).map(|c| (c[0] + c[1]) * 0.5).collect();
                            Some(mono)
                        } else {
                            Some(data)
                        }
                    })
                    .flatten();
            }
            Message::KeyPressed(key, modifiers) => {
                use iced::keyboard::{Key, key::Named};
                match key {
                    Key::Named(Named::Space) => {
                        if self.player_state.status == PlaybackStatus::Playing {
                            self.player.send(PlayerCommand::Pause);
                        } else {
                            self.player.send(PlayerCommand::Play);
                        }
                    }
                    Key::Named(Named::ArrowLeft) => {
                        let pos = (self.player_state.position_seconds - 5.0).max(0.0);
                        self.player.send(PlayerCommand::Seek(pos));
                        self.seek_position = pos;
                    }
                    Key::Named(Named::ArrowRight) => {
                        let pos = self.player_state.position_seconds + 5.0;
                        self.player.send(PlayerCommand::Seek(pos));
                        self.seek_position = pos;
                    }
                    Key::Named(Named::ArrowUp) => {
                        self.stereo_separation = (self.stereo_separation + 5).min(100);
                        self.player
                            .send(PlayerCommand::SetStereoSeparation(self.stereo_separation));
                    }
                    Key::Named(Named::ArrowDown) => {
                        self.stereo_separation = (self.stereo_separation - 5).max(0);
                        self.player
                            .send(PlayerCommand::SetStereoSeparation(self.stereo_separation));
                    }
                    Key::Character(ref c) if c.as_str() == "o" && modifiers.command() => {
                        return self.update(Message::OpenFileDialog);
                    }
                    Key::Character(ref c) if !modifiers.command() => {
                        // Keyjazz: map keyboard to notes
                        if let Some(note) = key_to_note(c.as_str()) {
                            if let Some(ref data) = self.keyjazz_data {
                                let rate_ratio = 2.0_f64.powf((note as f64 - 60.0) / 12.0);
                                self.player.send(PlayerCommand::PlaySample {
                                    data: data.clone(),
                                    rate_ratio,
                                });
                            }
                        }
                    }
                    _ => {}
                }
            }
            Message::StartBlindTest => {
                // Find remastered samples with both original and engine data
                let candidate_indices: Vec<usize> = self
                    .sample_slots
                    .iter()
                    .enumerate()
                    .filter(|(_, sl)| !sl.original.is_empty() && sl.has_engine_results())
                    .map(|(i, _)| i)
                    .collect();
                if !candidate_indices.is_empty() {
                    // Pick a random candidate using time-based seed
                    let seed = std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .map(|d| d.as_nanos() as usize)
                        .unwrap_or(0);
                    let idx = candidate_indices[seed % candidate_indices.len()];
                    let ai_is_b = seed % 2 == 0;
                    let slot = &self.sample_slots[idx];
                    let name = self
                        .module_info
                        .as_ref()
                        .and_then(|info| info.samples.iter().find(|s| s.index == slot.index))
                        .map(|s| format!("#{} {}", s.index + 1, s.name))
                        .unwrap_or_else(|| format!("Sample #{}", slot.index + 1));
                    let Some(ai) = slot.blind_test_data() else {
                        return Task::none();
                    };
                    let original = match slot.blind_test_original_data() {
                        Ok(data) => data,
                        Err(_) => return Task::none(),
                    };
                    let (correct, total) = self
                        .blind_test
                        .as_ref()
                        .map(|bt| (bt.correct, bt.total))
                        .unwrap_or((0, 0));
                    self.blind_test = Some(BlindTest {
                        sample_name: name,
                        original_data: original,
                        ai_data: ai,
                        ai_is_b,
                        revealed: false,
                        last_correct: false,
                        correct,
                        total,
                    });
                }
            }
            Message::PlayBlindA => {
                if let Some(ref bt) = self.blind_test {
                    let data = if bt.ai_is_b {
                        &bt.original_data
                    } else {
                        &bt.ai_data
                    };
                    self.player.send(PlayerCommand::PlaySample {
                        data: data.clone(),
                        rate_ratio: 1.0,
                    });
                }
            }
            Message::PlayBlindB => {
                if let Some(ref bt) = self.blind_test {
                    let data = if bt.ai_is_b {
                        &bt.ai_data
                    } else {
                        &bt.original_data
                    };
                    self.player.send(PlayerCommand::PlaySample {
                        data: data.clone(),
                        rate_ratio: 1.0,
                    });
                }
            }
            Message::GuessBlind(guess_a_is_ai) => {
                if let Some(ref mut bt) = self.blind_test {
                    if !bt.revealed {
                        bt.revealed = true;
                        bt.total += 1;
                        // guess_a_is_ai == true means user thinks A is the AI version
                        // A is AI when ai_is_b == false
                        let correct = guess_a_is_ai != bt.ai_is_b;
                        bt.last_correct = correct;
                        if correct {
                            bt.correct += 1;
                        }
                    }
                }
            }
            Message::NextBlindTest => {
                return self.update(Message::StartBlindTest);
            }
            Message::DismissBlindTest => {
                self.blind_test = None;
            }
            Message::FileDropped(path) => {
                return self.update(Message::FileSelected(Some(path)));
            }

            // Panel resizing via drag handles
            Message::DragStart(panel) => {
                self.drag_panel = Some(panel);
                self.drag_start_y = 0.0; // set on first DragMove
            }
            Message::DragMove(y) => {
                if let Some(panel) = self.drag_panel {
                    if self.drag_start_y == 0.0 {
                        self.drag_start_y = y;
                    } else {
                        let delta = y - self.drag_start_y;
                        match panel {
                            // Handle is below pattern viewer: drag down = grow pattern, shrink sample
                            DragPanel::PatternViewer => {
                                let grow = delta.clamp(
                                    PANEL_MIN_PATTERN_VIEWER_HEIGHT
                                        - self.preferred_panel_heights.pattern_viewer,
                                    self.preferred_panel_heights.sample_panel
                                        - PANEL_MIN_SAMPLE_PANEL_HEIGHT,
                                );
                                self.preferred_panel_heights.pattern_viewer += grow;
                                self.preferred_panel_heights.sample_panel -= grow;
                            }
                            // Handle is below VU meters: drag down = grow VU, shrink sample
                            DragPanel::VuMeters => {
                                let grow = delta.clamp(
                                    PANEL_MIN_VU_METERS_HEIGHT
                                        - self.preferred_panel_heights.vu_meters,
                                    self.preferred_panel_heights.sample_panel
                                        - PANEL_MIN_SAMPLE_PANEL_HEIGHT,
                                );
                                self.preferred_panel_heights.vu_meters += grow;
                                self.preferred_panel_heights.sample_panel -= grow;
                            }
                            // Handle is above sample panel: drag down = shrink sample, grow VU
                            DragPanel::SampleList => {
                                // Dragging down shrinks sample panel (negative delta = grow)
                                let grow = (-delta).clamp(
                                    PANEL_MIN_SAMPLE_PANEL_HEIGHT
                                        - self.preferred_panel_heights.sample_panel,
                                    self.preferred_panel_heights.vu_meters
                                        - PANEL_MIN_VU_METERS_HEIGHT,
                                );
                                self.preferred_panel_heights.sample_panel += grow;
                                self.preferred_panel_heights.vu_meters -= grow;
                            }
                        }
                        // Reset baseline so delta is per-frame, not cumulative
                        self.drag_start_y = y;
                    }
                }
            }
            Message::DragEnd => {
                self.drag_panel = None;
            }
        }
        Task::none()
    }

    fn drag_handle(&self, panel: DragPanel) -> Element<'_, Message> {
        let is_dragging = self.drag_panel == Some(panel);
        let bar_color = if is_dragging {
            theme::ACCENT_AMBER
        } else {
            theme::PANEL_SURFACE
        };
        let handle = container(horizontal_rule(1))
            .height(4)
            .width(Length::Fill)
            .style(move |_theme: &Theme| container::Style {
                background: Some(iced::Background::Color(bar_color)),
                ..Default::default()
            });
        mouse_area(handle)
            .on_press(Message::DragStart(panel))
            .interaction(iced::mouse::Interaction::ResizingVertically)
            .into()
    }

    pub fn view(&self) -> Element<'_, Message> {
        let panel_heights = self.resolved_panel_heights();
        let top = column![
            self.view_header(),
            horizontal_rule(1),
            self.view_transport(),
            horizontal_rule(1),
            self.view_oscilloscope(),
            self.view_spectrogram(),
            self.view_pattern_viewer(panel_heights.pattern_viewer),
            self.drag_handle(DragPanel::PatternViewer),
            self.view_vu_meters(panel_heights.vu_meters),
            self.drag_handle(DragPanel::VuMeters),
            self.view_info_panel(),
            self.drag_handle(DragPanel::SampleList),
            self.view_sample_list_with_message(panel_heights.sample_panel),
        ]
        .spacing(4)
        .height(Length::Fill);

        let bottom = column![horizontal_rule(1), self.view_remaster_panel()].spacing(4);

        let content = column![top, bottom].spacing(4).padding(8);

        let vinyl = Shader::new(vinyl_shader::VinylProgram {
            rotation_angle: self.background.angle as f32,
            frame: self.background.frame,
            groove_data: self.groove_data.clone(),
        })
        .width(Length::Fill)
        .height(Length::Fill);
        let bg_canvas = canvas(&self.background)
            .width(Length::Fill)
            .height(Length::Fill);
        let ui_layer = container(content).width(Length::Fill).height(Length::Fill);
        let ui_layer: Element<'_, Message> = if self.drag_panel.is_some() {
            mouse_area(ui_layer)
                .on_move(|point| Message::DragMove(point.y))
                .on_release(Message::DragEnd)
                .interaction(iced::mouse::Interaction::ResizingVertically)
                .into()
        } else {
            ui_layer.into()
        };
        let main_view: Element<'_, Message> = stack![vinyl, bg_canvas, ui_layer]
            .width(Length::Fill)
            .height(Length::Fill)
            .into();

        if self.remaster_interrupt_action.is_some() {
            stack![main_view, opaque(self.view_remaster_interrupt_dialog())]
                .width(Length::Fill)
                .height(Length::Fill)
                .into()
        } else if self.show_audio_error_dialog {
            stack![main_view, opaque(self.view_audio_error_dialog())]
                .width(Length::Fill)
                .height(Length::Fill)
                .into()
        } else if self.show_archive_picker {
            stack![main_view, opaque(self.view_archive_picker())]
                .width(Length::Fill)
                .height(Length::Fill)
                .into()
        } else if self.render_progress_rx.is_some() {
            stack![main_view, opaque(self.view_render_progress_dialog())]
                .width(Length::Fill)
                .height(Length::Fill)
                .into()
        } else if self.show_install_dialog {
            stack![main_view, opaque(self.view_install_dialog())]
                .width(Length::Fill)
                .height(Length::Fill)
                .into()
        } else if self.blind_test.is_some() {
            stack![main_view, opaque(self.view_blind_test_dialog())]
                .width(Length::Fill)
                .height(Length::Fill)
                .into()
        } else {
            main_view
        }
    }

    fn view_header(&self) -> Element<'_, Message> {
        let bold = iced::Font {
            weight: iced::font::Weight::Bold,
            ..Default::default()
        };
        let title_row: Element<'_, Message> = match &self.module_info {
            Some(info) => {
                let title = if info.title.is_empty() {
                    "Untitled"
                } else {
                    &info.title
                };
                let format = if info.format_type_long.is_empty() {
                    &info.format_type
                } else {
                    &info.format_type_long
                };
                row![
                    text(format!("{title}  "))
                        .size(18)
                        .color(theme::ACCENT_AMBER)
                        .font(bold),
                    text(format!("[{format}]"))
                        .size(14)
                        .color(theme::PANEL_LABEL),
                ]
                .align_y(iced::Alignment::End)
                .into()
            }
            None => text("No module loaded")
                .size(18)
                .color(theme::PANEL_LABEL)
                .into(),
        };

        let tracker = match &self.module_info {
            Some(info) if !info.tracker.is_empty() => {
                text(&info.tracker).size(12).color(theme::PANEL_LABEL)
            }
            _ => text("").size(12),
        };

        let rate_khz = self.player.current_playback_rate() / 1000;
        let fmt = self.player.output_format_label();
        let output_label = text(format!("{rate_khz}kHz / {fmt}"))
            .size(12)
            .color(theme::ACCENT_GREEN);

        stack![
            container(title_row)
                .width(Length::Fill)
                .center_x(Length::Fill),
            row![output_label, horizontal_space(), tracker,].align_y(iced::Alignment::Center),
        ]
        .into()
    }

    fn view_oscilloscope(&self) -> Element<'_, Message> {
        if let Some(idx) = self.selected_waveform_sample {
            if let Some(slot) = self.sample_slots.iter().find(|s| s.index == idx) {
                return self.view_sample_waveform(slot);
            }
            if let Some(element) = self.view_sample_waveform_from_module(idx) {
                return element;
            }
        }
        let frames = self.waveform.len() / 2;
        let mut left = Vec::with_capacity(frames);
        let mut right = Vec::with_capacity(frames);
        for i in 0..frames {
            left.push(self.waveform[i * 2] as f32);
            right.push(self.waveform.get(i * 2 + 1).copied().unwrap_or(0.0) as f32);
        }
        Shader::new(scope_shader::ScopeProgram {
            left,
            right,
            gain: self.oscilloscope_gain as f32,
        })
        .width(Length::Fill)
        .height(OSCILLOSCOPE_HEIGHT)
        .into()
    }

    fn view_sample_waveform_from_module(&self, index: i32) -> Option<Element<'_, Message>> {
        let info = self.module_info.as_ref()?;
        let sample_info = info.samples.iter().find(|s| s.index == index)?;
        let data = self.keyjazz_data.as_ref()?;
        if data.is_empty() {
            return None;
        }

        let normal_loop = if sample_info.has_loop {
            crate::openmpt::SampleLoopRegion {
                start_frames: sample_info.loop_start_frames,
                end_frames: sample_info.loop_end_frames,
                mode: sample_info.loop_mode,
            }
        } else {
            crate::openmpt::SampleLoopRegion::none()
        };
        let sustain_loop = if sample_info.has_sustain_loop {
            crate::openmpt::SampleLoopRegion {
                start_frames: sample_info.sustain_loop_start_frames,
                end_frames: sample_info.sustain_loop_end_frames,
                mode: sample_info.sustain_loop_mode,
            }
        } else {
            crate::openmpt::SampleLoopRegion::none()
        };

        let total_frames = data.len() as i64;
        let header_text = format!(
            "Sample #{} — Original @ {} Hz — {} frames (1 ch mono-mixed)",
            index + 1,
            sample_info.rate,
            total_frames,
        );

        Some(self.build_waveform_element(data.clone(), 1, normal_loop, sustain_loop, header_text))
    }

    fn view_sample_waveform(&self, slot: &SampleSlot) -> Element<'_, Message> {
        let (data, channels, rate, normal_loop, sustain_loop) = match &slot.mode {
            SampleMode::Original => (
                slot.original.as_slice(),
                slot.original_channels,
                slot.original_rate,
                slot.loop_info.normal,
                slot.loop_info.sustain,
            ),
            SampleMode::Reference48k => match &slot.reference_48k {
                Some(data) => (
                    data.as_slice(),
                    slot.original_channels,
                    48_000,
                    crate::remaster::scaled_loop_region(
                        slot.loop_info.normal,
                        slot.original_rate as u32,
                        48_000,
                    ),
                    crate::remaster::scaled_loop_region(
                        slot.loop_info.sustain,
                        slot.original_rate as u32,
                        48_000,
                    ),
                ),
                None => (
                    slot.original.as_slice(),
                    slot.original_channels,
                    slot.original_rate,
                    slot.loop_info.normal,
                    slot.loop_info.sustain,
                ),
            },
            SampleMode::Engine(name) => match slot.engine_result(name) {
                Some(er) => {
                    let (n, s) = match er.discovered_loops.as_ref() {
                        Some(d) => (d.normal, d.sustain),
                        None => (
                            crate::remaster::scaled_loop_region(
                                slot.loop_info.normal,
                                slot.original_rate as u32,
                                er.sample_rate_hz as u32,
                            ),
                            crate::remaster::scaled_loop_region(
                                slot.loop_info.sustain,
                                slot.original_rate as u32,
                                er.sample_rate_hz as u32,
                            ),
                        ),
                    };
                    (er.data.as_slice(), er.channels, er.sample_rate_hz, n, s)
                }
                None => (
                    slot.original.as_slice(),
                    slot.original_channels,
                    slot.original_rate,
                    slot.loop_info.normal,
                    slot.loop_info.sustain,
                ),
            },
        };

        let total_frames = data.len() as i64 / channels.max(1) as i64;
        let header_text = format!(
            "Sample #{} — {} @ {} Hz — {} frames ({} ch)",
            slot.index + 1,
            sample_mode_name_for_slot(slot, &slot.mode),
            rate,
            total_frames,
            channels.max(1),
        );

        self.build_waveform_element(
            data.to_vec(),
            channels.max(1),
            normal_loop,
            sustain_loop,
            header_text,
        )
    }

    fn build_waveform_element(
        &self,
        data: Vec<f64>,
        channels: i32,
        normal_loop: crate::openmpt::SampleLoopRegion,
        sustain_loop: crate::openmpt::SampleLoopRegion,
        header_text: String,
    ) -> Element<'_, Message> {
        let header = text(header_text).size(10).color(theme::CONTENT_TEXT);
        let program = WaveformProgram {
            data,
            channels,
            normal_loop,
            sustain_loop,
            cache: &self.waveform_cache,
        };
        let canvas_el = canvas(program)
            .width(Length::Fill)
            .height(OSCILLOSCOPE_HEIGHT - 14.0);
        column![header, canvas_el]
            .spacing(2)
            .width(Length::Fill)
            .into()
    }

    fn view_spectrogram(&self) -> Element<'_, Message> {
        let full_upload = if self.spectrogram.take_full_upload_needed() {
            Some(self.spectrogram.magnitudes().to_vec())
        } else {
            None
        };
        let dirty = if full_upload.is_some() {
            let _ = self.spectrogram.take_dirty_spans();
            Vec::new()
        } else {
            self.spectrogram.take_dirty_spans()
        };
        Shader::new(scope_shader::SpectroProgram {
            dirty,
            write_index: self.spectrogram.write_index(),
            sample_rate: self.player.current_playback_rate() as f32,
            full_upload,
            num_cols: self.spectrogram.num_cols(),
            observed_width_px: self.spectrogram.observed_width_handle(),
        })
        .width(Length::Fill)
        .height(SPECTROGRAM_HEIGHT)
        .into()
    }

    fn view_pattern_viewer(&self, panel_height: f32) -> Element<'_, Message> {
        if !self.show_pattern_viewer {
            return text("").size(1).into();
        }
        let (pat_data, current_row) = match &self.pattern_cache {
            Some((_, rows)) => (rows, self.player_state.current_row),
            None => {
                return container(
                    text("Pattern viewer: load a module to see pattern data")
                        .size(10)
                        .color(theme::PANEL_LABEL),
                )
                .height(panel_height)
                .center_y(panel_height)
                .into();
            }
        };

        let mono = iced::Font::MONOSPACE;
        let num_rows = pat_data.len();
        if num_rows == 0 {
            return container(text("Empty pattern").size(10).color(theme::PANEL_LABEL))
                .height(panel_height)
                .center_y(panel_height)
                .into();
        }

        // Determine visible channels (show up to 6)
        let num_ch = pat_data[0].len();
        let visible_ch = num_ch.min(6);

        // Build rows centered around current_row, adaptive to panel height
        let row_height = 14.0_f32; // size(10) text + padding
        let visible_rows = (panel_height / row_height).floor() as i32;
        let context = ((visible_rows - 1) / 2).max(1);
        let start = (current_row - context).max(0);
        let end = (current_row + context + 1).min(num_rows as i32);

        let mut col_items: Vec<Element<Message>> = Vec::new();

        for r in start..end {
            let is_current = r == current_row;
            let row_num = text(format!("{:02}", r))
                .size(10)
                .font(mono)
                .color(if is_current {
                    theme::CONTENT_TEXT
                } else {
                    theme::PANEL_LABEL
                });

            let mut row_items: Vec<Element<Message>> = vec![row_num.into()];

            if let Some(channels) = pat_data.get(r as usize) {
                for ch in 0..visible_ch {
                    if let Some((fmt, hl)) = channels.get(ch) {
                        // Color each character based on highlight code
                        let colored = highlight_pattern_text(fmt, hl, is_current);
                        let sep = text("|").size(10).font(mono).color(theme::PANEL_SURFACE);
                        row_items.push(sep.into());
                        row_items.push(colored);
                    }
                }
            }

            let row_widget = iced::widget::Row::with_children(row_items)
                .spacing(2)
                .align_y(iced::Alignment::Center);

            let row_container = container(row_widget).padding([1, 4]);
            let styled: Element<Message> = if is_current {
                // Currently-playing row: green bar (live signal).
                container(row_container)
                    .style(|_theme: &Theme| container::Style {
                        background: Some(iced::Background::Color(iced::Color {
                            a: 0.35,
                            ..theme::ACCENT_GREEN
                        })),
                        ..Default::default()
                    })
                    .width(Length::Fill)
                    .into()
            } else {
                row_container.width(Length::Fill).into()
            };
            col_items.push(styled);
        }

        let pattern_col = Column::with_children(col_items).spacing(0);
        container(scrollable(pattern_col).height(panel_height).direction(
            scrollable::Direction::Vertical(
                scrollable::Scrollbar::new().width(0).scroller_width(0),
            ),
        ))
        .into()
    }

    fn view_transport(&self) -> Element<'_, Message> {
        let is_playing = self.player_state.status == PlaybackStatus::Playing;
        let has_loaded_module = self.loaded_path.is_some();
        let blind_test_ready = self.sample_slots.iter().any(|sl| sl.has_engine_results());

        let play_pause: Element<'_, Message> = if is_playing {
            widgets::with_tooltip(
                button(text("Pause").size(13))
                    .on_press(Message::Pause)
                    .padding([4, 12]),
                play_pause_button_tooltip(true),
            )
        } else {
            widgets::with_tooltip(
                button(text("Play").size(13))
                    .on_press(Message::Play)
                    .padding([4, 12]),
                play_pause_button_tooltip(false),
            )
        };

        let stop = widgets::with_tooltip(
            button(text("Stop").size(13))
                .on_press(Message::Stop)
                .padding([4, 12]),
            stop_button_tooltip(),
        );

        let open = widgets::with_tooltip(
            button(text("Open").size(13))
                .on_press(Message::OpenFileDialog)
                .padding([4, 12]),
            open_button_tooltip(),
        );

        let render_enabled = has_loaded_module && self.render_progress_rx.is_none();
        // Secondary actions: outlined amber on panel background, amber text.
        // The filled-vs-outlined distinction tells the user which is primary
        // without introducing a second accent color.
        let secondary_outlined = move |_theme: &Theme, _status| button::Style {
            background: Some(iced::Background::Color(theme::PANEL_BG)),
            text_color: theme::ACCENT_AMBER,
            border: iced::Border {
                color: theme::ACCENT_AMBER,
                width: 1.0,
                radius: 4.0.into(),
            },
            ..button::Style::default()
        };
        let mut render_flac_button =
            button(text("Render to FLAC").size(13).color(theme::ACCENT_AMBER))
                .padding([4, 12])
                .style(secondary_outlined);
        if render_enabled {
            render_flac_button = render_flac_button.on_press(Message::RenderFlac);
        }
        let render_flac = widgets::with_tooltip(
            render_flac_button,
            render_button_tooltip("FLAC", render_enabled),
        );

        let mut render_aac_button =
            button(text("Render to M4A").size(13).color(theme::ACCENT_AMBER))
                .padding([4, 12])
                .style(secondary_outlined);
        if render_enabled {
            render_aac_button = render_aac_button.on_press(Message::RenderAac);
        }
        let render_aac = widgets::with_tooltip(
            render_aac_button,
            render_button_tooltip("M4A", render_enabled),
        );

        let mut blind_btn_button = button(text("Blind Test").size(13)).padding([4, 12]);
        if blind_test_ready {
            blind_btn_button = blind_btn_button.on_press(Message::StartBlindTest);
        }
        let blind_btn = widgets::with_tooltip(
            blind_btn_button,
            blind_test_button_tooltip(blind_test_ready),
        );

        let duration = self.player_state.duration_seconds.max(1.0);
        let seek = slider(0.0..=duration, self.seek_position, Message::SeekChanged)
            .on_release(Message::SeekReleased)
            .width(Length::Fill)
            .height(20);

        let pct = if duration > 0.0 {
            self.seek_position / duration * 100.0
        } else {
            0.0
        };
        let time = text(format!(
            "{} / {} ({:.0}%)",
            widgets::format_time(self.seek_position),
            widgets::format_time(self.player_state.duration_seconds),
            pct
        ))
        .size(13)
        .color(theme::CONTENT_TEXT);

        let position = text(widgets::format_position(
            self.player_state.current_order,
            self.player_state.current_row,
        ))
        .size(11)
        .color(theme::PANEL_LABEL);

        let interp = pick_list(
            INTERPOLATION_CHOICES,
            Some(self.interpolation),
            Message::SetInterpolation,
        )
        .text_size(12)
        .padding([2, 6]);
        let agc = checkbox("AGC", self.agc_enabled)
            .on_toggle(Message::SetAgcEnabled)
            .text_size(11)
            .size(14);
        let agc = widgets::with_tooltip(agc, agc_checkbox_tooltip(self.agc_enabled));
        let hrtf_cb = checkbox("HRTF", self.hrtf_enabled)
            .on_toggle(Message::SetHrtfEnabled)
            .text_size(11)
            .size(14);
        let hrtf_cb = widgets::with_tooltip(
            hrtf_cb,
            if self.hrtf_enabled {
                "Disable headphone HRTF spatialization."
            } else {
                "Enable headphone HRTF spatialization (binaural speaker simulation)."
            },
        );

        let mut col = Column::new().spacing(4);
        col = col.push(
            row![
                open,
                play_pause,
                stop,
                time,
                horizontal_space(),
                render_flac,
                render_aac,
                blind_btn,
            ]
            .spacing(6)
            .align_y(iced::Alignment::Center),
        );
        col = col.push(seek);
        col = col.push(
            row![
                position,
                text(format!(
                    "BPM: {:.0}  Spd: {}",
                    self.player_state.bpm, self.player_state.speed
                ))
                .size(11)
                .color(theme::PANEL_LABEL),
                horizontal_space(),
                text(format!("Stereo: {}%", self.stereo_separation))
                    .size(11)
                    .color(theme::PANEL_LABEL),
                slider(
                    0..=100,
                    self.stereo_separation,
                    Message::SetStereoSeparation
                )
                .width(100),
                agc,
                hrtf_cb,
                text(format!("Mix: {}%", self.hrtf_mix))
                    .size(11)
                    .color(if self.hrtf_enabled {
                        theme::PANEL_LABEL
                    } else {
                        theme::PANEL_SURFACE
                    }),
                slider(0..=100, self.hrtf_mix, Message::SetHrtfMix).width(80),
                text("Interp:").size(11).color(theme::PANEL_LABEL),
                interp,
            ]
            .spacing(6)
            .align_y(iced::Alignment::Center),
        );
        col.into()
    }

    fn view_vu_meters(&self, panel_height: f32) -> Element<'_, Message> {
        if self.player_state.channel_vu.is_empty() {
            return container(text("No channels").size(11).color(theme::PANEL_LABEL))
                .height(panel_height)
                .center_y(panel_height)
                .into();
        }

        let meters: Vec<Element<Message>> = self
            .player_state
            .channel_vu
            .iter()
            .enumerate()
            .map(|(i, (left, right))| {
                let label = text(format!("{:02}", i + 1))
                    .size(10)
                    .color(theme::PANEL_LABEL);
                row![
                    label,
                    widgets::vu_bar(*left as f32),
                    widgets::vu_bar(*right as f32),
                ]
                .spacing(2)
                .align_y(iced::Alignment::Center)
                .into()
            })
            .collect();

        let col = Column::with_children(meters).spacing(1);
        scrollable(col)
            .height(panel_height)
            .direction(scrollable::Direction::Vertical(
                scrollable::Scrollbar::default(),
            ))
            .into()
    }

    fn view_info_panel(&self) -> Element<'_, Message> {
        match &self.module_info {
            Some(info) => {
                let size_str = if info.file_size_bytes >= 1_048_576 {
                    format!("{:.1} MB", info.file_size_bytes as f64 / 1_048_576.0)
                } else if info.file_size_bytes > 0 {
                    format!("{} KB", info.file_size_bytes / 1024)
                } else {
                    String::new()
                };

                let mut badge_parts = vec![
                    info.format_type.to_uppercase(),
                    format!("{}ch", info.num_channels),
                    format!("{} samples", info.samples.len()),
                ];
                if !size_str.is_empty() {
                    badge_parts.push(size_str);
                }
                if !info.artist.is_empty() {
                    badge_parts.push(info.artist.clone());
                }
                if !info.date.is_empty() {
                    badge_parts.push(info.date.clone());
                }
                let badge = badge_parts.join(" | ");

                text(badge).size(12).color(theme::PANEL_LABEL).into()
            }
            None => text("").size(12).into(),
        }
    }

    fn view_sample_list_with_message(&self, panel_height: f32) -> Element<'_, Message> {
        let sample_list = container(self.view_sample_list(panel_height))
            .width(Length::FillPortion(3))
            .height(panel_height);
        if let Some(msg_panel) = self.view_module_message(panel_height) {
            row![sample_list, msg_panel]
                .spacing(8)
                .width(Length::Fill)
                .height(panel_height)
                .into()
        } else {
            sample_list.into()
        }
    }

    fn view_module_message(&self, panel_height: f32) -> Option<Element<'_, Message>> {
        let info = self.module_info.as_ref()?;
        if info.message.is_empty() {
            return None;
        }
        let msg_text = text(&info.message)
            .size(10)
            .color(theme::PANEL_LABEL)
            .font(iced::Font::MONOSPACE);
        Some(
            scrollable(msg_text)
                .width(Length::FillPortion(1))
                .height(panel_height)
                .direction(scrollable::Direction::Vertical(
                    scrollable::Scrollbar::default(),
                ))
                .into(),
        )
    }

    fn view_sample_list(&self, panel_height: f32) -> Element<'_, Message> {
        let samples = match &self.module_info {
            Some(info) => &info.samples,
            None => {
                return container(
                    text("Sample list: load a module to see samples")
                        .size(11)
                        .color(theme::PANEL_LABEL),
                )
                .height(panel_height)
                .center_y(panel_height)
                .into();
            }
        };

        let cycle_all_samples = widgets::with_tooltip(
            button(text("AI").size(11).color(theme::PANEL_LABEL))
                .on_press(Message::CycleAllSampleModes)
                .padding(0)
                .style(|_theme: &Theme, _status| button::Style {
                    background: None,
                    ..button::Style::default()
                })
                .width(Length::Shrink),
            cycle_all_samples_tooltip(),
        );

        let header = row![
            cycle_all_samples,
            text("#").size(11).color(theme::PANEL_LABEL).width(30),
            text("Name")
                .size(11)
                .color(theme::PANEL_LABEL)
                .width(Length::Fill),
            text("Rate").size(11).color(theme::PANEL_LABEL).width(70),
            text("Bits").size(11).color(theme::PANEL_LABEL).width(30),
            text("Len").size(11).color(theme::PANEL_LABEL).width(70),
            text("Lp").size(11).color(theme::PANEL_LABEL).width(20),
            text("Ch").size(11).color(theme::PANEL_LABEL).width(30),
        ]
        .spacing(4);

        let active = &self.player_state.active_samples;

        let rows: Vec<Element<Message>> = samples
            .iter()
            .filter(|s| s.length_frames > 0)
            .map(|s| {
                let is_active = active.contains(&s.index);

                // Already-48k native samples shine green (live, fully sampled);
                // active rows get the amber selection treatment per the design spec;
                // everything else uses the standard text hierarchy.
                let rate_color = if s.rate >= 48000 {
                    theme::ACCENT_GREEN
                } else if is_active {
                    theme::ACCENT_AMBER
                } else {
                    theme::CONTENT_TEXT
                };

                let name_color = if is_active {
                    theme::ACCENT_AMBER
                } else {
                    theme::CONTENT_TEXT
                };

                let dim_color = if is_active {
                    theme::ACCENT_AMBER
                } else {
                    theme::PANEL_LABEL
                };

                let idx_color = if is_active {
                    theme::ACCENT_AMBER
                } else {
                    theme::PANEL_LABEL
                };

                let duration = if s.rate > 0 {
                    format!("{:.2}s", s.length_frames as f64 / s.rate as f64)
                } else {
                    "?".into()
                };

                // Show mode cycling button if this sample has been remastered
                let slot = self.sample_slots.iter().find(|sl| sl.index == s.index);
                let toggle: Element<Message> = if let Some(sl) = slot {
                    if sl.has_engine_results() {
                        let idx = s.index;
                        let label = sample_mode_button_label(sl);
                        let bg_color = sample_mode_button_color(sl);
                        widgets::with_tooltip_at(
                            button(text(label).size(9).color(theme::PANEL_BG))
                                .on_press(Message::CycleSampleMode(idx))
                                .padding([1, 2])
                                .style(move |_theme: &Theme, _status| button::Style {
                                    background: Some(iced::Background::Color(bg_color)),
                                    text_color: theme::PANEL_BG,
                                    border: iced::Border::default().rounded(3),
                                    ..button::Style::default()
                                }),
                            sample_mode_tooltip(sl),
                            iced::widget::tooltip::Position::FollowCursor,
                        )
                    } else if sl.failed {
                        text("!").size(14).color(theme::ACCENT_AMBER).into()
                    } else if sl.engines_total > 0 && sl.engines_done > 0 {
                        text(format!("{}/{}", sl.engines_done, sl.engines_total))
                            .size(9)
                            .color(theme::ACCENT_GREEN)
                            .into()
                    } else if sl.original_rate >= 48_000 {
                        container(text("Original").size(9).color(theme::PANEL_BG))
                            .padding([1, 2])
                            .style(|_theme| container::Style {
                                background: Some(iced::Background::Color(theme::ACCENT_GREEN)),
                                border: iced::Border::default().rounded(3),
                                ..container::Style::default()
                            })
                            .into()
                    } else {
                        text("").width(14).into()
                    }
                } else {
                    text("").width(14).into()
                };

                let is_muted = slot.is_some_and(|sl| sl.muted);
                let is_soloed = !is_muted
                    && self.sample_slots.iter().any(|sl| sl.muted)
                    && slot.is_some_and(|sl| !sl.muted);
                // Toggle controls: amber fill when active, panel gray when inactive.
                let mute_btn: Element<Message> = button(text("M").size(8).color(if is_muted {
                    theme::PANEL_BG
                } else {
                    theme::PANEL_LABEL
                }))
                .on_press(Message::ToggleSampleMute(s.index))
                .padding([1, 3])
                .style(move |_theme: &Theme, _status| button::Style {
                    background: Some(iced::Background::Color(if is_muted {
                        theme::ACCENT_AMBER
                    } else {
                        theme::PANEL_SURFACE
                    })),
                    border: iced::Border::default().rounded(2),
                    ..button::Style::default()
                })
                .into();
                let solo_btn: Element<Message> = button(text("S").size(8).color(if is_soloed {
                    theme::PANEL_BG
                } else {
                    theme::PANEL_LABEL
                }))
                .on_press(Message::SoloSample(s.index))
                .padding([1, 3])
                .style(move |_theme: &Theme, _status| button::Style {
                    background: Some(iced::Background::Color(if is_soloed {
                        theme::ACCENT_AMBER
                    } else {
                        theme::PANEL_SURFACE
                    })),
                    border: iced::Border::default().rounded(2),
                    ..button::Style::default()
                })
                .into();

                let sample_row = row![
                    container(toggle).width(Length::Shrink),
                    container(mute_btn).width(Length::Shrink),
                    container(solo_btn).width(Length::Shrink),
                    text(format!("{:02}", s.index + 1))
                        .size(11)
                        .color(idx_color)
                        .width(30),
                    text(&s.name).size(11).color(name_color).width(Length::Fill),
                    text(format!("{}Hz", s.rate))
                        .size(11)
                        .color(rate_color)
                        .width(70),
                    text(s.sample_format.label())
                        .size(11)
                        .color(
                            if s.sample_format == SampleFormat::Float64
                                || s.sample_format == SampleFormat::Float32
                            {
                                theme::ACCENT_GREEN
                            } else {
                                dim_color
                            }
                        )
                        .width(30),
                    text(duration).size(11).color(dim_color).width(70),
                    text(if s.has_loop { "L" } else { "" })
                        .size(11)
                        .color(dim_color)
                        .width(20),
                    text(if s.channels == 2 { "ST" } else { "M" })
                        .size(11)
                        .color(dim_color)
                        .width(30),
                ]
                .spacing(4)
                .align_y(iced::Alignment::Center);

                let is_keyjazz = self.keyjazz_sample == Some(s.index);
                // Selection (keyjazz) → amber tint; live-playing → green tint;
                // both at low alpha so the row text stays primary.
                let bg = if is_keyjazz {
                    iced::Color {
                        a: 0.30,
                        ..theme::ACCENT_AMBER
                    }
                } else if is_active {
                    iced::Color {
                        a: 0.30,
                        ..theme::ACCENT_GREEN
                    }
                } else {
                    iced::Color::TRANSPARENT
                };
                let idx = s.index;
                widgets::with_tooltip_at(
                    button(
                        container(sample_row)
                            .style(move |_theme: &Theme| container::Style {
                                background: if is_keyjazz || is_active {
                                    Some(iced::Background::Color(bg))
                                } else {
                                    None
                                },
                                ..Default::default()
                            })
                            .width(Length::Fill),
                    )
                    .on_press(Message::SelectKeyjazzSample(idx))
                    .padding(0)
                    .width(Length::Fill)
                    .style(|_theme: &Theme, _status| button::Style {
                        background: None,
                        ..button::Style::default()
                    }),
                    sample_row_tooltip(s, is_keyjazz),
                    iced::widget::tooltip::Position::FollowCursor,
                )
            })
            .collect();

        let list = Column::with_children(rows).spacing(1);

        column![header, scrollable(list).height(Length::Fill),]
            .spacing(2)
            .height(panel_height)
            .into()
    }

    fn view_remaster_panel(&self) -> Element<'_, Message> {
        let attract = self.attract_phase;
        let primary_action = self.remaster_primary_action();
        // The Remaster button is the single most prominent interactive element:
        // amber fill, PANEL_BG text, attract pulse breathes from panel-surface
        // (resting) into the full amber accent. Hover holds the glow at 50% so
        // the button reads as armed even between pulses.
        let remaster_button = match primary_action {
            RemasterPrimaryAction::Install => {
                button(text("Install AI Engine").size(13).color(theme::PANEL_BG))
                    .on_press(Message::ShowInstallDialog)
                    .padding([6, 16])
                    .style(|_theme: &Theme, _status| button::Style {
                        background: Some(iced::Background::Color(theme::ACCENT_AMBER)),
                        text_color: theme::PANEL_BG,
                        border: iced::Border::default().rounded(4),
                        ..button::Style::default()
                    })
            }
            RemasterPrimaryAction::Start => button(
                text(format!(
                    "Start Remaster ({})",
                    self.remaster_engine.engine_name()
                ))
                .size(13),
            )
            .on_press(Message::StartRemaster)
            .padding([6, 16])
            .style(move |theme: &Theme, status| {
                // Rest state matches Play/Pause/Stop (button::primary at Active).
                // Pulse flashes the base toward white (CRT phosphor feel);
                // hover holds the flash at 50% so the button reads as armed.
                let base = button::primary(theme, button::Status::Active);
                let pulse = (attract * std::f32::consts::PI).sin().clamp(0.0, 1.0);
                let hover_floor = match status {
                    button::Status::Hovered | button::Status::Pressed => 0.5,
                    _ => 0.0,
                };
                let t = pulse.max(hover_floor);
                if t == 0.0 {
                    return base;
                }
                let from = match base.background {
                    Some(iced::Background::Color(c)) => c,
                    _ => theme::PANEL_SURFACE,
                };
                let to = iced::Color::WHITE;
                let bg = iced::Color::from_rgb(
                    from.r + (to.r - from.r) * t,
                    from.g + (to.g - from.g) * t,
                    from.b + (to.b - from.b) * t,
                );
                button::Style {
                    background: Some(iced::Background::Color(bg)),
                    ..base
                }
            }),
            RemasterPrimaryAction::Cancel => {
                button(text("Cancel Remaster").size(13).color(theme::PANEL_BG))
                    .on_press(Message::CancelRemaster)
                    .padding([6, 16])
                    .style(|_theme: &Theme, _status| button::Style {
                        background: Some(iced::Background::Color(theme::ACCENT_AMBER)),
                        text_color: theme::PANEL_BG,
                        border: iced::Border::default().rounded(4),
                        ..button::Style::default()
                    })
            }
            RemasterPrimaryAction::Cancelling => {
                button(text("Cancelling...").size(13).color(theme::PANEL_BG))
                    .padding([6, 16])
                    .style(|_theme: &Theme, _status| button::Style {
                        background: Some(iced::Background::Color(theme::PANEL_LABEL)),
                        text_color: theme::PANEL_BG,
                        border: iced::Border::default().rounded(4),
                        ..button::Style::default()
                    })
            }
            RemasterPrimaryAction::Complete => {
                button(text("Remaster Complete!").size(13).color(theme::PANEL_BG))
                    .padding([6, 16])
                    .style(|_theme: &Theme, _status| button::Style {
                        background: Some(iced::Background::Color(theme::ACCENT_GREEN)),
                        text_color: theme::PANEL_BG,
                        border: iced::Border::default().rounded(4),
                        ..button::Style::default()
                    })
            }
            RemasterPrimaryAction::Disabled => button(
                text(format!(
                    "Start Remaster ({})",
                    self.remaster_engine.engine_name()
                ))
                .size(13)
                .color(theme::PANEL_LABEL),
            )
            .padding([6, 16])
            .style(|_theme: &Theme, _status| button::Style {
                background: Some(iced::Background::Color(theme::PANEL_SURFACE)),
                text_color: theme::PANEL_LABEL,
                border: iced::Border::default().rounded(4),
                ..button::Style::default()
            }),
        };
        let remaster_btn =
            widgets::with_tooltip(remaster_button, remaster_primary_tooltip(primary_action));

        // Error state: special layout with scrollable error + copy button
        if let RemasterStatus::Failed(e) = &self.remaster_status {
            let error_scroll = scrollable(
                text(format!("Error: {e}"))
                    .size(11)
                    .color(theme::ACCENT_AMBER),
            )
            .height(150);

            let copy_btn = widgets::with_tooltip(
                button(text("Copy Error").size(11))
                    .on_press(Message::CopyRemasterError)
                    .padding([4, 10]),
                "Copy the remaster error to the clipboard.",
            );

            return column![
                row![remaster_btn, copy_btn]
                    .spacing(8)
                    .align_y(iced::Alignment::Center),
                error_scroll,
            ]
            .spacing(4)
            .into();
        }

        let status_text: Element<'_, Message> = match &self.remaster_status {
            RemasterStatus::Unavailable => text("No Quinlight AI engines installed")
                .size(12)
                .color(theme::PANEL_LABEL)
                .into(),
            RemasterStatus::Ready => text("Ready to remaster with Quinlight")
                .size(12)
                .color(theme::ACCENT_GREEN)
                .into(),
            RemasterStatus::Processing {
                current,
                total,
                sample_name,
            } => text(format!(
                "Processing sample {current}/{total}: {sample_name}"
            ))
            .size(12)
            .color(theme::ACCENT_AMBER)
            .into(),
            RemasterStatus::Cancelling => text("Cancelling remaster...")
                .size(12)
                .color(theme::ACCENT_AMBER)
                .into(),
            RemasterStatus::Complete => text("Remaster complete!")
                .size(12)
                .color(theme::ACCENT_GREEN)
                .into(),
            RemasterStatus::Cancelled => text("Remaster cancelled")
                .size(12)
                .color(theme::PANEL_LABEL)
                .into(),
            RemasterStatus::Failed(_) => unreachable!(),
            RemasterStatus::Log(_) | RemasterStatus::EngineProgress { .. } => text("").into(), // handled separately
        };

        let progress = match &self.remaster_status {
            RemasterStatus::Processing { current, total, .. } if *total > 0 => {
                let pct = *current as f32 / *total as f32;
                Some(
                    progress_bar(0.0..=1.0, pct)
                        .height(8)
                        .style(|_theme: &Theme| progress_bar::Style {
                            background: iced::Background::Color(theme::PANEL_SURFACE),
                            bar: iced::Background::Color(theme::ACCENT_GREEN),
                            border: iced::Border::default(),
                        }),
                )
            }
            _ => None,
        };

        let cache_enabled = self.loaded_path.is_some();
        let mut clear_cache_button = button(text("⚠️ Clear Song Cache").size(11)).padding([2, 8]);
        if cache_enabled {
            clear_cache_button = clear_cache_button.on_press(Message::ClearSongCache);
        }
        let clear_cache_btn =
            widgets::with_tooltip(clear_cache_button, clear_song_cache_tooltip(cache_enabled));

        // Per-engine checkboxes: show detected engines + install hint for missing ones
        let detected: Vec<&str> = self.remaster_engine.available_engine_names();
        let mut engine_checks: Vec<Element<'_, Message>> = Vec::new();
        let engines_info: [(&str, bool); 4] = [
            ("LavaSR", self.enable_lavasr),
            ("FLowHigh", self.enable_flowhigh),
            ("AP-BWE", self.enable_apbwe),
            ("AudioSR", self.enable_audiosr),
        ];
        for (name, enabled) in &engines_info {
            if detected.iter().any(|d| d.eq_ignore_ascii_case(name)) {
                let n = name.to_string();
                engine_checks.push(widgets::with_tooltip(
                    checkbox(*name, *enabled)
                        .on_toggle(move |v| Message::ToggleEngine(n.clone(), v))
                        .text_size(11)
                        .size(14),
                    engine_checkbox_tooltip(name, *enabled),
                ));
            }
        }
        if let Some(names) = self.remaster_engine.missing_engine_names() {
            engine_checks.push(widgets::with_tooltip(
                button(text(format!("Install {}", names.join(", "))).size(10))
                    .on_press(Message::ShowInstallMissing)
                    .padding([1, 6]),
                install_missing_button_tooltip(&names),
            ));
        }
        let engine_widget: Element<'_, Message> = iced::widget::Row::with_children(engine_checks)
            .spacing(6)
            .align_y(iced::Alignment::Center)
            .into();
        let cleanup_picker = pick_list(
            CLEANUP_MODES,
            Some(self.cleanup_settings.mode),
            Message::SetCleanupMode,
        )
        .text_size(11)
        .padding([2, 6]);
        let cleanup_engine_picker = pick_list(
            CLEANUP_ENGINES,
            Some(self.cleanup_settings.engine_version),
            Message::SetCleanupEngineVersion,
        )
        .text_size(11)
        .padding([2, 6]);

        let mut content = column![
            row![
                remaster_btn,
                status_text,
                horizontal_space(),
                clear_cache_btn,
            ]
            .spacing(6)
            .align_y(iced::Alignment::Center),
            row![
                text("Cleanup").size(11).color(theme::PANEL_LABEL),
                cleanup_picker,
                text("Engine").size(11).color(theme::PANEL_LABEL),
                cleanup_engine_picker,
                text(format!("Quality: {}", self.ddim_steps))
                    .size(11)
                    .color(theme::PANEL_LABEL),
                slider(25..=100, self.ddim_steps as i32, |v| Message::SetDdimSteps(
                    v as u32
                ))
                .width(80),
                horizontal_space(),
                engine_widget,
            ]
            .spacing(6)
            .align_y(iced::Alignment::Center),
        ]
        .spacing(4);

        if let Some(bar) = progress {
            content = content.push(bar);
        }

        if !self.remaster_log.is_empty() {
            let log_text = self.remaster_log.join("\n");
            content = content.push(
                scrollable(text(log_text).size(10).color(theme::PANEL_LABEL))
                    .height(Length::Fixed(120.0)),
            );
        }

        if let Some(note) = &self.remaster_notice {
            content = content.push(text(note).size(11).color(theme::ACCENT_GREEN));
        }

        content.into()
    }

    fn view_blind_test_dialog(&self) -> Element<'_, Message> {
        let bt = self.blind_test.as_ref().unwrap();

        let title = text("Can YOU hear the difference?")
            .size(18)
            .color(theme::ACCENT_AMBER)
            .font(iced::Font {
                weight: iced::font::Weight::Bold,
                ..Default::default()
            });

        let sample_label = text(format!("Sample: {}", bt.sample_name))
            .size(14)
            .color(theme::CONTENT_TEXT);

        let play_row = row![
            widgets::with_tooltip(
                button(text("Play A").size(14))
                    .on_press(Message::PlayBlindA)
                    .padding([8, 20]),
                blind_test_play_tooltip("A"),
            ),
            widgets::with_tooltip(
                button(text("Play B").size(14))
                    .on_press(Message::PlayBlindB)
                    .padding([8, 20]),
                blind_test_play_tooltip("B"),
            ),
        ]
        .spacing(20);

        let mut content_col = column![title, sample_label, play_row].spacing(12);

        if bt.revealed {
            let answer = if bt.ai_is_b { "B" } else { "A" };
            let result_color = if bt.last_correct {
                theme::ACCENT_GREEN
            } else {
                theme::ACCENT_AMBER
            };
            content_col = content_col.push(
                text(format!("The AI version was: {answer}"))
                    .size(16)
                    .color(result_color)
                    .font(iced::Font {
                        weight: iced::font::Weight::Bold,
                        ..Default::default()
                    }),
            );
            content_col = content_col.push(
                text(format!("Score: {} / {} correct", bt.correct, bt.total))
                    .size(13)
                    .color(theme::PANEL_LABEL),
            );
            content_col = content_col.push(
                row![
                    widgets::with_tooltip(
                        button(text("Next Round").size(13))
                            .on_press(Message::NextBlindTest)
                            .padding([6, 16]),
                        "Start another blind test round.",
                    ),
                    widgets::with_tooltip(
                        button(text("Close").size(13))
                            .on_press(Message::DismissBlindTest)
                            .padding([6, 16]),
                        "Close the blind test dialog.",
                    ),
                ]
                .spacing(12),
            );
        } else {
            content_col = content_col.push(
                text("Which one is the AI-remastered version?")
                    .size(13)
                    .color(theme::PANEL_LABEL),
            );
            content_col = content_col.push(
                row![
                    widgets::with_tooltip(
                        button(text("A is AI").size(13))
                            .on_press(Message::GuessBlind(true))
                            .padding([6, 16]),
                        blind_test_guess_tooltip("A"),
                    ),
                    widgets::with_tooltip(
                        button(text("B is AI").size(13))
                            .on_press(Message::GuessBlind(false))
                            .padding([6, 16]),
                        blind_test_guess_tooltip("B"),
                    ),
                ]
                .spacing(12),
            );
            content_col = content_col.push(widgets::with_tooltip(
                button(text("Cancel").size(11).color(theme::PANEL_LABEL))
                    .on_press(Message::DismissBlindTest)
                    .padding([4, 12]),
                "Leave the blind test without scoring this round.",
            ));
        }

        container(
            container(content_col.align_x(iced::Alignment::Center))
                .padding(20)
                .style(move |_theme: &Theme| container::Style {
                    background: Some(iced::Background::Color(theme::PANEL_SURFACE)),
                    border: iced::Border::default()
                        .rounded(8)
                        .color(theme::ACCENT_AMBER)
                        .width(1),
                    ..Default::default()
                })
                .max_width(400),
        )
        .width(Length::Fill)
        .height(Length::Fill)
        .center_x(Length::Fill)
        .center_y(Length::Fill)
        .style(|_theme: &Theme| container::Style {
            background: Some(iced::Background::Color(iced::Color::from_rgba(
                0.0, 0.0, 0.0, 0.7,
            ))),
            ..Default::default()
        })
        .into()
    }

    fn view_remaster_interrupt_dialog(&self) -> Element<'_, Message> {
        let Some(action) = self.remaster_interrupt_action.as_ref() else {
            return text("").into();
        };

        let (title, body, confirm_label) = match action {
            RemasterInterruptAction::Close(_) => (
                "Remaster In Progress".to_string(),
                "Cancel the current remaster and exit Quinlight?".to_string(),
                "Cancel Remaster & Exit".to_string(),
            ),
            RemasterInterruptAction::LoadFile(path) => {
                let name = path
                    .file_name()
                    .and_then(|name| name.to_str())
                    .unwrap_or("selected file");
                (
                    "Remaster In Progress".to_string(),
                    format!("Cancel the current remaster and load {name}?"),
                    "Cancel Remaster & Load".to_string(),
                )
            }
        };

        let keep_remastering = widgets::with_tooltip(
            button(text("Keep Remastering").size(13))
                .on_press(Message::DismissRemasterInterrupt)
                .padding([6, 16]),
            "Keep the current remaster running.",
        );
        let confirm_action = widgets::with_tooltip(
            button(text(confirm_label).size(13))
                .on_press(Message::ConfirmRemasterInterrupt)
                .padding([6, 16]),
            remaster_interrupt_confirm_tooltip(action),
        );

        let dialog = container(
            column![
                text(title).size(20).color(theme::ACCENT_AMBER),
                text(body).size(13).color(theme::CONTENT_TEXT),
                row![keep_remastering, confirm_action]
                    .spacing(12)
                    .align_y(iced::Alignment::Center),
            ]
            .spacing(12)
            .align_x(iced::Alignment::Center),
        )
        .padding(24)
        .max_width(500)
        .style(|_theme: &Theme| container::Style {
            background: Some(iced::Background::Color(theme::PANEL_SURFACE)),
            border: iced::Border {
                color: theme::ACCENT_AMBER,
                width: 1.0,
                radius: 8.0.into(),
            },
            ..Default::default()
        });

        container(dialog)
            .width(Length::Fill)
            .height(Length::Fill)
            .center_x(Length::Fill)
            .center_y(Length::Fill)
            .style(|_theme: &Theme| container::Style {
                background: Some(iced::Background::Color(iced::Color::from_rgba(
                    0.0, 0.0, 0.0, 0.7,
                ))),
                ..Default::default()
            })
            .into()
    }

    fn view_audio_error_dialog(&self) -> Element<'_, Message> {
        let error_msg = self.audio_error.as_deref().unwrap_or("Unknown audio error");

        let mut dialog_content = column![
            text("Audio Device Error")
                .size(20)
                .color(theme::ACCENT_AMBER),
            container(text(error_msg).size(12).color(theme::CONTENT_TEXT))
                .padding([8, 12])
                .width(Length::Fill)
                .style(|_theme: &Theme| container::Style {
                    background: Some(iced::Background::Color(theme::PANEL_BG)),
                    border: iced::Border {
                        color: theme::ACCENT_AMBER,
                        width: 1.0,
                        radius: 4.0.into(),
                    },
                    ..Default::default()
                }),
            text(
                "Check that your audio device is available and not in use\nby another application."
            )
            .size(13)
            .color(theme::CONTENT_TEXT),
        ]
        .spacing(12)
        .align_x(iced::Alignment::Center);

        dialog_content = dialog_content.push(widgets::with_tooltip(
            button(text("Close").size(13))
                .on_press(Message::DismissAudioError)
                .padding([6, 20]),
            "Close the audio error dialog.",
        ));

        let dialog =
            container(dialog_content)
                .padding(24)
                .max_width(520)
                .style(|_theme: &Theme| container::Style {
                    background: Some(iced::Background::Color(theme::PANEL_SURFACE)),
                    border: iced::Border {
                        color: theme::ACCENT_AMBER,
                        width: 1.0,
                        radius: 8.0.into(),
                    },
                    ..Default::default()
                });

        container(dialog)
            .width(Length::Fill)
            .height(Length::Fill)
            .center_x(Length::Fill)
            .center_y(Length::Fill)
            .style(|_theme: &Theme| container::Style {
                background: Some(iced::Background::Color(iced::Color::from_rgba(
                    0.0, 0.0, 0.0, 0.7,
                ))),
                ..Default::default()
            })
            .into()
    }

    fn view_install_dialog(&self) -> Element<'_, Message> {
        let detected = self.remaster_engine.available_engine_names();
        let smoke_status = if detected.is_empty() {
            "Current smoke check: no engines detected".to_string()
        } else {
            format!("Current smoke check: {}", detected.join(", "))
        };

        let title = self
            .install_dialog_title
            .clone()
            .unwrap_or_else(|| "No Quinlight Engines Installed".to_string());

        let dialog = container(
            column![
                text(title).size(20).color(theme::ACCENT_AMBER),
                text(crate::remaster::install_instructions())
                    .size(13)
                    .color(theme::CONTENT_TEXT),
                container(
                    text(crate::remaster::INSTALL_COMMAND)
                        .size(14)
                        .color(theme::ACCENT_GREEN)
                )
                .width(Length::Fill)
                .padding([8, 12])
                .style(|_theme: &Theme| container::Style {
                    background: Some(iced::Background::Color(theme::PANEL_BG)),
                    border: iced::Border {
                        color: theme::PANEL_LABEL,
                        width: 1.0,
                        radius: 4.0.into(),
                    },
                    ..Default::default()
                }),
                widgets::with_tooltip(
                    button(text("Copy Command").size(12))
                        .on_press(Message::CopyInstallCommand)
                        .padding([6, 16]),
                    "Copy the installer command to the clipboard.",
                ),
                text(smoke_status).size(11).color(theme::PANEL_LABEL),
                text("Restart Quinlight after the installer finishes to refresh engine detection.")
                    .size(11)
                    .color(theme::PANEL_LABEL),
                widgets::with_tooltip(
                    button(text("Close").size(13))
                        .on_press(Message::DismissInstallDialog)
                        .padding([6, 20]),
                    "Close the install instructions.",
                ),
            ]
            .spacing(12)
            .align_x(iced::Alignment::Center),
        )
        .padding(24)
        .max_width(500)
        .style(|_theme: &Theme| container::Style {
            background: Some(iced::Background::Color(theme::PANEL_SURFACE)),
            border: iced::Border {
                color: theme::ACCENT_AMBER,
                width: 1.0,
                radius: 8.0.into(),
            },
            ..Default::default()
        });

        // Semi-transparent backdrop + centered dialog
        container(dialog)
            .width(Length::Fill)
            .height(Length::Fill)
            .center_x(Length::Fill)
            .center_y(Length::Fill)
            .style(|_theme: &Theme| container::Style {
                background: Some(iced::Background::Color(iced::Color::from_rgba(
                    0.0, 0.0, 0.0, 0.7,
                ))),
                ..Default::default()
            })
            .into()
    }

    // ── Render export helpers ──────────────────────────────────────────

    fn start_render_export(&mut self, ext: &str, filter_label: &'static str) -> Task<Message> {
        let Some(ref orig_path) = self.loaded_path else {
            return Task::none();
        };
        if self.render_progress_rx.is_some() {
            // Already rendering
            return Task::none();
        }
        let render_rate = default_render_rate(self.player.current_playback_rate());
        let default_name = default_render_save_name(orig_path, ext, render_rate);
        let default_dir = dirs::audio_dir()
            .or_else(dirs::home_dir)
            .unwrap_or_else(|| PathBuf::from("/tmp"))
            .join("Quinlight Audio");
        let _ = std::fs::create_dir_all(&default_dir);
        let stereo_sep = self.stereo_separation;
        let interpolation = self.interpolation.to_filter_length();
        let agc_enabled = self.agc_enabled;
        let hrtf_enabled = self.hrtf_enabled;
        let hrtf_mix = self.hrtf_mix;
        let render_handle = self.player.render_handle();

        let (progress_tx, progress_rx) = crossbeam_channel::bounded::<f32>(64);
        self.render_progress_rx = Some(progress_rx);
        self.render_progress_pct = 0.0;
        self.render_format_label = filter_label;

        let is_aac = ext == "m4a";
        let metadata = crate::render::AudioMetadata {
            title: self
                .module_info
                .as_ref()
                .map(|i| i.title.clone())
                .unwrap_or_default(),
            artist: self
                .module_info
                .as_ref()
                .map(|i| i.artist.clone())
                .unwrap_or_default(),
            album: "Quinlight Audio".into(),
        };
        let ext_owned = ext.to_string();
        let filters: &[&str] = if is_aac { &["m4a"] } else { &["flac"] };
        let filters_owned: Vec<String> = filters.iter().map(|s| s.to_string()).collect();

        let complete_msg = if is_aac {
            Message::RenderAacComplete as fn(Result<PathBuf, String>) -> Message
        } else {
            Message::RenderFlacComplete as fn(Result<PathBuf, String>) -> Message
        };

        Task::perform(
            async move {
                let filter_refs: Vec<&str> = filters_owned.iter().map(|s| s.as_str()).collect();
                let Some(handle) = rfd::AsyncFileDialog::new()
                    .add_filter(ext_owned.to_uppercase(), &filter_refs)
                    .set_file_name(&default_name)
                    .set_directory(&default_dir)
                    .save_file()
                    .await
                else {
                    return Err("Cancelled".into());
                };
                let out_path = handle.path().to_path_buf();

                // Phase 1: render module to samples (0% - 50%)
                let mut samples = render_handle.render_live_to_samples(
                    stereo_sep,
                    interpolation,
                    agc_enabled,
                    render_rate,
                    Some((&progress_tx, 0.0, 0.5)),
                )?;

                // Phase 1.5: HRTF binaural processing (offline)
                if hrtf_enabled {
                    crate::hrtf::process_offline_with_mix(&mut samples, render_rate, hrtf_mix)
                        .unwrap_or_else(|e| eprintln!("HRTF offline failed: {e}"));
                }

                // Phase 2: encode to file (50% - 100%)
                if is_aac {
                    crate::render::encode_samples_to_aac(
                        samples,
                        &out_path,
                        render_rate,
                        &metadata,
                        &progress_tx,
                        0.5,
                        1.0,
                    )?;
                } else {
                    crate::render::encode_samples_to_flac(
                        samples,
                        &out_path,
                        render_rate,
                        &metadata,
                        &progress_tx,
                        0.5,
                        1.0,
                    )?;
                }
                Ok(out_path)
            },
            complete_msg,
        )
    }

    fn finish_render_export(&mut self, result: Result<PathBuf, String>) {
        self.render_progress_rx = None;
        self.render_progress_pct = 0.0;
        match result {
            Ok(path) => {
                self.remaster_notice = Some(format!(
                    "Rendered {} to {}",
                    self.render_format_label,
                    path.display()
                ));
            }
            Err(e) if e == "Cancelled" => {}
            Err(e) => {
                self.remaster_notice = None;
                self.remaster_status = RemasterStatus::Failed(format!("Render failed: {e}"));
            }
        }
    }

    fn view_render_progress_dialog(&self) -> Element<'_, Message> {
        let pct = (self.render_progress_pct * 100.0).round() as u32;
        let dialog = container(
            column![
                text(format!("Rendering {}...", self.render_format_label))
                    .size(16)
                    .color(theme::ACCENT_AMBER),
                text(format!("{pct}%")).size(13).color(theme::CONTENT_TEXT),
                progress_bar(0.0..=1.0, self.render_progress_pct)
                    .height(10)
                    .style(|_theme: &Theme| progress_bar::Style {
                        background: iced::Background::Color(theme::PANEL_SURFACE),
                        bar: iced::Background::Color(theme::ACCENT_GREEN),
                        border: iced::Border::default(),
                    }),
            ]
            .spacing(10)
            .align_x(iced::Alignment::Center),
        )
        .padding(24)
        .max_width(360)
        .style(|_theme: &Theme| container::Style {
            background: Some(iced::Background::Color(theme::PANEL_SURFACE)),
            border: iced::Border {
                color: theme::ACCENT_AMBER,
                width: 1.0,
                radius: 8.0.into(),
            },
            ..Default::default()
        });

        container(dialog)
            .width(Length::Fill)
            .height(Length::Fill)
            .center_x(Length::Fill)
            .center_y(Length::Fill)
            .style(|_theme: &Theme| container::Style {
                background: Some(iced::Background::Color(iced::Color::from_rgba(
                    0.0, 0.0, 0.0, 0.7,
                ))),
                ..Default::default()
            })
            .into()
    }

    fn view_archive_picker(&self) -> Element<'_, Message> {
        let archive_name = self
            .archive_path
            .as_ref()
            .and_then(|p| p.file_name())
            .and_then(|n| n.to_str())
            .unwrap_or("archive");

        let mut entries_col = Column::new().spacing(2);
        for (i, entry) in self.archive_entries.iter().enumerate() {
            entries_col = entries_col.push(widgets::with_tooltip_at(
                button(row![text(&entry.filename).size(13).width(Length::Fill),].spacing(8))
                    .on_press(Message::ArchiveSelect(i))
                    .padding([4, 12])
                    .width(Length::Fill)
                    .style(|_theme: &Theme, _status| button::Style {
                        background: Some(iced::Background::Color(theme::PANEL_BG)),
                        text_color: theme::CONTENT_TEXT,
                        border: iced::Border {
                            color: theme::PANEL_LABEL,
                            width: 1.0,
                            radius: 4.0.into(),
                        },
                        ..Default::default()
                    }),
                archive_entry_tooltip(&entry.filename),
                iced::widget::tooltip::Position::FollowCursor,
            ));
        }

        let dialog = container(
            column![
                text(format!("Select module from {archive_name}"))
                    .size(16)
                    .color(theme::ACCENT_AMBER),
                text(format!("{} modules found", self.archive_entries.len()))
                    .size(11)
                    .color(theme::PANEL_LABEL),
                scrollable(entries_col).height(Length::FillPortion(1)),
                widgets::with_tooltip(
                    button(text("Cancel").size(13))
                        .on_press(Message::DismissArchivePicker)
                        .padding([6, 20]),
                    "Close the archive picker.",
                ),
            ]
            .spacing(10)
            .align_x(iced::Alignment::Center),
        )
        .padding(20)
        .max_width(500)
        .max_height(500)
        .style(|_theme: &Theme| container::Style {
            background: Some(iced::Background::Color(theme::PANEL_SURFACE)),
            border: iced::Border {
                color: theme::ACCENT_AMBER,
                width: 1.0,
                radius: 8.0.into(),
            },
            ..Default::default()
        });

        container(dialog)
            .width(Length::Fill)
            .height(Length::Fill)
            .center_x(Length::Fill)
            .center_y(Length::Fill)
            .style(|_theme: &Theme| container::Style {
                background: Some(iced::Background::Color(iced::Color::from_rgba(
                    0.0, 0.0, 0.0, 0.7,
                ))),
                ..Default::default()
            })
            .into()
    }

    pub fn subscription(&self) -> Subscription<Message> {
        // 120 Hz tick — covers 60/120 Hz monitors with headroom; higher rates
        // saturate the per-tick canvas redraw and starve the audio callback.
        Subscription::batch([
            iced::time::every(Duration::from_micros(8_333)).map(|_| Message::Tick),
            iced::window::resize_events().map(|(_id, size)| Message::WindowResized(size)),
            iced::window::close_requests().map(Message::CloseRequested),
            iced::keyboard::on_key_press(|key, modifiers| {
                Some(Message::KeyPressed(key, modifiers))
            }),
            // Drag and drop
            iced::event::listen_with(|event, _status, _id| {
                if let iced::Event::Window(iced::window::Event::FileDropped(path)) = event {
                    Some(Message::FileDropped(path))
                } else {
                    None
                }
            }),
        ])
    }

    pub fn theme(&self) -> Theme {
        theme::quinlight_theme()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;
    use std::process::Child;

    const XM_FIXTURE: &str = "openmpt/test/test.xm";
    const COMMAND_NOTE: i32 = 0;
    const COMMAND_INSTRUMENT: i32 = 1;
    const COMMAND_EFFECT: i32 = 3;
    const COMMAND_PARAMETER: i32 = 5;
    const CMD_OFFSET: u8 = 10;

    fn original_signal() -> Vec<f64> {
        (0..4096)
            .map(|i| (i as f64 * 220.0 * 2.0 * std::f64::consts::PI / 48_000.0).sin())
            .collect()
    }

    /// Test-only wrapper: forces `is_playing = false` so the channel-state
    /// refresh is bypassed (these tests don't exercise live audio).
    fn apply_sample_mode_to_slot_t(
        module: &mut Module,
        slot: &mut SampleSlot,
        target_mode: SampleMode,
    ) -> Result<(), String> {
        apply_sample_mode_to_slot(module, slot, target_mode, false)
    }

    fn sample_slot() -> SampleSlot {
        SampleSlot {
            index: 7,
            original: original_signal(),
            original_rate: 48_000,
            original_channels: 1,
            original_length_frames: 4096,
            loop_info: crate::openmpt::SampleLoopInfo::none(),
            engine_results: Vec::new(),
            quinlight_result: None,
            quinlight_original_fallback: false,
            mode: SampleMode::Original,
            failed: false,
            original_effects: Vec::new(),
            engines_done: 0,
            engines_total: 3,
            muted: false,
            reference_48k: None,
        }
    }

    struct GuiDummyEngine;

    impl crate::engine::UpsampleEngine for GuiDummyEngine {
        fn name(&self) -> &str {
            "AudioSR"
        }

        fn cache_id(&self) -> &str {
            "audiosr-v0.1"
        }

        fn output_rate(&self) -> u32 {
            48_000
        }

        fn max_batch_size(&self) -> usize {
            1
        }

        fn min_duration_secs(&self) -> f64 {
            0.0
        }

        fn spawn_batch(
            &self,
            _input_manifest: &Path,
            _output_dir: &Path,
            _device: &str,
            _ddim_steps: u32,
            _cpu_thread_budget: usize,
        ) -> Result<Child, String> {
            unreachable!("gui dummy engine should not spawn")
        }

        fn find_output_wav(&self, _output_dir: &Path, _stem: &str) -> Result<PathBuf, String> {
            unreachable!("gui dummy engine should not read output")
        }
    }

    struct GuiLavaRateLimitedEngine;

    impl crate::engine::UpsampleEngine for GuiLavaRateLimitedEngine {
        fn name(&self) -> &str {
            "LavaSR"
        }

        fn cache_id(&self) -> &str {
            "lavasr-v0.1"
        }

        fn supports_original_rate(&self, original_rate_hz: u32) -> bool {
            original_rate_hz <= 16_000
        }

        fn output_rate(&self) -> u32 {
            48_000
        }

        fn max_batch_size(&self) -> usize {
            1
        }

        fn min_duration_secs(&self) -> f64 {
            0.0
        }

        fn spawn_batch(
            &self,
            _input_manifest: &Path,
            _output_dir: &Path,
            _device: &str,
            _ddim_steps: u32,
            _cpu_thread_budget: usize,
        ) -> Result<Child, String> {
            unreachable!("gui lava test engine should not spawn")
        }

        fn find_output_wav(&self, _output_dir: &Path, _stem: &str) -> Result<PathBuf, String> {
            unreachable!("gui lava test engine should not read output")
        }
    }

    fn quinlight_app() -> Quinlight {
        let detect_handle = std::thread::spawn(RemasterEngine::empty);
        let flag = Arc::new(AtomicBool::new(false));
        let (app, _task) = Quinlight::new(UpscaleMode::CpuOnly, detect_handle, flag, None);
        app
    }

    fn quinlight_app_with_engine() -> Quinlight {
        let detect_handle = std::thread::spawn(|| {
            RemasterEngine::from_test_engines(vec![Box::new(GuiDummyEngine)])
        });
        let flag = Arc::new(AtomicBool::new(false));
        let (app, _task) = Quinlight::new(UpscaleMode::CpuOnly, detect_handle, flag, None);
        app
    }

    fn fixture_prepared_load_handle(
        app: &Quinlight,
        loaded_path: PathBuf,
    ) -> (PreparedModuleLoadHandle, Vec<u8>) {
        let file_data = std::fs::read("mods/2ND_PM.S3M").expect("fixture should exist");
        let prepared = crate::player::prepare_module_load_from_bytes(file_data.clone())
            .expect("fixture should prepare");
        let handle = PreparedModuleLoadHandle::new(PreparedModuleLoadPackage {
            request_id: app.active_module_load_request,
            context: PendingModuleLoadContext {
                loaded_path: Some(loaded_path),
                stereo_separation: crate::openmpt::MOD_STEREO_SEPARATION_PERCENT,
            },
            prepared,
        });
        (handle, file_data)
    }

    #[test]
    fn aniso64_interpolation_choice_uses_shared_default_filter_length() {
        assert_eq!(InterpolationChoice::Aniso64.to_string(), "Aniso-64");
        assert_eq!(
            InterpolationChoice::Aniso64.to_filter_length(),
            crate::openmpt::DEFAULT_INTERPOLATION_FILTER_LENGTH
        );
    }

    #[test]
    fn quinlight_app_defaults_to_aniso64_interpolation() {
        let app = quinlight_app();
        assert_eq!(app.interpolation, InterpolationChoice::Aniso64);
    }

    #[test]
    fn render_save_name_uses_render_rate() {
        assert_eq!(
            render_save_name("2ND_PM", "flac", 96_000),
            "2ND_PM-Quinlight-Audio-Remastered-96Khz.flac"
        );
        assert_eq!(
            render_save_name("2ND_PM", "m4a", 48_000),
            "2ND_PM-Quinlight-Audio-Remastered-48Khz.m4a"
        );
        assert_eq!(
            render_save_name("2ND_PM", "flac", 192_000),
            "2ND_PM-Quinlight-Audio-Remastered-192Khz.flac"
        );
        assert_eq!(
            render_save_name("2ND_PM", "flac", 44_100),
            "2ND_PM-Quinlight-Audio-Remastered-44.1Khz.flac"
        );
    }

    #[test]
    fn default_render_save_name_uses_playback_rate() {
        let path = Path::new("/tmp/2ND_PM.S3M");
        assert_eq!(
            default_render_save_name(path, "flac", 96_000),
            "2ND_PM-Quinlight-Audio-Remastered-96Khz.flac"
        );
    }

    #[test]
    fn default_render_rate_falls_back_when_playback_rate_is_missing() {
        assert_eq!(default_render_rate(0), 48_000);
    }

    fn engine_result(name: &str, data: Vec<f64>) -> SampleResult {
        SampleResult {
            index: 7,
            length_frames: data.len() as i64,
            channels: 1,
            data,
            sample_rate_hz: 48_000,
            engine_name: name.to_string(),
            discovered_loops: None,
        }
    }

    fn slot_result(slot: &SampleSlot, name: &str, data: Vec<f64>) -> SampleResult {
        SampleResult {
            index: slot.index,
            length_frames: data.len() as i64 / slot.original_channels.max(1) as i64,
            channels: slot.original_channels,
            data,
            sample_rate_hz: 48_000,
            engine_name: name.to_string(),
            discovered_loops: None,
        }
    }

    fn xm_offset_slot_fixture() -> (Module, SampleSlot, u8) {
        let data = std::fs::read(XM_FIXTURE).expect("Failed to read XM fixture");
        let mut module = Module::from_memory(&data).expect("Failed to load XM fixture");
        let mapped_note = (1u8..=120)
            .find(|&note| module.instrument_sample_for_note(0, note) == Some(1))
            .expect("Fixture instrument should map at least one note to sample 2");

        assert!(
            module.pattern_num_rows(0) > 0,
            "XM fixture should have pattern data"
        );
        assert!(module.set_pattern_command(0, 0, 0, COMMAND_NOTE, mapped_note));
        assert!(module.set_pattern_command(0, 0, 0, COMMAND_INSTRUMENT, 1));
        assert!(module.set_pattern_command(0, 0, 0, COMMAND_EFFECT, CMD_OFFSET));
        assert!(module.set_pattern_command(0, 0, 0, COMMAND_PARAMETER, 0x10));

        let original = original_signal();
        let original_rate = 16_000;
        let expected_param = ((0x10u32 * 48_000) / original_rate as u32).min(255) as u8;
        let slot = SampleSlot {
            index: 1,
            original,
            original_rate,
            original_channels: 1,
            original_length_frames: 4096,
            loop_info: crate::openmpt::SampleLoopInfo::none(),
            engine_results: Vec::new(),
            quinlight_result: None,
            quinlight_original_fallback: false,
            mode: SampleMode::Original,
            failed: false,
            original_effects: Vec::new(),
            engines_done: 0,
            engines_total: 1,
            muted: false,
            reference_48k: None,
        };

        (module, slot, expected_param)
    }

    fn expected_quinlight_mix(slot: &SampleSlot) -> crate::remaster::QuinlightMix {
        let reference = crate::remaster::build_quinlight_reference_48k_with_loop_info(
            &slot.original,
            slot.original_rate as u32,
            slot.original_channels as usize,
            slot.loop_info,
            CleanupSettings::default(),
        )
        .expect("Should build reference");
        let engines: Vec<(String, Vec<f64>, i64, i32)> = slot
            .engine_results
            .iter()
            .map(|er| {
                (
                    er.name.clone(),
                    er.data.clone(),
                    er.length_frames,
                    er.channels,
                )
            })
            .collect();
        let looped = slot.loop_info.normal.mode != crate::openmpt::SampleLoopMode::None
            || slot.loop_info.sustain.mode != crate::openmpt::SampleLoopMode::None;
        crate::remaster::select_quinlight_mix(
            &reference,
            slot.original_channels,
            slot.original_rate as u32,
            &engines,
            engines.len(),
            looped,
        )
    }

    fn expected_quinlight_result(slot: &SampleSlot) -> SampleResult {
        let mix = expected_quinlight_mix(slot);
        SampleResult {
            index: slot.index,
            data: mix.data,
            length_frames: mix.length_frames,
            channels: mix.channels,
            sample_rate_hz: 48_000,
            engine_name: mix.name,
            discovered_loops: None,
        }
    }

    fn assert_close(actual: f32, expected: f32) {
        assert!(
            (actual - expected).abs() < 0.01,
            "expected {expected}, got {actual}"
        );
    }

    #[test]
    fn resolve_panel_heights_preserves_preferences_when_budget_is_sufficient() {
        let preferred = PanelHeights::new(200.0, 120.0, 300.0);
        let resolved = resolve_panel_heights(preferred, preferred.total() + 64.0);

        assert_close(resolved.pattern_viewer, preferred.pattern_viewer);
        assert_close(resolved.vu_meters, preferred.vu_meters);
        assert_close(resolved.sample_panel, preferred.sample_panel);
    }

    #[test]
    fn resolve_panel_heights_shrinks_proportionally_without_violating_minimums() {
        let preferred = PanelHeights::new(200.0, 120.0, 300.0);
        let available = PanelHeights::minimums().total() + 80.0;
        let resolved = resolve_panel_heights(preferred, available);
        let minimums = PanelHeights::minimums();

        assert!(resolved.pattern_viewer >= minimums.pattern_viewer);
        assert!(resolved.vu_meters >= minimums.vu_meters);
        assert!(resolved.sample_panel >= minimums.sample_panel);
        assert_close(resolved.total(), available);

        let scale = (available - minimums.total()) / (preferred.total() - minimums.total());
        assert_close(
            resolved.pattern_viewer,
            minimums.pattern_viewer + (preferred.pattern_viewer - minimums.pattern_viewer) * scale,
        );
        assert_close(
            resolved.vu_meters,
            minimums.vu_meters + (preferred.vu_meters - minimums.vu_meters) * scale,
        );
        assert_close(
            resolved.sample_panel,
            minimums.sample_panel + (preferred.sample_panel - minimums.sample_panel) * scale,
        );
    }

    #[test]
    fn resolve_panel_heights_restore_preferred_when_budget_returns() {
        let preferred = PanelHeights::new(200.0, 120.0, 300.0);

        let _shrunk = resolve_panel_heights(preferred, PanelHeights::minimums().total() + 40.0);
        let restored = resolve_panel_heights(preferred, preferred.total());

        assert_close(restored.pattern_viewer, preferred.pattern_viewer);
        assert_close(restored.vu_meters, preferred.vu_meters);
        assert_close(restored.sample_panel, preferred.sample_panel);
    }

    #[test]
    fn resolve_panel_heights_total_never_exceeds_available_budget() {
        let preferred = PanelHeights::new(200.0, 120.0, 300.0);
        let available = PanelHeights::minimums().total() + 137.0;
        let resolved = resolve_panel_heights(preferred, available);

        assert!(resolved.total() <= available + 0.01);
    }

    #[test]
    fn enabled_engine_names_reflect_gui_toggles() {
        assert_eq!(
            enabled_engine_names(true, false, true, false),
            vec!["FLowHigh".to_string(), "AudioSR".to_string()]
        );
        assert_eq!(
            enabled_engine_names(true, true, true, true),
            vec![
                "LavaSR".to_string(),
                "FLowHigh".to_string(),
                "AP-BWE".to_string(),
                "AudioSR".to_string(),
            ]
        );
        assert!(enabled_engine_names(false, false, false, false).is_empty());
    }

    #[test]
    fn render_button_tooltips_explain_loaded_state() {
        assert_eq!(
            render_button_tooltip("FLAC", false),
            "Load a module to render a FLAC file."
        );
        assert_eq!(
            render_button_tooltip("M4A", true),
            "Render the current module to an M4A file."
        );
    }

    #[test]
    fn blind_test_button_tooltips_explain_availability() {
        assert_eq!(
            blind_test_button_tooltip(false),
            "Remaster at least one sample to unlock blind test."
        );
        assert_eq!(
            blind_test_button_tooltip(true),
            "Start an A/B blind test for remastered samples."
        );
    }

    #[test]
    fn remaster_primary_tooltips_cover_all_states() {
        assert_eq!(
            remaster_primary_tooltip(RemasterPrimaryAction::Install),
            "Show AI engine install instructions."
        );
        assert_eq!(
            remaster_primary_tooltip(RemasterPrimaryAction::Start),
            "Start remastering the loaded module."
        );
        assert_eq!(
            remaster_primary_tooltip(RemasterPrimaryAction::Cancel),
            "Cancel the current remaster run."
        );
        assert_eq!(
            remaster_primary_tooltip(RemasterPrimaryAction::Cancelling),
            "Remaster cancellation is in progress."
        );
        assert_eq!(
            remaster_primary_tooltip(RemasterPrimaryAction::Complete),
            "Remastering finished for the current module."
        );
        assert_eq!(
            remaster_primary_tooltip(RemasterPrimaryAction::Disabled),
            "Load a module to start remastering."
        );
    }

    #[test]
    fn sample_tooltips_describe_mode_cycle_and_keyjazz_selection() {
        let mut slot = sample_slot();
        let original = slot.original.clone();
        slot.apply_candidate_result(&engine_result("AudioSR", original), Vec::new());
        assert_eq!(
            sample_mode_tooltip(&slot),
            "Cycle sample 8 from original audio to AudioSR."
        );

        let sample = crate::openmpt::SampleInfo {
            index: 7,
            name: "Kick".into(),
            rate: 48_000,
            length_frames: 4096,
            channels: 1,
            bits_per_sample: 16,
            sample_format: crate::openmpt::SampleFormat::Int16,
            has_loop: false,
            loop_start_frames: 0,
            loop_end_frames: 0,
            loop_mode: crate::openmpt::SampleLoopMode::None,
            has_sustain_loop: false,
            sustain_loop_start_frames: 0,
            sustain_loop_end_frames: 0,
            sustain_loop_mode: crate::openmpt::SampleLoopMode::None,
        };
        assert_eq!(
            sample_row_tooltip(&sample, false),
            "Select sample 8 \"Kick\" for keyjazz preview."
        );
        assert_eq!(
            sample_row_tooltip(&sample, true),
            "Sample 8 \"Kick\" is selected for keyjazz preview."
        );
    }

    #[test]
    fn sample_mode_labels_distinguish_manual_original_from_quinlight_fallback() {
        let mut slot = sample_slot();
        assert_eq!(sample_mode_button_label(&slot), "Original");
        assert_eq!(
            sample_mode_name_for_slot(&slot, &SampleMode::Original),
            "original audio"
        );
        assert_eq!(sample_mode_button_color(&slot), theme::ACCENT_AMBER);

        slot.quinlight_original_fallback = true;
        assert_eq!(sample_mode_button_label(&slot), QUINLIGHT_ORIGINAL_TAG);
        assert_eq!(
            sample_mode_name_for_slot(&slot, &SampleMode::Original),
            QUINLIGHT_ORIGINAL_TAG
        );
        assert_eq!(sample_mode_button_color(&slot), theme::PANEL_LABEL);
    }

    #[test]
    fn reference_48k_sits_between_engines_and_original_in_cycle_order() {
        let mut slot = sample_slot();
        slot.quinlight_result = Some(EngineResult {
            name: QUINLIGHT_ENGINE_NAME.into(),
            data: slot.original.clone(),
            length_frames: slot.original_length_frames,
            channels: slot.original_channels,
            sample_rate_hz: 48_000,
            discovered_loops: None,
        });
        slot.engine_results.push(EngineResult {
            name: "AudioSR".into(),
            data: slot.original.clone(),
            length_frames: slot.original_length_frames,
            channels: slot.original_channels,
            sample_rate_hz: 48_000,
            discovered_loops: None,
        });

        assert_eq!(
            slot.available_modes(),
            vec![
                SampleMode::Engine(QUINLIGHT_ENGINE_NAME.into()),
                SampleMode::Engine("AudioSR".into()),
                SampleMode::Reference48k,
                SampleMode::Original,
            ],
        );

        slot.mode = SampleMode::Engine("AudioSR".into());
        assert_eq!(slot.next_mode(), SampleMode::Reference48k);

        slot.mode = SampleMode::Reference48k;
        assert_eq!(sample_mode_button_label(&slot), "Ref 48k");
        assert_eq!(
            sample_mode_name_for_slot(&slot, &SampleMode::Reference48k),
            "Reference 48k (SINC)"
        );
        assert_eq!(sample_mode_button_color(&slot), theme::PANEL_LABEL);
        assert_eq!(slot.next_mode(), SampleMode::Original);

        slot.mode = SampleMode::Original;
        assert_eq!(
            slot.next_mode(),
            SampleMode::Engine(QUINLIGHT_ENGINE_NAME.into())
        );
    }

    #[test]
    fn archive_entry_tooltip_mentions_filename() {
        assert_eq!(
            archive_entry_tooltip("mods/demo.mod"),
            "Load \"mods/demo.mod\" from the archive."
        );
    }

    #[test]
    fn cleanup_mode_and_engine_update_independently() {
        let mut app = quinlight_app();

        assert_eq!(app.cleanup_settings.mode, CleanupMode::Off);
        assert_eq!(
            app.cleanup_settings.engine_version,
            CleanupEngineVersion::V21
        );

        let _ = app.update(Message::SetCleanupMode(CleanupMode::Decrackle));
        assert_eq!(app.cleanup_settings.mode, CleanupMode::Decrackle);
        assert_eq!(
            app.cleanup_settings.engine_version,
            CleanupEngineVersion::V21
        );

        let _ = app.update(Message::SetCleanupEngineVersion(CleanupEngineVersion::V1));
        assert_eq!(app.cleanup_settings.mode, CleanupMode::Decrackle);
        assert_eq!(
            app.cleanup_settings.engine_version,
            CleanupEngineVersion::V1
        );
    }

    #[test]
    fn remaster_primary_action_switches_to_cancel_states_while_active() {
        let mut app = quinlight_app_with_engine();

        app.remaster_status = RemasterStatus::Processing {
            current: 1,
            total: 3,
            sample_name: "Kick".into(),
        };
        assert_eq!(app.remaster_primary_action(), RemasterPrimaryAction::Cancel);

        app.remaster_status = RemasterStatus::Cancelling;
        assert_eq!(
            app.remaster_primary_action(),
            RemasterPrimaryAction::Cancelling
        );
    }

    #[test]
    fn direct_cancel_drops_receivers_and_restores_idle_state() {
        let mut app = quinlight_app_with_engine();
        let mut slot = sample_slot();
        slot.apply_candidate_result(
            &engine_result("AudioSR", slot.original.clone()),
            vec![(0, 0, 0, 0x12)],
        );
        app.sample_slots = vec![slot];
        app.blind_test = Some(BlindTest {
            sample_name: "Kick".into(),
            original_data: vec![0.1, 0.2],
            ai_data: vec![0.2, 0.1],
            ai_is_b: true,
            revealed: false,
            last_correct: false,
            correct: 0,
            total: 0,
        });

        let cancel_flag = Arc::new(AtomicBool::new(false));
        let (progress_tx, progress_rx) = crossbeam_channel::unbounded();
        let (result_tx, result_rx) = crossbeam_channel::unbounded();
        app.remaster_cancel_flag = Some(cancel_flag.clone());
        app.remaster_rx = Some(progress_rx);
        app.remaster_result_rx = Some(result_rx);
        app.remaster_status = RemasterStatus::Processing {
            current: 1,
            total: 3,
            sample_name: "Kick".into(),
        };

        let _ = app.update(Message::CancelRemaster);

        assert!(cancel_flag.load(Ordering::Relaxed));
        assert!(matches!(app.remaster_status, RemasterStatus::Cancelling));
        assert!(progress_tx.send(RemasterStatus::Complete).is_err());
        assert!(
            result_tx
                .send(RemasterOutput::Candidate(engine_result(
                    "AudioSR",
                    vec![0.0; 32]
                )))
                .is_err()
        );

        let _ = app.update(Message::RemasterUpdate(RemasterStatus::Cancelled));

        assert!(matches!(app.remaster_status, RemasterStatus::Ready));
        assert!(app.sample_slots.is_empty());
        assert!(app.blind_test.is_none());
        assert!(app.remaster_cancel_flag.is_none());
        assert_eq!(app.remaster_notice.as_deref(), Some("Remaster cancelled"));
    }

    #[test]
    fn shutdown_tick_requests_real_window_id_lookup() {
        let mut app = quinlight_app();

        app.shutdown_flag.store(true, Ordering::Relaxed);

        let _ = app.update(Message::Tick);

        assert!(app.shutdown_close_resolution_pending);
    }

    #[test]
    fn resolved_oldest_window_starts_close_sequence_when_idle() {
        let mut app = quinlight_app();
        let window_id = iced::window::Id::unique();

        app.shutdown_close_resolution_pending = true;

        let _ = app.update(Message::ResolvedOldestWindow(Some(window_id)));

        assert!(!app.shutdown_close_resolution_pending);
        assert!(app.closing);
        assert_eq!(app.closing_window, Some(window_id));
    }

    #[test]
    fn close_request_during_remaster_opens_interrupt_dialog() {
        let mut app = quinlight_app_with_engine();
        app.remaster_status = RemasterStatus::Processing {
            current: 1,
            total: 3,
            sample_name: "Kick".into(),
        };
        app.remaster_cancel_flag = Some(Arc::new(AtomicBool::new(false)));
        let window_id = iced::window::Id::unique();

        let _ = app.update(Message::CloseRequested(window_id));

        assert!(!app.closing);
        assert!(matches!(
            app.remaster_interrupt_action.as_ref(),
            Some(RemasterInterruptAction::Close(id)) if *id == window_id
        ));
    }

    #[test]
    fn confirming_close_during_remaster_cancels_then_starts_close_sequence() {
        let mut app = quinlight_app_with_engine();
        let cancel_flag = Arc::new(AtomicBool::new(false));
        let window_id = iced::window::Id::unique();
        app.remaster_status = RemasterStatus::Processing {
            current: 1,
            total: 3,
            sample_name: "Kick".into(),
        };
        app.remaster_cancel_flag = Some(cancel_flag.clone());
        app.remaster_interrupt_action = Some(RemasterInterruptAction::Close(window_id));

        let _ = app.update(Message::ConfirmRemasterInterrupt);

        assert!(cancel_flag.load(Ordering::Relaxed));
        assert!(matches!(app.remaster_status, RemasterStatus::Cancelling));
        assert!(matches!(
            app.pending_post_cancel_action.as_ref(),
            Some(RemasterInterruptAction::Close(id)) if *id == window_id
        ));

        let _ = app.update(Message::RemasterUpdate(RemasterStatus::Cancelled));

        assert!(app.closing);
        assert_eq!(app.closing_window, Some(window_id));
    }

    #[test]
    fn file_selection_during_remaster_cancels_then_loads_selected_file() {
        let mut app = quinlight_app_with_engine();
        let cancel_flag = Arc::new(AtomicBool::new(false));
        let path = PathBuf::from("/tmp/cancel-load.mod");
        app.remaster_status = RemasterStatus::Processing {
            current: 1,
            total: 3,
            sample_name: "Kick".into(),
        };
        app.remaster_cancel_flag = Some(cancel_flag.clone());

        let _ = app.update(Message::FileSelected(Some(path.clone())));
        assert!(matches!(
            app.remaster_interrupt_action.as_ref(),
            Some(RemasterInterruptAction::LoadFile(pending)) if pending == &path
        ));

        let _ = app.update(Message::ConfirmRemasterInterrupt);
        assert!(cancel_flag.load(Ordering::Relaxed));
        assert!(matches!(app.remaster_status, RemasterStatus::Cancelling));

        let _ = app.update(Message::RemasterUpdate(RemasterStatus::Cancelled));
        let (handle, _) = fixture_prepared_load_handle(&app, path.clone());
        let _ = app.update(Message::PreparedModuleLoadReady(
            PreparedModuleLoadOutcome::Success(handle),
        ));

        assert_eq!(app.loaded_path.as_ref(), Some(&path));
        assert!(app.remaster_interrupt_action.is_none());
        assert!(app.pending_post_cancel_action.is_none());
    }

    #[test]
    fn archive_backed_module_load_uses_in_memory_bytes_for_groove_render() {
        let mut app = quinlight_app();
        let virtual_path = PathBuf::from("/definitely/not/a/real/archive-entry.mod");
        let (handle, file_data) = fixture_prepared_load_handle(&app, virtual_path.clone());

        assert!(!virtual_path.exists());

        let pending = app
            .apply_prepared_module_load(handle)
            .expect("matching request should produce groove render input");

        assert_eq!(app.loaded_path.as_ref(), Some(&virtual_path));
        assert_eq!(pending.request_id, app.active_module_load_request);
        assert_eq!(pending.file_data.as_slice(), file_data.as_slice());

        let groove_data = vinyl_shader::render_groove_data(
            &pending.file_data,
            pending.stereo_separation,
            pending.interpolation_filter,
        )
        .expect("prepared bytes should render without reading from loaded_path");

        assert!(!groove_data.is_empty());
    }

    #[test]
    fn stale_groove_data_ready_is_ignored() {
        let mut app = quinlight_app();
        app.active_module_load_request = 7;
        app.groove_data = vec![0.25, 0.5];

        let _ = app.update(Message::GrooveDataReady {
            request_id: 6,
            data: vec![0.9, 1.0],
        });

        assert_eq!(app.groove_data, vec![0.25, 0.5]);

        let _ = app.update(Message::GrooveDataReady {
            request_id: 7,
            data: vec![0.1, 0.2],
        });

        assert_eq!(app.groove_data, vec![0.1, 0.2]);
    }

    #[test]
    fn reset_loaded_module_state_clears_spectrogram_and_groove_data() {
        let mut app = quinlight_app();
        let waveform: Vec<f64> = (0..2048)
            .flat_map(|i| {
                let sample = (i as f64 * 330.0 * 2.0 * std::f64::consts::PI / 48_000.0).sin() * 0.8;
                [sample, sample]
            })
            .collect();
        app.waveform = waveform.clone();
        app.groove_data = vec![0.2, 0.4, 0.6];
        app.spectrogram.push_fft(&waveform);

        assert!(
            app.spectrogram
                .magnitudes()
                .iter()
                .any(|&value| value > 0.0)
        );
        assert_eq!(app.spectrogram.write_index(), 1);

        app.reset_loaded_module_state();

        assert!(app.waveform.is_empty());
        assert!(app.groove_data.is_empty());
        assert_eq!(app.spectrogram.write_index(), 0);
        assert!(
            app.spectrogram
                .magnitudes()
                .iter()
                .all(|&value| value == 0.0)
        );
    }

    #[test]
    fn candidates_stream_independently_and_final_promotes_quinlight() {
        let mut slot = sample_slot();
        let original = slot.original.clone();
        let lava: Vec<f64> = original
            .iter()
            .enumerate()
            .map(|(i, &s)| s + 0.03 * (i as f64 * 7.3).sin())
            .collect();
        let audio = original.clone();

        let candidate_mode = slot
            .apply_candidate_result(&engine_result("LavaSR", lava.clone()), Vec::new())
            .expect("Candidate should select the LavaSR mode");

        assert!(matches!(candidate_mode, SampleMode::Engine(ref name) if name == "LavaSR"));
        assert!(slot.quinlight_result.is_none());

        slot.apply_candidate_result(&engine_result("AudioSR", audio.clone()), Vec::new());
        let expected_final = expected_quinlight_result(&slot);
        let final_mode = slot
            .apply_final_result(&expected_final, Vec::new())
            .expect("Final should select the Quinlight mode");

        let modes = slot.available_modes();

        assert!(
            matches!(modes.first(), Some(SampleMode::Engine(n)) if n.starts_with(QUINLIGHT_ENGINE_NAME))
        );
        assert!(
            matches!(final_mode, SampleMode::Engine(ref n) if n.starts_with(QUINLIGHT_ENGINE_NAME))
        );
        assert_eq!(
            slot.quinlight_result.as_ref().map(|er| er.data.clone()),
            Some(expected_final.data)
        );
        assert_eq!(
            sample_mode_name_for_slot(&slot, &final_mode),
            expected_final.engine_name
        );
    }

    #[test]
    fn three_candidate_arrivals_preserve_raw_modes_and_accept_final_output() {
        let mut slot = sample_slot();
        let a = slot.original.clone();
        let b: Vec<f64> = a
            .iter()
            .enumerate()
            .map(|(i, &s)| s + 0.01 * (i as f64 * 7.3).sin())
            .collect();
        let c: Vec<f64> = (0..4096)
            .map(|i| (i as f64 * 1500.0 * 2.0 * std::f64::consts::PI / 48000.0).sin())
            .collect();

        slot.apply_candidate_result(&engine_result("AudioSR", a.clone()), Vec::new());
        slot.apply_candidate_result(&engine_result("LavaSR", b.clone()), Vec::new());
        slot.apply_candidate_result(&engine_result("FLowHigh", c.clone()), Vec::new());

        let expected = expected_quinlight_result(&slot);
        let final_mode = slot
            .apply_final_result(&expected, Vec::new())
            .expect("Final should select the Quinlight mode");

        assert_eq!(slot.engine_results.len(), 3);
        assert_eq!(
            slot.engine_results
                .iter()
                .map(|result| result.name.as_str())
                .collect::<Vec<_>>(),
            vec!["LavaSR", "FLowHigh", "AudioSR"]
        );
        assert!(
            slot.available_modes()
                .contains(&SampleMode::Engine("AudioSR".to_string()))
        );
        assert!(
            slot.available_modes()
                .contains(&SampleMode::Engine("LavaSR".to_string()))
        );
        assert!(
            slot.available_modes()
                .contains(&SampleMode::Engine("FLowHigh".to_string()))
        );
        assert_eq!(
            slot.quinlight_result.as_ref().map(|er| er.data.clone()),
            Some(expected.data)
        );
        assert!(
            matches!(final_mode, SampleMode::Engine(ref n) if n.starts_with(QUINLIGHT_ENGINE_NAME))
        );
        assert_eq!(
            sample_mode_name_for_slot(&slot, &final_mode),
            expected.engine_name
        );
    }

    #[test]
    fn later_result_replaces_only_that_engine_output() {
        let mut slot = sample_slot();
        let original = slot.original.clone();
        let lava: Vec<f64> = original
            .iter()
            .enumerate()
            .map(|(i, &s)| s + 0.02 * (i as f64 * 9.1).sin())
            .collect();
        let audio_v1: Vec<f64> = original
            .iter()
            .enumerate()
            .map(|(i, &s)| s + 0.05 * (i as f64 * 3.7).sin())
            .collect();
        let audio_v2 = original.clone();

        slot.apply_candidate_result(&engine_result("AudioSR", audio_v1), Vec::new());
        slot.apply_candidate_result(&engine_result("LavaSR", lava.clone()), Vec::new());
        slot.apply_candidate_result(&engine_result("AudioSR", audio_v2.clone()), Vec::new());
        let expected_final = expected_quinlight_result(&slot);
        let final_mode = slot
            .apply_final_result(&expected_final, Vec::new())
            .expect("Final should select the Quinlight mode");

        assert_eq!(
            slot.engine_result("AudioSR").map(|er| er.data.clone()),
            Some(audio_v2)
        );
        assert_eq!(
            slot.engine_result("LavaSR").map(|er| er.data.clone()),
            Some(lava)
        );
        assert!(
            matches!(final_mode, SampleMode::Engine(ref n) if n.starts_with(QUINLIGHT_ENGINE_NAME))
        );
        assert_eq!(
            slot.quinlight_result.as_ref().map(|er| er.data.clone()),
            Some(expected_final.data)
        );
        assert_eq!(
            sample_mode_name_for_slot(&slot, &final_mode),
            expected_final.engine_name
        );
    }

    #[test]
    fn blind_test_prefers_derived_quinlight_when_present() {
        let mut slot = sample_slot();
        let original = slot.original.clone();
        slot.apply_candidate_result(&engine_result("AudioSR", original.clone()), Vec::new());
        slot.apply_candidate_result(
            &engine_result(
                "LavaSR",
                original
                    .iter()
                    .enumerate()
                    .map(|(i, &s)| s + 0.03 * (i as f64 * 8.1).sin())
                    .collect(),
            ),
            Vec::new(),
        );
        let expected_final = expected_quinlight_result(&slot);
        slot.apply_final_result(&expected_final, Vec::new());

        let expected = slot.quinlight_result.as_ref().map(|er| er.data.clone());

        assert!(
            slot.preferred_blind_test_result()
                .map(|er| er.name.starts_with(QUINLIGHT_ENGINE_NAME))
                .unwrap_or(false)
        );
        assert_eq!(slot.blind_test_data(), expected);
    }

    #[test]
    fn marks_failed_when_all_engines_finish_without_any_audio_result() {
        let mut slot = sample_slot();
        slot.engines_done = 3;
        slot.engines_total = 3;

        slot.refresh_failed_state();

        assert!(slot.failed);

        slot.apply_candidate_result(&engine_result("AudioSR", slot.original.clone()), Vec::new());
        assert!(!slot.failed);
    }

    #[test]
    fn lava_only_above_16khz_initializes_with_zero_eligible_engines() {
        let engine = RemasterEngine::from_test_engines(vec![Box::new(GuiLavaRateLimitedEngine)]);
        let enabled = vec!["LavaSR".to_string()];
        let mut slot = sample_slot();
        slot.original_rate = 22_050;
        slot.engines_total =
            engine.eligible_enabled_engine_count_for_rate(slot.original_rate, &enabled);
        slot.engines_done = 0;

        slot.refresh_failed_state();

        assert_eq!(slot.engines_total, 0);
        assert!(!slot.failed);
    }

    #[test]
    fn streamed_xm_preview_repatches_from_original_effect_snapshot() {
        let (mut module, mut slot, expected_param) = xm_offset_slot_fixture();
        let saved_effects = crate::remaster::save_effect_params(&module, slot.index);
        assert_eq!(saved_effects, vec![(0, 0, 0, 0x10)]);

        let candidate = slot_result(&slot, "AudioSR", slot.original.clone());
        let candidate_mode = slot
            .apply_candidate_result(&candidate, saved_effects)
            .expect("Candidate should activate an engine mode");
        apply_sample_mode_to_slot_t(&mut module, &mut slot, candidate_mode)
            .expect("Candidate mode should apply cleanly");
        assert_eq!(
            module.get_pattern_command(0, 0, 0, COMMAND_PARAMETER),
            expected_param,
            "First streamed preview should scale XM offsets once",
        );

        let final_result = slot_result(&slot, "Quinlight Audio (A)", slot.original.clone());
        let final_mode = slot
            .apply_final_result(&final_result, Vec::new())
            .expect("Final should activate the Quinlight mode");
        apply_sample_mode_to_slot_t(&mut module, &mut slot, final_mode)
            .expect("Final mode should reapply from the original effect snapshot");
        assert_eq!(
            module.get_pattern_command(0, 0, 0, COMMAND_PARAMETER),
            expected_param,
            "Final streamed preview should not compound XM offset scaling",
        );
    }

    #[test]
    fn no_consensus_final_restores_original_sample_after_candidate_preview() {
        let (mut module, mut slot, expected_param) = xm_offset_slot_fixture();
        let saved_effects = crate::remaster::save_effect_params(&module, slot.index);
        let candidate = slot_result(&slot, "AudioSR", slot.original.clone());
        let candidate_mode = slot
            .apply_candidate_result(&candidate, saved_effects)
            .expect("Candidate should activate an engine mode");
        apply_sample_mode_to_slot_t(&mut module, &mut slot, candidate_mode)
            .expect("Candidate mode should apply cleanly");
        assert_eq!(
            module.get_pattern_command(0, 0, 0, COMMAND_PARAMETER),
            expected_param,
            "Candidate preview should scale XM offsets once",
        );

        let final_result = SampleResult {
            index: slot.index,
            data: slot.original.clone(),
            length_frames: slot.original_length_frames,
            channels: slot.original_channels,
            sample_rate_hz: slot.original_rate,
            engine_name: QUINLIGHT_ENGINE_NAME.to_string(),
            discovered_loops: None,
        };
        let final_mode = slot
            .apply_final_result(&final_result, Vec::new())
            .expect("No-consensus final should restore original mode");
        apply_sample_mode_to_slot_t(&mut module, &mut slot, final_mode)
            .expect("Original mode should restore the original sample");

        assert_eq!(slot.mode, SampleMode::Original);
        assert_eq!(sample_mode_button_label(&slot), QUINLIGHT_ORIGINAL_TAG);
        assert_eq!(
            sample_mode_tooltip(&slot),
            "Cycle sample 2 from Quinlight Audio (Original) to AudioSR."
        );
        assert_eq!(module.sample_rate(slot.index), slot.original_rate);
        assert_eq!(
            module.get_pattern_command(0, 0, 0, COMMAND_PARAMETER),
            0x10,
            "Original restore should reinstate the saved XM offset parameter",
        );
    }

    #[test]
    fn reference_48k_scales_offsets_and_caches_sinc_buffer() {
        let (mut module, mut slot, expected_param) = xm_offset_slot_fixture();
        slot.original_effects = crate::remaster::save_effect_params(&module, slot.index);
        assert_eq!(slot.original_effects, vec![(0, 0, 0, 0x10)]);
        assert!(slot.reference_48k.is_none());

        apply_sample_mode_to_slot_t(&mut module, &mut slot, SampleMode::Reference48k)
            .expect("Reference 48k should apply cleanly");

        assert_eq!(slot.mode, SampleMode::Reference48k);
        assert_eq!(module.sample_rate(slot.index), 48_000);
        assert_eq!(
            module.get_pattern_command(0, 0, 0, COMMAND_PARAMETER),
            expected_param,
            "Reference 48k should scale XM offset params like the AI path",
        );
        let cached_ptr = slot
            .reference_48k
            .as_ref()
            .expect("SINC buffer should be cached after first entry")
            .as_ptr();

        apply_sample_mode_to_slot_t(&mut module, &mut slot, SampleMode::Reference48k)
            .expect("Re-entering Reference 48k should succeed");
        assert_eq!(
            slot.reference_48k.as_ref().unwrap().as_ptr(),
            cached_ptr,
            "Second entry should reuse the cached SINC buffer rather than resample again",
        );
        assert_eq!(
            module.get_pattern_command(0, 0, 0, COMMAND_PARAMETER),
            expected_param,
            "Re-entering Reference 48k should not compound offset scaling",
        );
    }

    #[test]
    fn applying_same_engine_mode_twice_is_idempotent() {
        let (mut module, mut slot, expected_param) = xm_offset_slot_fixture();
        let saved_effects = crate::remaster::save_effect_params(&module, slot.index);
        let candidate = slot_result(&slot, "AudioSR", slot.original.clone());
        let mode = slot
            .apply_candidate_result(&candidate, saved_effects)
            .expect("Candidate should activate an engine mode");

        apply_sample_mode_to_slot_t(&mut module, &mut slot, mode.clone())
            .expect("First engine mode apply should succeed");
        let after_first = module.get_pattern_command(0, 0, 0, COMMAND_PARAMETER);

        apply_sample_mode_to_slot_t(&mut module, &mut slot, mode)
            .expect("Second engine mode apply should succeed");
        assert_eq!(after_first, expected_param);
        assert_eq!(
            module.get_pattern_command(0, 0, 0, COMMAND_PARAMETER),
            expected_param,
            "Reapplying the same engine mode should preserve a single forward XM offset scale",
        );
    }

    #[test]
    fn one_usable_engine_final_restores_original_mode() {
        let mut slot = sample_slot();
        let original = slot.original.clone();
        let usable: Vec<f64> = original
            .iter()
            .enumerate()
            .map(|(i, &s)| s + 0.02 * (i as f64 * 6.7).sin())
            .collect();

        slot.apply_candidate_result(&engine_result("AudioSR", usable), Vec::new());
        let no_consensus_final = SampleResult {
            index: slot.index,
            data: slot.original.clone(),
            length_frames: slot.original_length_frames,
            channels: slot.original_channels,
            sample_rate_hz: slot.original_rate,
            engine_name: QUINLIGHT_ENGINE_NAME.to_string(),
            discovered_loops: None,
        };
        let final_mode = slot
            .apply_final_result(&no_consensus_final, Vec::new())
            .expect("Final should restore the original mode");

        assert!(matches!(final_mode, SampleMode::Original));
        assert!(slot.quinlight_result.is_none());
        assert_eq!(sample_mode_button_label(&slot), QUINLIGHT_ORIGINAL_TAG);
        assert_eq!(
            slot.preferred_blind_test_result()
                .map(|er| er.name.as_str()),
            Some("AudioSR"),
            "a no-consensus final should not become the preferred blind-test result",
        );
    }
}
