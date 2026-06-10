// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Kind Computers, LLC.

use std::fs::File;
use std::io::{self, Read, Write};
use std::path::Path;

/// Known tracker module file extensions.
const MODULE_EXTENSIONS: &[&str] = &[
    "mod", "s3m", "xm", "it", "mptm", "stm", "nst", "m15", "stk", "wow", "ult", "669", "mtm",
    "med", "far", "mdl", "ams", "dsm", "amf", "okt", "dmf", "ptm", "psm", "mt2", "dbm", "digi",
    "imf", "j2b", "gdm", "umx", "plm", "mo3", "xpk", "ppm", "mmcmp",
];

/// Archive file extensions we recognize.
const ARCHIVE_EXTENSIONS: &[&str] = &[
    "zip", "7z", "rar", "tar", "gz", "tgz", "bz2", "tbz2", "xz", "txz", "zst", "lha", "lzh", "cab",
    "iso",
];

pub const MAX_MODULE_BYTES: u64 = 128 * 1024 * 1024;
pub const MAX_ARCHIVE_MODULE_ENTRIES: usize = 2048;

/// A module file found inside an archive.
#[derive(Debug, Clone)]
pub struct ArchiveEntry {
    /// Full path within the archive (e.g. "subdir/track.s3m").
    pub path: String,
    /// Just the filename (e.g. "track.s3m").
    pub filename: String,
    /// Uncompressed size in bytes (0 if unknown).
    #[allow(dead_code)]
    pub size: u64,
}

#[derive(Debug)]
struct BoundedVecWriter {
    buf: Vec<u8>,
    limit: usize,
    hit_limit: bool,
}

impl BoundedVecWriter {
    fn new(limit: usize) -> Self {
        Self {
            buf: Vec::new(),
            limit,
            hit_limit: false,
        }
    }

    fn into_inner(self) -> Vec<u8> {
        self.buf
    }
}

impl Write for BoundedVecWriter {
    fn write(&mut self, data: &[u8]) -> io::Result<usize> {
        let remaining = self.limit.saturating_sub(self.buf.len());
        if data.len() > remaining {
            self.hit_limit = true;
            return Err(io::Error::other("archive entry exceeds extraction limit"));
        }
        self.buf.extend_from_slice(data);
        Ok(data.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

/// Check if a path looks like an archive by its extension.
pub fn is_archive(path: &Path) -> bool {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    ARCHIVE_EXTENSIONS.contains(&ext.as_str())
        || path.to_str().is_some_and(|s| {
            s.ends_with(".tar.gz")
                || s.ends_with(".tar.bz2")
                || s.ends_with(".tar.xz")
                || s.ends_with(".tar.zst")
        })
}

pub fn is_module_file(name: &str) -> bool {
    let lower = name.to_ascii_lowercase();
    if let Some(dot) = lower.rfind('.') {
        MODULE_EXTENSIONS.contains(&&lower[dot + 1..])
    } else {
        false
    }
}

fn size_limit_message(context: &str, size: u64, limit: u64) -> String {
    format!("Refusing to load {context}: {size} bytes exceeds the {limit} byte safety limit",)
}

fn read_to_vec_with_limit<R: Read>(
    mut reader: R,
    limit: u64,
    context: &str,
) -> Result<Vec<u8>, String> {
    let limit_usize =
        usize::try_from(limit).map_err(|_| "Archive size limit overflowed".to_string())?;
    let mut buf = Vec::new();
    let mut chunk = [0u8; 8192];

    loop {
        let read = reader
            .read(&mut chunk)
            .map_err(|e| format!("Failed to read {context}: {e}"))?;
        if read == 0 {
            break;
        }
        if buf.len().saturating_add(read) > limit_usize {
            return Err(size_limit_message(
                context,
                buf.len() as u64 + read as u64,
                limit,
            ));
        }
        buf.extend_from_slice(&chunk[..read]);
    }

    Ok(buf)
}

fn collect_module_entries<I>(entries: I) -> Result<Vec<ArchiveEntry>, String>
where
    I: IntoIterator<Item = String>,
{
    let mut modules = Vec::new();
    for path in entries {
        if !is_module_file(&path) {
            continue;
        }
        if modules.len() >= MAX_ARCHIVE_MODULE_ENTRIES {
            return Err(format!(
                "Archive contains more than {MAX_ARCHIVE_MODULE_ENTRIES} candidate module files; refusing to browse it"
            ));
        }
        let filename = path.rsplit('/').next().unwrap_or(&path).to_string();
        modules.push(ArchiveEntry {
            path,
            filename,
            size: 0,
        });
    }

    modules.sort_by(|a, b| {
        a.filename
            .to_ascii_lowercase()
            .cmp(&b.filename.to_ascii_lowercase())
    });
    Ok(modules)
}

pub fn read_module_file(path: &Path) -> Result<Vec<u8>, String> {
    let metadata = std::fs::metadata(path)
        .map_err(|e| format!("Failed to inspect {}: {e}", path.display()))?;
    if metadata.len() > MAX_MODULE_BYTES {
        return Err(size_limit_message(
            &format!("file '{}'", path.display()),
            metadata.len(),
            MAX_MODULE_BYTES,
        ));
    }

    let file = File::open(path).map_err(|e| format!("Failed to open {}: {e}", path.display()))?;
    read_to_vec_with_limit(
        file,
        MAX_MODULE_BYTES,
        &format!("file '{}'", path.display()),
    )
}

/// List all tracker module files inside an archive, sorted by name.
///
/// Known limitation: `compress_tools::list_archive_files` materializes the
/// full entry-name list before our candidate cap applies, so a hostile
/// archive with millions of entries inflates memory during the listing step
/// (names only — extraction itself is bounded by `MAX_MODULE_BYTES` and the
/// candidate cap). compress-tools exposes no streaming listing API to bound
/// this without dropping to raw libarchive.
pub fn list_modules_in_archive(path: &Path) -> Result<Vec<ArchiveEntry>, String> {
    let file = File::open(path).map_err(|e| format!("Failed to open archive: {e}"))?;
    let entries = compress_tools::list_archive_files(file)
        .map_err(|e| format!("Failed to read archive: {e}"))?;
    collect_module_entries(entries)
}

/// Extract a single file from an archive into memory.
pub fn extract_from_archive(archive_path: &Path, entry_path: &str) -> Result<Vec<u8>, String> {
    let file = File::open(archive_path).map_err(|e| format!("Failed to open archive: {e}"))?;
    let mut buf = BoundedVecWriter::new(MAX_MODULE_BYTES as usize);
    let extracted = compress_tools::uncompress_archive_file(file, &mut buf, entry_path);
    if buf.hit_limit {
        return Err(size_limit_message(
            &format!("archive entry '{entry_path}'"),
            MAX_MODULE_BYTES + 1,
            MAX_MODULE_BYTES,
        ));
    }
    extracted.map_err(|e| format!("Failed to extract '{entry_path}': {e}"))?;
    Ok(buf.into_inner())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn oversized_direct_module_is_rejected() {
        let temp = tempfile::NamedTempFile::new().expect("tempfile");
        std::fs::write(temp.path(), b"tiny").expect("seed file");
        let oversized = MAX_MODULE_BYTES + 1;
        temp.as_file().set_len(oversized).expect("resize file");

        let err = read_module_file(temp.path()).expect_err("oversized file should fail");
        assert!(err.contains("safety limit"));
    }

    #[test]
    fn too_many_archive_candidates_are_rejected() {
        let names = (0..=MAX_ARCHIVE_MODULE_ENTRIES)
            .map(|idx| format!("mods/track-{idx}.xm"))
            .collect::<Vec<_>>();

        let err = collect_module_entries(names).expect_err("archive should be rejected");
        assert!(err.contains("candidate module files"));
        assert!(err.contains("2048"));
    }

    #[test]
    fn bounded_archive_writer_rejects_oversized_entry() {
        let mut writer = BoundedVecWriter::new(4);
        writer.write_all(b"1234").expect("fits");
        let err = writer
            .write_all(b"5")
            .expect_err("fifth byte should exceed limit");

        assert!(writer.hit_limit);
        assert_eq!(err.kind(), io::ErrorKind::Other);
    }

    #[test]
    fn malformed_archive_reports_extract_error() {
        let temp = tempfile::NamedTempFile::new().expect("tempfile");
        std::fs::write(temp.path(), b"not an archive").expect("junk archive");

        let err = extract_from_archive(temp.path(), "song.mod").expect_err("junk should fail");
        assert!(err.starts_with("Failed to extract 'song.mod':"));
    }

    #[test]
    fn module_reader_enforces_stream_limit_even_without_metadata() {
        let payload = [1u8; 16];
        let err = read_to_vec_with_limit(&payload[..], 8, "test payload")
            .expect_err("stream should exceed limit");
        assert!(err.contains("test payload"));
        assert!(err.contains("8 byte safety limit"));
    }
}
