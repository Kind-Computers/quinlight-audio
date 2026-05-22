// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Kind Computers, LLC.

#![allow(clippy::collapsible_if, clippy::unnecessary_cast)]

#[cfg(all(target_os = "linux", target_env = "gnu"))]
mod imp {
    use std::env;
    use std::ffi::CStr;
    use std::fs::File;
    use std::io::{self, Read};
    use std::mem;
    use std::os::fd::{FromRawFd, RawFd};
    use std::process::{Command, Stdio};
    use std::ptr;
    use std::slice;
    use std::sync::atomic::{AtomicBool, AtomicI32, AtomicPtr, AtomicUsize, Ordering};

    use libc::{
        F_GETFD, F_SETFD, FD_CLOEXEC, PF_X, PT_LOAD, SA_ONSTACK, SA_RESETHAND, SA_SIGINFO, SIG_DFL,
        SIGABRT, SIGBUS, SIGFPE, SIGILL, SIGSEGV, STDERR_FILENO, c_int, c_void, fcntl, sigaction,
        sigaltstack, sigemptyset, sighandler_t, stack_t,
    };

    const DIAGNOSTICS_ENV: &str = "QUINLIGHT_AUDIO_NATIVE_DIAGNOSTICS";
    const TEST_SIGNAL_ENV: &str = "QUINLIGHT_AUDIO_TEST_FATAL_SIGNAL";
    const HELPER_ENV: &str = "QUINLIGHT_AUDIO_NATIVE_SYMBOLIZER_HELPER";
    const ALTSTACK_SIZE: usize = libc::SIGSTKSZ as usize + 64 * 1024;
    const MAX_BACKTRACE_FRAMES: usize = 128;
    const MAX_SYMBOL_MAP_PATH_BYTES: usize = 1024;

    static INSTALLED: AtomicBool = AtomicBool::new(false);
    static ALTSTACK_PTR: AtomicPtr<c_void> = AtomicPtr::new(ptr::null_mut());
    static SYMBOLIZER_PIPE_FD: AtomicI32 = AtomicI32::new(-1);
    static SYMBOL_MAP_PTR: AtomicPtr<SymbolMapEntry> = AtomicPtr::new(ptr::null_mut());
    static SYMBOL_MAP_LEN: AtomicUsize = AtomicUsize::new(0);

    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    enum DiagnosticsMode {
        Off,
        Symbols,
        Asan,
        Ubsan,
    }

    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    enum TestSignal {
        Abrt,
        Segv,
    }

    #[derive(Clone, Copy)]
    #[repr(C)]
    struct SymbolMapEntry {
        start: usize,
        end: usize,
        base: usize,
        path_len: usize,
        path: [u8; MAX_SYMBOL_MAP_PATH_BYTES],
    }

    impl SymbolMapEntry {
        fn new(start: usize, end: usize, base: usize, path: &str) -> Option<Self> {
            if start >= end || path.is_empty() {
                return None;
            }
            let path_bytes = path.as_bytes();
            if path_bytes.len() > MAX_SYMBOL_MAP_PATH_BYTES {
                return None;
            }
            let mut entry = Self {
                start,
                end,
                base,
                path_len: path_bytes.len(),
                path: [0; MAX_SYMBOL_MAP_PATH_BYTES],
            };
            entry.path[..path_bytes.len()].copy_from_slice(path_bytes);
            Some(entry)
        }

        fn path_bytes(&self) -> &[u8] {
            &self.path[..self.path_len]
        }
    }

    struct SymbolMapCollector {
        entries: Vec<SymbolMapEntry>,
        current_exe: Option<String>,
    }

    #[derive(Debug, PartialEq, Eq)]
    enum HelperRecord<'a> {
        Signal {
            signal: &'a str,
            fault_addr: &'a str,
        },
        Frame {
            index: usize,
            absolute_addr: &'a str,
            module_path: &'a str,
            relative_addr: &'a str,
        },
        RawFrame {
            index: usize,
            absolute_addr: &'a str,
        },
    }

    fn parse_mode(raw: Option<&str>) -> Result<DiagnosticsMode, String> {
        match raw
            .map(str::trim)
            .unwrap_or("off")
            .to_ascii_lowercase()
            .as_str()
        {
            "" | "off" => Ok(DiagnosticsMode::Off),
            "symbols" => Ok(DiagnosticsMode::Symbols),
            "asan" => Ok(DiagnosticsMode::Asan),
            "ubsan" => Ok(DiagnosticsMode::Ubsan),
            other => Err(format!(
                "invalid {DIAGNOSTICS_ENV} value '{other}'; expected off, symbols, asan, or ubsan"
            )),
        }
    }

    fn parse_test_signal(raw: &str) -> Result<TestSignal, String> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "abrt" => Ok(TestSignal::Abrt),
            "segv" => Ok(TestSignal::Segv),
            other => Err(format!(
                "invalid {TEST_SIGNAL_ENV} value '{other}'; expected abrt or segv"
            )),
        }
    }

    fn diagnostics_mode_from_env() -> DiagnosticsMode {
        match env::var(DIAGNOSTICS_ENV) {
            Ok(value) => match parse_mode(Some(&value)) {
                Ok(mode) => mode,
                Err(err) => {
                    eprintln!("quinlight: {err}");
                    DiagnosticsMode::Off
                }
            },
            Err(_) => DiagnosticsMode::Off,
        }
    }

    pub fn run_symbolizer_helper_from_env() -> bool {
        if env::var_os(HELPER_ENV).is_none() {
            return false;
        }

        if let Err(err) = run_symbolizer_helper() {
            eprintln!("quinlight: native symbolizer helper failed: {err}");
        }

        true
    }

    pub fn install_from_env() {
        if diagnostics_mode_from_env() == DiagnosticsMode::Off {
            return;
        }

        if let Err(err) = start_symbolizer_helper() {
            eprintln!("quinlight: failed to start native symbolizer helper: {err}");
        }
        if let Err(err) = install_symbol_map() {
            eprintln!("quinlight: failed to prepare native symbol map: {err}");
        }
        if let Err(err) = unsafe { install_handler() } {
            eprintln!("quinlight: failed to install native crash diagnostics: {err}");
        }
    }

    pub fn maybe_trigger_test_signal_from_env() {
        if diagnostics_mode_from_env() == DiagnosticsMode::Off {
            return;
        }

        let Ok(value) = env::var(TEST_SIGNAL_ENV) else {
            return;
        };

        match parse_test_signal(&value) {
            Ok(TestSignal::Abrt) => {
                eprintln!("quinlight: triggering SIGABRT for native diagnostics validation");
                unsafe {
                    libc::raise(SIGABRT);
                }
            }
            Ok(TestSignal::Segv) => {
                eprintln!("quinlight: triggering SIGSEGV for native diagnostics validation");
                unsafe {
                    libc::raise(SIGSEGV);
                }
            }
            Err(err) => eprintln!("quinlight: {err}"),
        }
    }

    fn start_symbolizer_helper() -> io::Result<()> {
        if SYMBOLIZER_PIPE_FD.load(Ordering::SeqCst) >= 0 {
            return Ok(());
        }

        let current_exe = env::current_exe()?;
        let mut pipe_fds = [-1; 2];
        if unsafe { libc::pipe(pipe_fds.as_mut_ptr()) } != 0 {
            return Err(io::Error::last_os_error());
        }

        let read_fd = pipe_fds[0];
        let write_fd = pipe_fds[1];
        if let Err(err) = set_cloexec(write_fd) {
            unsafe {
                libc::close(read_fd);
                libc::close(write_fd);
            }
            return Err(err);
        }

        let read_end = unsafe { File::from_raw_fd(read_fd) };
        let mut command = Command::new(current_exe);
        command
            .env(HELPER_ENV, "1")
            .env(DIAGNOSTICS_ENV, "off")
            .env_remove(TEST_SIGNAL_ENV)
            .stdin(Stdio::from(read_end))
            .stdout(Stdio::inherit())
            .stderr(Stdio::inherit());

        if let Err(err) = command.spawn() {
            unsafe {
                libc::close(write_fd);
            }
            return Err(err);
        }

        SYMBOLIZER_PIPE_FD.store(write_fd, Ordering::SeqCst);
        Ok(())
    }

    fn set_cloexec(fd: RawFd) -> io::Result<()> {
        let flags = unsafe { fcntl(fd, F_GETFD) };
        if flags < 0 {
            return Err(io::Error::last_os_error());
        }
        if unsafe { fcntl(fd, F_SETFD, flags | FD_CLOEXEC) } != 0 {
            return Err(io::Error::last_os_error());
        }
        Ok(())
    }

    fn install_symbol_map() -> io::Result<()> {
        if !SYMBOL_MAP_PTR.load(Ordering::SeqCst).is_null() {
            return Ok(());
        }

        let mut collector = SymbolMapCollector {
            entries: Vec::new(),
            current_exe: env::current_exe()
                .ok()
                .map(|path| path.to_string_lossy().into_owned()),
        };

        let result = unsafe {
            libc::dl_iterate_phdr(
                Some(collect_symbol_map),
                (&mut collector as *mut SymbolMapCollector).cast::<c_void>(),
            )
        };
        if result < 0 {
            return Err(io::Error::other("dl_iterate_phdr failed"));
        }

        if collector.entries.is_empty() {
            return Ok(());
        }

        let boxed = collector.entries.into_boxed_slice();
        let len = boxed.len();
        let ptr = Box::into_raw(boxed) as *mut SymbolMapEntry;
        SYMBOL_MAP_LEN.store(len, Ordering::SeqCst);
        SYMBOL_MAP_PTR.store(ptr, Ordering::SeqCst);
        Ok(())
    }

    unsafe extern "C" fn collect_symbol_map(
        info: *mut libc::dl_phdr_info,
        _size: usize,
        data: *mut c_void,
    ) -> c_int {
        if info.is_null() || data.is_null() {
            return 0;
        }

        let collector = unsafe { &mut *data.cast::<SymbolMapCollector>() };
        let info = unsafe { &*info };
        let Some(module_path) = module_path_from_phdr_info(info, collector.current_exe.as_deref())
        else {
            return 0;
        };

        for idx in 0..info.dlpi_phnum as usize {
            let phdr = unsafe { &*info.dlpi_phdr.add(idx) };
            if phdr.p_type != PT_LOAD || phdr.p_memsz == 0 || (phdr.p_flags & PF_X) == 0 {
                continue;
            }

            let start = info.dlpi_addr as usize + phdr.p_vaddr as usize;
            let end = start.saturating_add(phdr.p_memsz as usize);
            if let Some(entry) =
                SymbolMapEntry::new(start, end, info.dlpi_addr as usize, module_path)
            {
                collector.entries.push(entry);
            }
        }

        0
    }

    fn module_path_from_phdr_info<'a>(
        info: &'a libc::dl_phdr_info,
        current_exe: Option<&'a str>,
    ) -> Option<&'a str> {
        if !info.dlpi_name.is_null() {
            let raw_name = unsafe { CStr::from_ptr(info.dlpi_name) }.to_bytes();
            if !raw_name.is_empty() {
                return std::str::from_utf8(raw_name).ok();
            }
        }
        current_exe
    }

    unsafe fn install_handler() -> io::Result<()> {
        if INSTALLED.swap(true, Ordering::SeqCst) {
            return Ok(());
        }

        let altstack_mem = unsafe { libc::malloc(ALTSTACK_SIZE) };
        if altstack_mem.is_null() {
            INSTALLED.store(false, Ordering::SeqCst);
            return Err(io::Error::other(
                "malloc returned null while allocating crash-handler altstack",
            ));
        }
        ALTSTACK_PTR.store(altstack_mem, Ordering::SeqCst);

        let altstack = stack_t {
            ss_sp: altstack_mem,
            ss_flags: 0,
            ss_size: ALTSTACK_SIZE,
        };
        if unsafe { sigaltstack(&altstack, ptr::null_mut()) } != 0 {
            INSTALLED.store(false, Ordering::SeqCst);
            return Err(io::Error::last_os_error());
        }

        let mut action: sigaction = unsafe { mem::zeroed() };
        unsafe {
            sigemptyset(&mut action.sa_mask);
        }
        action.sa_flags = SA_SIGINFO | SA_ONSTACK | SA_RESETHAND;
        action.sa_sigaction = signal_handler
            as unsafe extern "C" fn(c_int, *mut libc::siginfo_t, *mut c_void)
            as sighandler_t;

        for signal in [SIGSEGV, SIGBUS, SIGILL, SIGABRT, SIGFPE] {
            if unsafe { sigaction(signal, &action, ptr::null_mut()) } != 0 {
                INSTALLED.store(false, Ordering::SeqCst);
                return Err(io::Error::last_os_error());
            }
        }
        Ok(())
    }

    unsafe extern "C" fn signal_handler(
        signum: c_int,
        info: *mut libc::siginfo_t,
        _ucontext: *mut c_void,
    ) {
        write_bytes(b"\nquinlight: native crash diagnostics caught ");
        write_signal_name(signum);
        write_bytes(b" (signal ");
        write_signed_decimal(signum as isize);
        write_bytes(b", fault_addr=");
        let fault_addr = if info.is_null() {
            0
        } else {
            unsafe { (*info).si_addr().addr() }
        };
        if fault_addr == 0 {
            write_bytes(b"unavailable");
        } else {
            write_bytes(b"0x");
            write_hex(fault_addr);
        }
        write_bytes(b")\n");

        let mut frames = [ptr::null_mut::<c_void>(); MAX_BACKTRACE_FRAMES];
        let frame_count = unsafe { libc::backtrace(frames.as_mut_ptr(), frames.len() as c_int) };
        if frame_count > 0 {
            write_bytes(b"quinlight: native backtrace follows\n");
            unsafe {
                libc::backtrace_symbols_fd(frames.as_ptr(), frame_count, STDERR_FILENO);
            }
            write_symbolizer_payload(signum, fault_addr, &frames[..frame_count as usize]);
        } else {
            write_bytes(b"quinlight: native backtrace unavailable\n");
        }

        reset_signal(signum);
        unsafe {
            libc::raise(signum);
            libc::_exit(128 + signum);
        }
    }

    fn write_symbolizer_payload(signum: c_int, fault_addr: usize, frames: &[*mut c_void]) {
        let fd = SYMBOLIZER_PIPE_FD.swap(-1, Ordering::SeqCst);
        if fd < 0 {
            return;
        }

        write_bytes_fd(fd, b"signal\t");
        write_signal_name_fd(fd, signum);
        write_bytes_fd(fd, b"\t");
        if fault_addr == 0 {
            write_bytes_fd(fd, b"unavailable");
        } else {
            write_bytes_fd(fd, b"0x");
            write_hex_fd(fd, fault_addr);
        }
        write_bytes_fd(fd, b"\n");

        for (idx, frame) in frames.iter().enumerate() {
            let address = *frame as usize;
            if let Some(entry) = find_symbol_map_entry(address) {
                write_bytes_fd(fd, b"frame\t");
                write_unsigned_decimal_fd(fd, idx);
                write_bytes_fd(fd, b"\t0x");
                write_hex_fd(fd, address);
                write_bytes_fd(fd, b"\t");
                write_bytes_fd(fd, entry.path_bytes());
                write_bytes_fd(fd, b"\t0x");
                write_hex_fd(fd, address.saturating_sub(entry.base));
                write_bytes_fd(fd, b"\n");
            } else {
                write_bytes_fd(fd, b"raw\t");
                write_unsigned_decimal_fd(fd, idx);
                write_bytes_fd(fd, b"\t0x");
                write_hex_fd(fd, address);
                write_bytes_fd(fd, b"\n");
            }
        }

        unsafe {
            libc::close(fd);
        }
    }

    fn find_symbol_map_entry(address: usize) -> Option<&'static SymbolMapEntry> {
        let ptr = SYMBOL_MAP_PTR.load(Ordering::SeqCst);
        let len = SYMBOL_MAP_LEN.load(Ordering::SeqCst);
        if ptr.is_null() || len == 0 {
            return None;
        }

        let entries = unsafe { slice::from_raw_parts(ptr, len) };
        entries
            .iter()
            .find(|entry| address >= entry.start && address < entry.end)
    }

    fn run_symbolizer_helper() -> io::Result<()> {
        let mut input = String::new();
        io::stdin().read_to_string(&mut input)?;
        if input.trim().is_empty() {
            return Ok(());
        }

        let mut signal_name = None;
        let mut fault_addr = None;
        let mut saw_frames = false;

        for line in input.lines() {
            match parse_helper_record(line) {
                Some(HelperRecord::Signal {
                    signal,
                    fault_addr: addr,
                }) => {
                    signal_name = Some(signal.to_string());
                    fault_addr = Some(addr.to_string());
                }
                Some(HelperRecord::Frame {
                    index,
                    absolute_addr,
                    module_path,
                    relative_addr,
                }) => {
                    if !saw_frames {
                        eprintln!("quinlight: symbolized native backtrace follows");
                        saw_frames = true;
                    }
                    eprintln!("#{index:02} {absolute_addr} {module_path}");
                    let symbolized = symbolize_frame(module_path, relative_addr)
                        .unwrap_or_else(|err| format!("<symbolization failed: {err}>"));
                    for line in symbolized.lines() {
                        eprintln!("    {line}");
                    }
                }
                Some(HelperRecord::RawFrame {
                    index,
                    absolute_addr,
                }) => {
                    if !saw_frames {
                        eprintln!("quinlight: symbolized native backtrace follows");
                        saw_frames = true;
                    }
                    eprintln!("#{index:02} {absolute_addr} <no module mapping>");
                }
                None => {}
            }
        }

        if saw_frames {
            if let Some(signal_name) = signal_name {
                if let Some(fault_addr) = fault_addr {
                    eprintln!(
                        "quinlight: symbolized backtrace context: signal={signal_name}, fault_addr={fault_addr}"
                    );
                }
            }
        }

        Ok(())
    }

    fn parse_helper_record(line: &str) -> Option<HelperRecord<'_>> {
        let mut parts = line.split('\t');
        match parts.next()? {
            "signal" => Some(HelperRecord::Signal {
                signal: parts.next()?,
                fault_addr: parts.next()?,
            }),
            "frame" => Some(HelperRecord::Frame {
                index: parts.next()?.parse().ok()?,
                absolute_addr: parts.next()?,
                module_path: parts.next()?,
                relative_addr: parts.next()?,
            }),
            "raw" => Some(HelperRecord::RawFrame {
                index: parts.next()?.parse().ok()?,
                absolute_addr: parts.next()?,
            }),
            _ => None,
        }
    }

    fn symbolize_frame(module_path: &str, relative_addr: &str) -> io::Result<String> {
        if let Ok(output) = Command::new("llvm-symbolizer")
            .arg("--obj")
            .arg(module_path)
            .arg("--relative-address")
            .arg("--functions")
            .arg("--demangle")
            .arg("--inlines")
            .arg("--pretty-print")
            .arg(relative_addr)
            .output()
        {
            if output.status.success() {
                let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
                if !stdout.is_empty() {
                    return Ok(stdout);
                }
            }
        }

        let output = Command::new("addr2line")
            .arg("-Cpfie")
            .arg(module_path)
            .arg(relative_addr)
            .output()?;
        if output.status.success() {
            let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
            if !stdout.is_empty() {
                return Ok(stdout);
            }
        }

        Err(io::Error::other(
            "llvm-symbolizer and addr2line did not produce any output",
        ))
    }

    fn write_signal_name(signum: c_int) {
        write_signal_name_fd(STDERR_FILENO, signum);
    }

    fn write_signal_name_fd(fd: RawFd, signum: c_int) {
        match signum {
            SIGSEGV => write_bytes_fd(fd, b"SIGSEGV"),
            SIGBUS => write_bytes_fd(fd, b"SIGBUS"),
            SIGILL => write_bytes_fd(fd, b"SIGILL"),
            SIGABRT => write_bytes_fd(fd, b"SIGABRT"),
            SIGFPE => write_bytes_fd(fd, b"SIGFPE"),
            _ => write_bytes_fd(fd, b"signal"),
        }
    }

    fn write_signed_decimal(value: isize) {
        write_signed_decimal_fd(STDERR_FILENO, value);
    }

    fn write_signed_decimal_fd(fd: RawFd, value: isize) {
        if value < 0 {
            write_bytes_fd(fd, b"-");
            write_unsigned_decimal_fd(fd, value.unsigned_abs());
        } else {
            write_unsigned_decimal_fd(fd, value as usize);
        }
    }

    fn write_unsigned_decimal_fd(fd: RawFd, mut value: usize) {
        let mut buf = [0u8; 32];
        let mut idx = buf.len();
        loop {
            idx -= 1;
            buf[idx] = b'0' + (value % 10) as u8;
            value /= 10;
            if value == 0 {
                break;
            }
        }
        write_bytes_fd(fd, &buf[idx..]);
    }

    fn write_hex(value: usize) {
        write_hex_fd(STDERR_FILENO, value);
    }

    fn write_hex_fd(fd: RawFd, mut value: usize) {
        if value == 0 {
            write_bytes_fd(fd, b"0");
            return;
        }

        let mut buf = [0u8; 2 * mem::size_of::<usize>()];
        let mut idx = buf.len();
        while value != 0 {
            let digit = (value & 0xF) as u8;
            idx -= 1;
            buf[idx] = match digit {
                0..=9 => b'0' + digit,
                _ => b'a' + (digit - 10),
            };
            value >>= 4;
        }
        write_bytes_fd(fd, &buf[idx..]);
    }

    fn write_bytes(bytes: &[u8]) {
        write_bytes_fd(STDERR_FILENO, bytes);
    }

    fn write_bytes_fd(fd: RawFd, mut bytes: &[u8]) {
        while !bytes.is_empty() {
            let written = unsafe { libc::write(fd, bytes.as_ptr().cast::<c_void>(), bytes.len()) };
            if written <= 0 {
                break;
            }
            bytes = &bytes[written as usize..];
        }
    }

    fn reset_signal(signum: c_int) {
        unsafe {
            let mut action: sigaction = mem::zeroed();
            sigemptyset(&mut action.sa_mask);
            action.sa_sigaction = SIG_DFL;
            sigaction(signum, &action, ptr::null_mut());
        }
    }

    #[cfg(test)]
    mod tests {
        use super::{
            DiagnosticsMode, HelperRecord, TestSignal, parse_helper_record, parse_mode,
            parse_test_signal,
        };

        #[test]
        fn parse_mode_defaults_to_off() {
            assert_eq!(parse_mode(None).unwrap(), DiagnosticsMode::Off);
        }

        #[test]
        fn parse_mode_recognizes_supported_values() {
            assert_eq!(
                parse_mode(Some("symbols")).unwrap(),
                DiagnosticsMode::Symbols
            );
            assert_eq!(parse_mode(Some("asan")).unwrap(), DiagnosticsMode::Asan);
            assert_eq!(parse_mode(Some("ubsan")).unwrap(), DiagnosticsMode::Ubsan);
        }

        #[test]
        fn parse_test_signal_accepts_known_values() {
            assert_eq!(parse_test_signal("abrt").unwrap(), TestSignal::Abrt);
            assert_eq!(parse_test_signal("segv").unwrap(), TestSignal::Segv);
        }

        #[test]
        fn parse_mode_rejects_unknown_values() {
            assert!(parse_mode(Some("nope")).is_err());
        }

        #[test]
        fn parse_test_signal_rejects_unknown_values() {
            assert!(parse_test_signal("boom").is_err());
        }

        #[test]
        fn parse_helper_signal_record() {
            assert_eq!(
                parse_helper_record("signal\tSIGSEGV\t0x1234"),
                Some(HelperRecord::Signal {
                    signal: "SIGSEGV",
                    fault_addr: "0x1234",
                })
            );
        }

        #[test]
        fn parse_helper_frame_record() {
            assert_eq!(
                parse_helper_record("frame\t3\t0x7f00\t/tmp/quinlight\t0xabc"),
                Some(HelperRecord::Frame {
                    index: 3,
                    absolute_addr: "0x7f00",
                    module_path: "/tmp/quinlight",
                    relative_addr: "0xabc",
                })
            );
        }

        #[test]
        fn parse_helper_raw_record() {
            assert_eq!(
                parse_helper_record("raw\t4\t0x7f01"),
                Some(HelperRecord::RawFrame {
                    index: 4,
                    absolute_addr: "0x7f01",
                })
            );
        }
    }
}

#[cfg(not(all(target_os = "linux", target_env = "gnu")))]
mod imp {
    pub fn run_symbolizer_helper_from_env() -> bool {
        false
    }

    pub fn install_from_env() {}

    pub fn maybe_trigger_test_signal_from_env() {}
}

pub use imp::{
    install_from_env, maybe_trigger_test_signal_from_env, run_symbolizer_helper_from_env,
};
