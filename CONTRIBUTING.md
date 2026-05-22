# Contributing to Quinlight Audio

Thanks for your interest in contributing. Quinlight Audio is the audio app in
Kind Computers' growing `quinlight-*` family.

## Build prerequisites

Quinlight Audio targets `x86_64-unknown-linux-gnu`. The supported public install
path is:

```bash
./install_prerequisites.sh
```

That script installs the system packages (build toolchain, SDL2/GTK/FFmpeg
headers, libopenmpt-dev, libarchive-dev), creates a Python venv at
`~/.local/share/quinlight-audio/venv`, and installs the pinned PyTorch + AI
backend packages used by the optional remastering engines.

Once prerequisites are in place:

```bash
cargo build --release
cargo test --workspace
```

Plan for at least **30 GB free** before installing — the full build artifacts +
venv + AI checkpoints land around 26 GB.

## Bug reports and feature requests

Please file issues at
<https://github.com/Kind-Computers/quinlight-audio/issues>. A good report
includes:

- The exact version (`quinlight-audio --version`)
- The platform (`uname -a`, GPU vendor)
- A minimal reproduction (a tracker module file is helpful when the bug is
  format-specific)
- Logs from `RUST_LOG=info quinlight-audio …` if relevant

## AI backends are external

The optional remastering engines (AudioSR, LavaSR, FLowHigh, AP-BWE) are not
bundled with this repository. They are installed externally by
`install_prerequisites.sh` from their upstream sources, with weights pulled at
install time. See the **Legal / Backend Note** section in `README.md` for the
redistribution posture.

## Pull requests

Before opening a PR:

- `cargo fmt --all` clean
- `cargo clippy --workspace --all-targets` clean (no new warnings)
- `cargo test --workspace` passes
- Document any user-visible change in `CHANGELOG.md` under an `## [Unreleased]`
  section

Small, focused PRs are easier to review than sweeping ones. If you're planning
a large change, please open an issue to discuss the approach first.
