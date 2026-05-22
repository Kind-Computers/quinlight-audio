#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Kind Computers, LLC.
set -e

VENV="$HOME/.local/share/quinlight-audio/venv"

echo "Quinlight Audio AI engine installer"

# Pick the highest installed python3.X (>=3.12) whose matching -venv package is
# available from apt. Some distros (e.g. Ubuntu 26.04) ship a python3.13 binary
# but only package python3.14-venv, so we can't just take the highest binary.
PY=""
for CAND in python3.14 python3.13 python3.12 python3; do
  command -v "$CAND" &>/dev/null || continue
  CAND_MINOR=$("$CAND" -c 'import sys; print(sys.version_info.minor)')
  [ "$CAND_MINOR" -lt 12 ] && continue
  CAND_PKG="python3.${CAND_MINOR}-venv"
  if apt-cache show "$CAND_PKG" &>/dev/null; then
    PY="$CAND"
    VENV_PKG="$CAND_PKG"
    break
  fi
done
if [ -z "$PY" ]; then
  echo "Error: no python3 (>=3.12) found with an installable -venv package." >&2
  exit 1
fi
echo "Using $($PY --version), venv package: $VENV_PKG, torch 2.11.x"

sudo apt install -y pipewire-alsa "$VENV_PKG" \
  build-essential clang mold \
  libopenmpt-dev libarchive-dev libsdl2-dev libgtk-3-dev \
  libpango1.0-dev libglib2.0-dev libgdk-pixbuf-2.0-dev libatk1.0-dev libcairo2-dev \
  libavcodec-dev libavformat-dev libavutil-dev libswresample-dev libswscale-dev

# If the existing venv's bin/python is a different minor version than $PY, the
# venv module won't replace the binary — pip ends up installing under a new
# lib/python3.X/site-packages while bin/python still points at the old one, so
# nothing imports. Recreate from scratch in that case.
if [ -d "$VENV" ]; then
  EXISTING_MINOR=$("$VENV/bin/python" -c 'import sys; print(sys.version_info.minor)' 2>/dev/null || true)
  if [ "$EXISTING_MINOR" != "$CAND_MINOR" ]; then
    echo "Existing venv uses python3.${EXISTING_MINOR:-?}; recreating with $PY"
    rm -rf "$VENV"
  fi
fi
$PY -m venv "$VENV"
"$VENV/bin/pip" install --upgrade pip setuptools==70.2.0

# Detect GPU for torch index (mirrors src/engine/mod.rs detect_gpu())
if nvidia-smi &>/dev/null; then
  echo "Detected NVIDIA GPU, installing default torch (CUDA)"
  "$VENV/bin/pip" install torch==2.11.0 torchaudio==2.11.0 torchvision==0.26.0
elif [ -d /opt/rocm ] || { command -v rocminfo &>/dev/null && rocminfo 2>/dev/null | grep '^ *Name: *gfx' >/dev/null; }; then
  # Ubuntu's distro ROCm packages (libamdhip64, rocminfo) install to /usr, not /opt/rocm.
  echo "Detected AMD GPU (ROCm), installing ROCm 7.2 torch"
  "$VENV/bin/pip" install torch==2.11.0 torchaudio==2.11.0 torchvision==0.26.0 \
    --index-url https://download.pytorch.org/whl/rocm7.2
elif xpu-smi discovery &>/dev/null || grep -qs 0x8086 /sys/class/drm/card*/device/vendor; then
  echo "Detected Intel GPU (XPU), installing XPU torch"
  DRIVER=$(basename "$(readlink /sys/class/drm/card0/device/driver 2>/dev/null)" 2>/dev/null)
  [ "$DRIVER" = "xe" ] && echo "  Xe kernel driver detected (required for Battlemage)"
  "$VENV/bin/pip" install torch==2.11.0 torchaudio==2.11.0 torchvision==0.26.0 \
    --index-url https://download.pytorch.org/whl/xpu
  if ! ldconfig -p 2>/dev/null | grep -q libze_loader; then
    echo "  Warning: Level Zero runtime (libze_loader) not found — install intel-level-zero-gpu"
  fi
else
  echo "No GPU detected, installing CPU-only torch"
  "$VENV/bin/pip" install torch==2.11.0+cpu torchaudio==2.11.0+cpu torchvision==0.26.0+cpu \
    --index-url https://download.pytorch.org/whl/cpu
fi
"$VENV/bin/pip" install audiosr==0.0.7 --no-deps
"$VENV/bin/pip" install \
  numpy==2.4.3 \
  librosa==0.11.0 \
  soundfile==0.13.1 \
  scipy==1.17.1 \
  transformers==5.3.0 \
  einops==0.8.2 \
  PyYAML==6.0.3 \
  tqdm==4.67.3 \
  chardet==7.3.0 \
  huggingface_hub==1.8.0 \
  torchlibrosa==0.1.0 \
  timm==1.0.26 \
  progressbar2==4.5.0 \
  ftfy==6.3.1 \
  Unidecode==1.4.0 \
  phonemizer==3.3.0 \
  pandas==3.0.1 \
  matplotlib==3.10.8 \
  torchcodec==0.11.0
"$VENV/bin/pip" install \
  "vocos @ git+https://github.com/langtech-bsc/vocos.git@451e522f7a11c3652c9522f63ea6780736d93de0"
"$VENV/bin/pip" install \
  "LavaSR @ git+https://github.com/ysharma3501/LavaSR.git@2bad8f7e505e7cd590b0e21bf24c6a0b362bdfac"
"$VENV/bin/pip" install \
  "flowhigh @ git+https://github.com/resemble-ai/flowhigh.git@8b5abcf7f0bc82aaef0e8936f0f5b9ab61990dd2"

# AP-BWE — speech bandwidth extension (GAN, parallel amplitude/phase prediction).
# Upstream repo has no setup.py, so clone it and download weights from Google Drive.
APBWE_DIR="$HOME/.local/share/quinlight-audio/apbwe"
APBWE_COMMIT="751710f22404c27e5bcc983248f8b856a04b8422"
APBWE_FOLDER_ID="1IIYTf2zbJWzelu4IftKD6ooHloJ8mnZF"
APBWE_WEIGHTS_URL="https://drive.google.com/drive/folders/${APBWE_FOLDER_ID}"
if [ ! -d "$APBWE_DIR/.git" ]; then
  echo "Cloning AP-BWE into $APBWE_DIR"
  git clone https://github.com/yxlu-0102/AP-BWE.git "$APBWE_DIR"
fi
git -C "$APBWE_DIR" fetch --quiet
git -C "$APBWE_DIR" checkout --quiet "$APBWE_COMMIT"
mkdir -p "$APBWE_DIR/checkpoints"
if ! ls "$APBWE_DIR/checkpoints"/g_*to48k >/dev/null 2>&1; then
  # Prefer an OAuth-authenticated rclone remote (bypasses Drive's anonymous
  # quota). Fall back to gdown for users who haven't configured rclone.
  RCLONE_DRIVE_REMOTE=""
  if command -v rclone >/dev/null 2>&1; then
    RCLONE_DRIVE_REMOTE=$(rclone listremotes --long 2>/dev/null \
      | awk '$2 == "drive" { sub(/:$/, "", $1); print $1; exit }')
  fi
  if [ -n "$RCLONE_DRIVE_REMOTE" ]; then
    echo "Downloading AP-BWE checkpoints via rclone remote '$RCLONE_DRIVE_REMOTE'"
    rclone copy --drive-root-folder-id="$APBWE_FOLDER_ID" \
      "${RCLONE_DRIVE_REMOTE}:" "$APBWE_DIR/checkpoints"
  else
    echo "Downloading AP-BWE checkpoints via gdown (anonymous — subject to Google throttling)"
    echo "  Tip: run 'rclone config' to set up an OAuth Drive remote for reliable downloads."
    "$VENV/bin/pip" install --quiet gdown
    (cd "$APBWE_DIR/checkpoints" && "$VENV/bin/gdown" --folder "$APBWE_WEIGHTS_URL")
  fi
  # Both tools may drop files into a named subdirectory; flatten.
  find "$APBWE_DIR/checkpoints" -mindepth 2 -type f -name 'g_*to48k' \
    -exec mv -n {} "$APBWE_DIR/checkpoints/" \;
  ls "$APBWE_DIR/checkpoints"/g_*to48k >/dev/null 2>&1 || { \
    echo "AP-BWE: no checkpoints downloaded. If using gdown, Google Drive may be throttling —" >&2; \
    echo "  retry, configure rclone ('rclone config'), or download manually from:" >&2; \
    echo "  $APBWE_WEIGHTS_URL" >&2; exit 1; }
else
  echo "AP-BWE checkpoints already present in $APBWE_DIR/checkpoints"
fi

"$VENV/bin/python" -c "import importlib.util, os.path; ok=[m for m in ('audiosr','LavaSR','flowhigh') if importlib.util.find_spec(m)]; ok += ['apbwe'] if os.path.isfile(os.path.expanduser('~/.local/share/quinlight-audio/apbwe/models/model.py')) else []; print('Quinlight Audio smoke check:', ', '.join(ok) if ok else 'no engines detected')"
