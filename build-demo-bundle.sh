#!/usr/bin/env bash
#
# build-demo-bundle.sh — regenerate the audio demos for the README "Listen"
# table from the freely-licensed modules in mods/.
#
# For every module in mods/ it produces two 96 kHz AAC (.m4a) files:
#   * before  — deterministic render, original samples, NO AI  (--no-remaster)
#   * after   — Quinlight AI remaster (multi-engine registration consensus)
# then assembles quinlight-audio-96khz-bundle.zip laid out as the GitHub Pages
# workflow expects:  quinlight-audio-96khz/{rendered,remastered}/
#
# The "after" pass runs the AI engines, so it needs the engine venv + GPU (same
# prerequisites as install_prerequisites.sh). By default the binary is built
# fresh from this source tree so the demos use the CURRENT algorithm.
#
# Engine selection / low-VRAM note:
#   The consensus needs >= 2 engines that produce output. By default all detected
#   engines run. Override with QUINLIGHT_ENGINES (space/comma separated), e.g.
#       QUINLIGHT_ENGINES="flowhigh audiosr" ./build-demo-bundle.sh
#   On a small-VRAM GPU (~6 GB) two diffusion models can't be resident at once,
#   so the single combined pass OOMs (and falls back to slow CPU). Set STAGED=1
#   to instead warm the per-engine sample cache one engine at a time (each fits
#   the card), then build the consensus from cache:
#       STAGED=1 QUINLIGHT_ENGINES="flowhigh audiosr" ./build-demo-bundle.sh
#
# Usage:   ./build-demo-bundle.sh
#          QUINLIGHT_BIN=/path/to/quinlight-audio ./build-demo-bundle.sh

set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODS="$REPO/mods"
OUT="$REPO/demo-build"
RATE=96000
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# --- resolve the binary (fresh build by default, so demos use current code) ---
if [ -n "${QUINLIGHT_BIN:-}" ]; then
  QUINLIGHT=("$QUINLIGHT_BIN")
elif command -v cargo >/dev/null 2>&1 && [ -f "$REPO/Cargo.toml" ]; then
  echo "[build] cargo build --release (fresh binary)"
  cargo build --release --manifest-path "$REPO/Cargo.toml" || exit 1
  QUINLIGHT=(cargo run --release --quiet --manifest-path "$REPO/Cargo.toml" --)
elif command -v quinlight-audio >/dev/null 2>&1; then
  QUINLIGHT=(quinlight-audio)
else
  echo "ERROR: no quinlight-audio binary (set QUINLIGHT_BIN or install cargo)." >&2
  exit 1
fi

# Build per-engine --engine flags from QUINLIGHT_ENGINES (empty = all detected).
ENGINE_ARGS=()
for e in ${QUINLIGHT_ENGINES:-}; do ENGINE_ARGS+=(--engine "${e//,/}"); done

BEFORE="$OUT/quinlight-audio-96khz/rendered"
AFTER="$OUT/quinlight-audio-96khz/remastered"
rm -rf "$OUT"; mkdir -p "$BEFORE" "$AFTER"

echo "[1/3] before — deterministic render, no AI  ->  rendered/"
"${QUINLIGHT[@]}" convert "$MODS" --output "$BEFORE" \
    --format m4a --sample-rate "$RATE" --no-remaster

echo "[2/3] after — AI remaster (registration consensus)  ->  remastered/"
if [ "${STAGED:-0}" = "1" ]; then
  # Warm the per-engine sample cache one engine at a time (fits low-VRAM GPUs),
  # then the combined pass below is served from cache with no model contention.
  for e in ${QUINLIGHT_ENGINES:-}; do
    echo "    [warm] ${e//,/} alone"
    "${QUINLIGHT[@]}" convert "$MODS" --output "$OUT/_warm" \
        --format m4a --sample-rate "$RATE" --engine "${e//,/}" || true
  done
fi
"${QUINLIGHT[@]}" convert "$MODS" --output "$AFTER" \
    --format m4a --sample-rate "$RATE" --reduce robust "${ENGINE_ARGS[@]}"

echo "[3/3] packaging quinlight-audio-96khz-bundle.zip"
rm -rf "$OUT/_warm"
( cd "$OUT" && zip -qr quinlight-audio-96khz-bundle.zip quinlight-audio-96khz )

echo
echo "Bundle ready: $OUT/quinlight-audio-96khz-bundle.zip"
( cd "$OUT" && unzip -l quinlight-audio-96khz-bundle.zip | sed -n '1,40p' )
echo
echo "Next:"
echo "  1. gh release upload audio-bundle-v1 \"$OUT/quinlight-audio-96khz-bundle.zip\" --clobber"
echo "  2. gh workflow run pages.yml"
