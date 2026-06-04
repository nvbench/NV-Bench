#!/usr/bin/env bash
# ===========================================================================
# NV-Bench one-command setup
# ===========================================================================
# Builds the TWO conda environments NV-Bench needs and provisions every
# third-party repo and checkpoint, so a fresh clone is runnable end-to-end.
#
#   nvbench-core      (py3.10, modern torch)  -> Step 1 infer, 2 evaluate, 3a DNSMOS
#   nvbench-acoustic  (py3.9,  torch 2.5.1)   -> Step 3a SIM, 3b FAD/FD/KL
#
# Why two envs: funasr needs a modern torch; audioldm_eval (FAD) needs
# numpy<2 + older torch; the s3prl WavLM stack (SIM) rides with the latter.
# These cannot coexist in one env — see README "Environments".
#
# Usage:
#   bash scripts/setup.sh                 # everything
#   bash scripts/setup.sh envs            # just create/populate the conda envs
#   bash scripts/setup.sh assets          # just clone third-party + fetch checkpoints
#   bash scripts/setup.sh check           # just run the env self-check
#
# Already have an asset somewhere on disk? Point an env var at it and setup
# will symlink it instead of downloading:
#   NVBENCH_SIM_CKPT=/path/wavlm_large_finetune.pth \
#   NVBENCH_DNSMOS_MODEL=/path/sig_bak_ovr.onnx \
#   NVBENCH_CNN14_16K=/path/Cnn14_16k_mAP=0.438.pth \
#   NVBENCH_CNN14=/path/Cnn14_mAP=0.431.pth \
#   NVBENCH_NVASR_MODEL_DIR=/path/Multilingual-NVASR \
#       bash scripts/setup.sh
# ===========================================================================
set -euo pipefail

# ── locate repo root (this script lives in <root>/scripts) ──
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CORE_ENV="${NVBENCH_CORE_ENV:-nvbench-core}"
ACOUSTIC_ENV="${NVBENCH_ACOUSTIC_ENV:-nvbench-acoustic}"
THIRD_PARTY="$ROOT/third_party"
CKPT_DL="$ROOT/checkpoints"   # SIM / DNSMOS checkpoints
CKPT_FAD="$ROOT/ckpt"         # FAD Cnn14 checkpoints (audioldm_eval reads ./ckpt)

# torch wheels (override if your CUDA differs)
CORE_TORCH_INDEX="${NVBENCH_CORE_TORCH_INDEX:-}"               # "" = default PyPI
ACOUSTIC_TORCH_INDEX="${NVBENCH_ACOUSTIC_TORCH_INDEX:-https://download.pytorch.org/whl/cu124}"

log()  { printf '\n\033[1;36m▶ %s\033[0m\n' "$*"; }
ok()   { printf '  \033[1;32m✓\033[0m %s\n' "$*"; }
warn() { printf '  \033[1;33m!\033[0m %s\n' "$*"; }

CONDA_BASE=""
need_conda() {
  if ! command -v conda >/dev/null 2>&1; then
    echo "ERROR: conda not found on PATH. Install Miniconda first." >&2
    exit 1
  fi
  CONDA_BASE="$(conda info --base)"
  # shellcheck disable=SC1091
  source "$CONDA_BASE/etc/profile.d/conda.sh"
}

env_exists() { conda env list | awk '{print $1}' | grep -qx "$1"; }

# Absolute paths to an env's python/pip. We do NOT use `conda run -n ENV python`
# because some machines force-activate another env early on PATH, and conda run
# then picks the wrong interpreter. Absolute paths are unambiguous.
env_py()  { echo "$CONDA_BASE/envs/$1/bin/python"; }
env_pip() { echo "$CONDA_BASE/envs/$1/bin/pip"; }

# symlink $1 -> $2 if source exists; return 1 otherwise
link_if_present() {
  local src="$1" dst="$2"
  if [[ -n "$src" && -e "$src" ]]; then
    mkdir -p "$(dirname "$dst")"; ln -sf "$src" "$dst"; ok "linked $(basename "$dst") <- $src"; return 0
  fi
  return 1
}

# download $1 (url) -> $2 (dest) unless dest exists
fetch() {
  local url="$1" dst="$2"
  [[ -e "$dst" ]] && { ok "$(basename "$dst") already present"; return 0; }
  mkdir -p "$(dirname "$dst")"
  log "downloading $(basename "$dst")"
  if command -v wget >/dev/null 2>&1; then wget -q --show-progress -O "$dst" "$url" || { rm -f "$dst"; return 1; }
  else curl -fL --progress-bar -o "$dst" "$url" || { rm -f "$dst"; return 1; }; fi
}

# ───────────────────────── envs ─────────────────────────
setup_envs() {
  need_conda

  log "core env: $CORE_ENV (python 3.10)"
  env_exists "$CORE_ENV" && warn "$CORE_ENV exists — skipping create (delete it to rebuild)" \
                         || conda create -y -n "$CORE_ENV" python=3.10
  local CORE_PIP; CORE_PIP="$(env_pip "$CORE_ENV")"
  if [[ -n "$CORE_TORCH_INDEX" ]]; then
    "$CORE_PIP" install torch torchaudio --index-url "$CORE_TORCH_INDEX"
  fi
  "$CORE_PIP" install -r requirements/core.txt
  ok "$CORE_ENV ready"

  log "acoustic env: $ACOUSTIC_ENV (python 3.9, torch 2.5.1)"
  env_exists "$ACOUSTIC_ENV" && warn "$ACOUSTIC_ENV exists — skipping create (delete it to rebuild)" \
                             || conda create -y -n "$ACOUSTIC_ENV" python=3.9
  local AC_PIP; AC_PIP="$(env_pip "$ACOUSTIC_ENV")"
  # torch first, pinned, from the CUDA index
  "$AC_PIP" install torch==2.5.1 torchaudio==2.5.1 --index-url "$ACOUSTIC_TORCH_INDEX"
  # SIM + numpy<2 stack
  "$AC_PIP" install -r requirements/acoustic.txt
  # FAD libraries that MUST be installed with --no-deps (broken transitive deps)
  log "FAD libs (--no-deps; ssr_eval/audioldm_eval have broken transitive deps)"
  "$AC_PIP" install --no-deps "ssr_eval==0.0.7"
  "$AC_PIP" install --no-deps "git+https://github.com/haoheliu/audioldm_eval"
  "$AC_PIP" install "scikit-image==0.24.0" "transformers==4.36.0" "torchlibrosa==0.1.0"
  ok "$ACOUSTIC_ENV ready"
}

# ─────────────────────── assets ─────────────────────────
setup_assets() {
  # --- third-party: seed-tts-eval (SIM verification code) ---
  log "third-party: seed-tts-eval"
  if [[ -d "$THIRD_PARTY/seed-tts-eval/.git" ]]; then
    ok "seed-tts-eval already cloned"
  elif link_if_present "${NVBENCH_SEED_TTS_EVAL_DIR:-}" "$THIRD_PARTY/seed-tts-eval"; then :; else
    git clone --recursive https://github.com/BytedanceSpeech/seed-tts-eval.git "$THIRD_PARTY/seed-tts-eval"
    ok "cloned seed-tts-eval"
  fi

  # --- DNSMOS ONNX (scorer code is vendored in third_party/dnsmos) ---
  log "checkpoint: DNSMOS sig_bak_ovr.onnx"
  if ! link_if_present "${NVBENCH_DNSMOS_MODEL:-}" "$CKPT_DL/dnsmos/sig_bak_ovr.onnx"; then
    fetch "https://github.com/microsoft/DNS-Challenge/raw/master/DNSMOS/DNSMOS/sig_bak_ovr.onnx" \
          "$CKPT_DL/dnsmos/sig_bak_ovr.onnx" || warn "DNSMOS download failed — set NVBENCH_DNSMOS_MODEL"
  fi

  # --- WavLM finetune checkpoint (SIM) ---
  log "checkpoint: wavlm_large_finetune.pth (SIM)"
  if ! link_if_present "${NVBENCH_SIM_CKPT:-}" "$CKPT_DL/wavlm_large_finetune.pth"; then
    warn "No WavLM checkpoint. Download wavlm_large_finetune.pth from the seed-tts-eval"
    warn "release (https://github.com/BytedanceSpeech/seed-tts-eval) and place it at:"
    warn "  $CKPT_DL/wavlm_large_finetune.pth   (or set NVBENCH_SIM_CKPT)"
  fi

  # --- FAD Cnn14 checkpoints into ./ckpt (audioldm_eval reads RELATIVE ./ckpt) ---
  log "checkpoints: Cnn14 (FAD) -> ./ckpt"
  link_if_present "${NVBENCH_CNN14_16K:-}" "$CKPT_FAD/Cnn14_16k_mAP=0.438.pth" \
    || fetch "https://zenodo.org/records/3987831/files/Cnn14_16k_mAP=0.438.pth" "$CKPT_FAD/Cnn14_16k_mAP=0.438.pth" \
    || warn "Cnn14_16k download failed — set NVBENCH_CNN14_16K"
  link_if_present "${NVBENCH_CNN14:-}" "$CKPT_FAD/Cnn14_mAP=0.431.pth" \
    || fetch "https://zenodo.org/records/3576403/files/Cnn14_mAP=0.431.pth" "$CKPT_FAD/Cnn14_mAP=0.431.pth" \
    || warn "Cnn14 download failed — set NVBENCH_CNN14"

  # --- NVASR model (Step 1) ---
  log "checkpoint: Multilingual-NVASR (Step 1)"
  if ! link_if_present "${NVBENCH_NVASR_MODEL_DIR:-}" "$CKPT_DL/nvasr"; then
    warn "No NVASR model. Download from https://huggingface.co/CharlesNi/Multilingual-NVASR"
    warn "  e.g.  huggingface-cli download CharlesNi/Multilingual-NVASR --local-dir $CKPT_DL/nvasr"
    warn "  (or set NVBENCH_NVASR_MODEL_DIR)"
  fi
}

# ─────────────────────── check ──────────────────────────
run_check() {
  need_conda
  log "self-check: $CORE_ENV (core role)"
  "$(env_py "$CORE_ENV")" scripts/check_env.py core || true
  log "self-check: $ACOUSTIC_ENV (acoustic role)"
  "$(env_py "$ACOUSTIC_ENV")" scripts/check_env.py acoustic || true
}

case "${1:-all}" in
  envs)   setup_envs ;;
  assets) setup_assets ;;
  check)  run_check ;;
  all)    setup_envs; setup_assets; run_check ;;
  *) echo "usage: bash scripts/setup.sh [all|envs|assets|check]"; exit 1 ;;
esac

log "Done. Activate an env and run:  conda activate $CORE_ENV  (then see README Quick Start)"
