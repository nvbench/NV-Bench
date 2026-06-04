"""
NV-Bench Configuration
======================
Single source of truth for every path and constant NV-Bench needs.

Design goals (so other people can actually run this):
  * No machine-specific absolute paths. Everything defaults to a location
    *inside the repo* and can be overridden with an environment variable.
  * `setup.sh` populates `third_party/` and `checkpoints/` with exactly the
    paths assumed here, so a fresh clone works with zero edits.
  * Every optional dependency (SIM / DNSMOS / FAD) resolves its assets the same
    way: env var -> repo-relative default.

Override any path by exporting the matching `NVBENCH_*` variable before running.
"""

import os

# ─── Repo layout ──────────────────────────────────────────────────────────
# Absolute path to this repository, regardless of where you launch from.
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Conda env names — single source of truth, shared with scripts/setup.sh
# (which honours the same NVBENCH_*_ENV overrides). The per-step env self-checks
# point users at these, so an override here keeps the guidance accurate.
CORE_ENV = os.environ.get("NVBENCH_CORE_ENV", "nvbench-core")
ACOUSTIC_ENV = os.environ.get("NVBENCH_ACOUSTIC_ENV", "nvbench-acoustic")


def _path(env_var: str, *default_parts: str) -> str:
    """Return ``$env_var`` if set, else a path built relative to the repo root.

    Keeping the fallback repo-relative is what makes a fresh clone runnable
    without editing this file.
    """
    val = os.environ.get(env_var)
    if val:
        return os.path.expanduser(val)
    return os.path.join(PROJECT_ROOT, *default_parts)


# Conventional locations populated by scripts/setup.sh
THIRD_PARTY_DIR = _path("NVBENCH_THIRD_PARTY_DIR", "third_party")
CHECKPOINT_DIR = _path("NVBENCH_CHECKPOINT_DIR", "checkpoints")
DATA_DIR = _path("NVBENCH_DATA_DIR", "data")
OUTPUT_DIR = _path("NVBENCH_OUTPUT_DIR", "results")

# ─── Core: NVASR model (Step 1) ─────────────────────────────────────────────
# Multilingual NVASR — https://huggingface.co/CharlesNi/Multilingual-NVASR
NVASR_MODEL_DIR = _path("NVBENCH_NVASR_MODEL_DIR", "checkpoints", "nvasr")
DEVICE = os.environ.get("NVBENCH_DEVICE", "cuda:0")

# ─── Acoustic env: Speaker Similarity / SIM (Step 3a) ───────────────────────
# WavLM checkpoint + the seed-tts-eval repo that provides the verification code.
# Download instructions: scripts/setup.sh (or the README "Acoustic env" section).
SIM_CKPT_PATH = _path("NVBENCH_SIM_CKPT", "checkpoints", "wavlm_large_finetune.pth")
SEED_TTS_EVAL_DIR = _path("NVBENCH_SEED_TTS_EVAL_DIR", "third_party", "seed-tts-eval")

# ─── Core env: DNSMOS (Step 3a) ─────────────────────────────────────────────
# Only the ONNX model is external; the scorer code is vendored at
# third_party/dnsmos (so no Amphion clone, no `models` namespace collision).
# setup.sh fetches just this 1 MB file.
DNSMOS_MODEL_PATH = _path("NVBENCH_DNSMOS_MODEL", "checkpoints", "dnsmos", "sig_bak_ovr.onnx")

# ─── Acoustic env: AudioLDM Eval / FAD-FD-KL (Step 3b) ──────────────────────
AUDIOLDM_EVAL_BACKBONE = os.environ.get("NVBENCH_FAD_BACKBONE", "cnn14")
AUDIOLDM_EVAL_SAMPLE_RATE = int(os.environ.get("NVBENCH_FAD_SAMPLE_RATE", "16000"))

# Cnn14 (PANNs) checkpoints for FAD. audioldm_eval loads these from ./ckpt
# RELATIVE TO CWD, so the default stays repo-relative ("ckpt") and you must run
# compute_fad.py from the repo root. setup.sh fetches them here.
FAD_CKPT_DIR = _path("NVBENCH_FAD_CKPT_DIR", "ckpt")
FAD_CNN14_16K = os.path.join(FAD_CKPT_DIR, "Cnn14_16k_mAP=0.438.pth")
FAD_CNN14 = os.path.join(FAD_CKPT_DIR, "Cnn14_mAP=0.431.pth")

# ─── NVV Taxonomy ───────────────────────────────────────────────────────────
# Mandarin vocabulary (13 categories)
VOCAB_ZH = [
    '[Laughter]', '[Cough]',
    '[Sigh]', '[Breathing]',
    '[Question-en]', '[Question-oh]', '[Question-ah]',
    '[Dissatisfaction-hnn]', '[Surprise-oh]', '[Surprise-ah]',
    '[Uhm]', '[Confirmation-en]', '[Question-ei]'
]

# English vocabulary (7 categories)
VOCAB_EN = [
    '[Laughter]', '[Cough]',
    # '[Sigh]', '[Breathing]',
    '[Surprise-oh]', '[Question-huh]', '[Uhm]'
]

# Paralinguistic Group A (Vegetative / Affect Burst)
PARA_GROUP_A = ['[Laughter]', '[Cough]']
# PARA_GROUP_A = ['[Laughter]', '[Cough]', '[Sigh]', '[Breathing]']
