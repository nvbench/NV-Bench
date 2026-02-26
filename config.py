"""
NV-Bench Configuration
======================
All paths can be overridden via environment variables or CLI arguments.
"""

import os

# ─── Paths (set via environment variables) ────
# NVASR model directory (Multi-lingual NVASR)
NVASR_MODEL_DIR = os.environ.get("NVBENCH_NVASR_MODEL_DIR", "./models/nvasr")
DEVICE = os.environ.get("NVBENCH_DEVICE", "cuda:0")

# Output directory
OUTPUT_DIR = os.environ.get("NVBENCH_OUTPUT_DIR", "./results")

# Speaker Similarity — WavLM checkpoint
# Download from: https://github.com/BytedanceSpeech/seed-tts-eval
SIM_CKPT_PATH = os.environ.get("NVBENCH_SIM_CKPT", "./models/wavlm_large_finetune.pth")
SEED_TTS_EVAL_DIR = os.environ.get("NVBENCH_SEED_TTS_EVAL_DIR", "./third_party/seed-tts-eval")

# DNSMOS — ONNX model path
# Download from: https://github.com/open-mmlab/Amphion
DNSMOS_MODEL_PATH = os.environ.get("NVBENCH_DNSMOS_MODEL", "./models/sig_bak_ovr.onnx")

# ─── AudioLDM Eval (FAD / FD / KL) ───────────
AUDIOLDM_EVAL_BACKBONE = "cnn14"
AUDIOLDM_EVAL_SAMPLE_RATE = 16000

# ─── NVV Taxonomy ─────────────────────────────
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
    '[Sigh]', '[Breathing]',
    '[Surprise-oh]', '[Question-huh]', '[Uhm]'
]

# Paralinguistic Group A (Vegetative / Affect Burst)
PARA_GROUP_A = ['[Laughter]', '[Cough]', '[Sigh]', '[Breathing]']
