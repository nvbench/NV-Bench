# NV-Bench

**NV-Bench: Benchmarking Nonverbal Vocalization Synthesis in Expressive Text-to-Speech Models**

[![Paper](https://img.shields.io/badge/arXiv-2603.15352-b31b1b)](https://arxiv.org/abs/2603.15352)
[![Demo Page](https://img.shields.io/badge/Demo-Page-blue)](https://charlesnii.github.io/nvbench.github.io/)
[![Dataset](https://img.shields.io/badge/Dataset-HuggingFace-yellow)](https://huggingface.co/datasets/CharlesNi/NV-Bench)
[![Model](https://img.shields.io/badge/Model-HuggingFace-yellow)](https://huggingface.co/CharlesNi/Multilingual-NVASR)

NV-Bench evaluates TTS models on their ability to generate **nonverbal vocalizations** (NVVs) — laughter, coughs, sighs, hesitations, and more — using a dual-dimensional evaluation protocol:

1. **Instruction Alignment** — Can the model produce the correct NVV events at the right positions? *(PCER / PWER)*
2. **Acoustic Fidelity** — How realistic and natural is the synthesized speech? *(DNSMOS, SIM, FAD / FD / KL)*

---

## TL;DR — Install & Run

```bash
git clone https://github.com/nvbench/NV-Bench.git
cd NV-Bench
bash scripts/setup.sh        # builds 2 conda envs + fetches all assets, then self-checks
```

```bash
# Steps 1–3a (DNSMOS) — core env
conda activate nvbench-core
python run_pipeline.py --input_json data/testset_zh.json --lang zh \
    --output_dir results/my_model --steps infer,evaluate,acoustic

# Steps 3a (SIM) + 3b (FAD/FD/KL) — acoustic env
conda activate nvbench-acoustic
python compute_acoustic.py --input_json data/testset_zh.json --output_dir results/my_model --metrics sim
python compute_fad.py --pairs data/gt_zh.json:results/my_model/gen_zh.json --output_json results/my_model/fad.json
```

---

## Environments

> [!IMPORTANT]
> **NV-Bench needs two conda environments.** This is not optional bureaucracy — the
> metric backends pull *mutually incompatible* dependency stacks and cannot live
> together. `scripts/setup.sh` builds both for you.

| Env | Python / torch | Steps it runs | Why it must be separate |
|-----|----------------|---------------|--------------------------|
| **`nvbench-core`** | 3.10 / torch ≥2.4 | 1 `infer`, 2 `evaluate`, 3a **DNSMOS** | `funasr` (NVASR) requires a modern torch; DNSMOS is ONNX-only and rides along. |
| **`nvbench-acoustic`** | 3.9 / torch 2.5.1 / **numpy<2** | 3a **SIM**, 3b **FAD/FD/KL** | `audioldm_eval` (FAD) hard-requires `numpy<2` + older torch; the `s3prl` WavLM stack (SIM) is happiest there too. |

Every step script **fails fast with an explicit message** if you run it in the
wrong env (telling you which env to switch to and what is missing), so you can't
silently get half-broken results. Run the doctor anytime:

```bash
python scripts/check_env.py            # checks both roles against the active env
python scripts/check_env.py acoustic   # just the acoustic role
```

---

## Installation

### One command (recommended)

```bash
bash scripts/setup.sh           # envs + assets + self-check
# or piecewise:
bash scripts/setup.sh envs      # just the two conda envs
bash scripts/setup.sh assets    # just third-party repos + checkpoints
bash scripts/setup.sh check     # just the self-check
```

If your CUDA differs from the defaults, override the torch wheel index:

```bash
NVBENCH_ACOUSTIC_TORCH_INDEX=https://download.pytorch.org/whl/cu121 bash scripts/setup.sh envs
```

### Already have the assets somewhere on disk?

`setup.sh` will **symlink** existing files instead of downloading, if you point it at them:

```bash
NVBENCH_NVASR_MODEL_DIR=/path/Multilingual-NVASR \
NVBENCH_SIM_CKPT=/path/wavlm_large_finetune.pth \
NVBENCH_DNSMOS_MODEL=/path/sig_bak_ovr.onnx \
NVBENCH_CNN14_16K=/path/Cnn14_16k_mAP=0.438.pth \
NVBENCH_CNN14=/path/Cnn14_mAP=0.431.pth \
NVBENCH_SEED_TTS_EVAL_DIR=/path/seed-tts-eval \
    bash scripts/setup.sh
```

### Assets fetched by setup

| Asset | Used by | Lands at | Source |
|-------|---------|----------|--------|
| Multilingual-NVASR model | Step 1 | `checkpoints/nvasr/` | [HF: CharlesNi/Multilingual-NVASR](https://huggingface.co/CharlesNi/Multilingual-NVASR) |
| `sig_bak_ovr.onnx` (DNSMOS) | Step 3a | `checkpoints/dnsmos/` | Microsoft DNS-Challenge |
| `wavlm_large_finetune.pth` | Step 3a SIM | `checkpoints/` | [seed-tts-eval](https://github.com/BytedanceSpeech/seed-tts-eval) release |
| seed-tts-eval repo | Step 3a SIM | `third_party/seed-tts-eval/` | GitHub |
| `Cnn14_16k_mAP=0.438.pth` + `Cnn14_mAP=0.431.pth` | Step 3b FAD | `ckpt/` | Zenodo (PANNs) |

> The DNSMOS *scorer code* is **vendored** at `third_party/dnsmos/` (CC BY 4.0), so
> you do **not** need to clone Amphion. Only the 1 MB ONNX model is fetched.

---

## Quick Start

### Full core pipeline (Steps 1 → 2 → 3a-DNSMOS)

```bash
conda activate nvbench-core
python run_pipeline.py \
    --input_json /path/to/testset.json \
    --model_dir  checkpoints/nvasr \
    --lang zh \
    --output_dir ./results/model_name \
    --steps infer,evaluate,acoustic
```

### Step-by-step

```bash
# ── core env ───────────────────────────────────────────────
conda activate nvbench-core

# Step 1: NVASR Inference — transcribe TTS outputs
python infer.py --input_json /path/to/testset.json --output_dir ./results/model_name --lang auto

# Step 2: Instruction Alignment Evaluation
python evaluate.py --input ./results/model_name/infer_results.json --lang zh \
    --output ./results/model_name/eval_results.json

# Step 3a (DNSMOS): per-sample perceptual quality
python compute_acoustic.py --input_json /path/to/testset.json --output_dir ./results/model_name --metrics dnsmos

# ── acoustic env ───────────────────────────────────────────
conda activate nvbench-acoustic

# Step 3a (SIM): speaker similarity (WavLM)
python compute_acoustic.py --input_json /path/to/testset.json --output_dir ./results/model_name --metrics sim

# Step 3b: distribution-level metrics (FAD / FD / KL)
python compute_fad.py \
    --pairs gt_zh.json:gen_zh.json gt_en.json:gen_en.json \
    --output_json ./results/model_name/fad_results.json
```

> [!IMPORTANT]
> **FAD / FD / KL** are distribution-level metrics. Pool all subsets (zh + en)
> in a single `compute_fad.py` call for stable estimates. Run it **from the repo
> root** — `audioldm_eval` loads its Cnn14 checkpoint from `./ckpt/` relative to
> the current directory.

---

## Data Format

Each item in the benchmark JSON contains:

| Field | Description |
|-------|-------------|
| `wav_path` | Ground-truth reference audio path |
| `text` | Target text with NVV tags, e.g. `今天很开心[Laughter]` |
| `ref_wav_path` | Reference audio for speaker similarity |
| `ref_text` | Reference text for speaker prompt |
| `target_wav_path` | TTS-generated audio to evaluate |

---

## Evaluation Metrics

### Instruction Alignment (`evaluate.py`) — core env

| Metric | Description |
|--------|-------------|
| **OCER / OWER** | Overall Character/Word Error Rate (text + NVV tags) |
| **PCER / PWER** | Paralinguistic CER/WER (NVV tags only) |
| **CER / WER** | Text-only error rate (NVV tags removed) |

### Acoustic Fidelity — Per-sample (`compute_acoustic.py`)

| Metric | Env | Description |
|--------|-----|-------------|
| **DNSMOS** | core | Perceptual speech quality (MOS prediction) |
| **SIM** | acoustic | Speaker similarity via WavLM cosine similarity |

### Acoustic Fidelity — Distribution-level (`compute_fad.py`) — acoustic env

| Metric | Description |
|--------|-------------|
| **FAD** | Fréchet Audio Distance |
| **FD** | Fréchet Distance |
| **KL** | KL Divergence (sigmoid & softmax) |

---

## NVV Taxonomy

NVVs are organized into three functional levels:

| Level | Function | Categories |
|-------|----------|------------|
| **Level 1** | Vegetative | `[Laughter]`, `[Cough]`, `[Sigh]`, `[Breathing]` |
| **Level 2** | Affect Burst | `[Surprise-oh]`, `[Surprise-ah]`, `[Dissatisfaction-hnn]` |
| **Level 3** | Conversational Grunt | `[Uhm]`, `[Question-en/oh/ah/ei/huh]`, `[Confirmation-en]` |

> [!NOTE]
> Mandarin supports 13 NVV categories; English supports 7 categories.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `environment not ready for Step ...` banner | Wrong conda env active | Switch to the env the banner names (`conda activate ...`). |
| `compute_fad.py` hangs then `FileNotFoundError: ./ckpt/Cnn14_16k_mAP=0.438.pth` | Cnn14 checkpoint missing, or you ran it from another directory | Run from the repo root; `bash scripts/setup.sh assets` places the checkpoints in `./ckpt/`. |
| FAD tries to download a 1.3 GB file from Zenodo | The Cnn14 gate file is absent so `audioldm_eval` re-downloads | Ensure both `ckpt/Cnn14_16k_mAP=0.438.pth` and `ckpt/Cnn14_mAP=0.431.pth` exist. |
| SIM first run is slow / "Downloading s3prl/s3prl" | `torch.hub` fetches the s3prl repo + WavLM weights once | Let it finish (cached afterward); ensure network/proxy is reachable. |
| `numpy` errors inside `audioldm_eval` | numpy≥2 in the acoustic env | The acoustic env must keep `numpy<2`; rebuild with `requirements/acoustic.txt`. |
| `ModuleNotFoundError: MySQL-python` while installing FAD libs | `ssr_eval`/`audioldm_eval` have a broken transitive dep | Install them with `--no-deps` (setup.sh already does this). |
| `import fairseq` errors | Not needed — `s3prl 0.4.x` has a native WavLM | Ignore; SIM does not require fairseq. |

---

## Project Structure

```
NV-Bench/
├── config.py              # Centralized, env-overridable config (no machine paths)
├── infer.py               # Step 1: NVASR inference                [core]
├── evaluate.py            # Step 2: Instruction Alignment           [core]
├── compute_acoustic.py    # Step 3a: DNSMOS [core] + SIM [acoustic]
├── compute_fad.py         # Step 3b: FAD / FD / KL                  [acoustic]
├── run_pipeline.py        # Orchestrator for Steps 1–3a (DNSMOS)    [core]
├── model.py               # SenseVoiceSmall NVASR model
├── data/loader.py         # Benchmark JSON loader
├── utils/
│   ├── envcheck.py        # Fail-fast env self-check used by every step
│   ├── postprocess.py     # NVASR output post-processing
│   ├── cer_accuracy.py    # CER/WER computation
│   └── ctc_alignment.py   # CTC forced alignment
├── third_party/
│   ├── dnsmos/            # Vendored DNSMOS scorer (CC BY 4.0)
│   └── seed-tts-eval/     # Cloned by setup.sh (SIM verification code)
├── requirements/
│   ├── core.txt           # nvbench-core deps
│   └── acoustic.txt       # nvbench-acoustic deps
├── scripts/
│   ├── setup.sh           # One-command env + asset setup
│   └── check_env.py       # Environment doctor
├── checkpoints/           # nvasr/, dnsmos/, wavlm (populated by setup.sh)
├── ckpt/                  # Cnn14 FAD checkpoints (audioldm_eval reads ./ckpt)
└── README.md
```

---

## Configuration

Every path is overridable via environment variable; defaults are **relative to
the repo**, so a fresh clone + `setup.sh` works with no edits.

```bash
export NVBENCH_NVASR_MODEL_DIR=checkpoints/nvasr
export NVBENCH_DEVICE=cuda:0
export NVBENCH_OUTPUT_DIR=./results
export NVBENCH_SIM_CKPT=checkpoints/wavlm_large_finetune.pth
export NVBENCH_SEED_TTS_EVAL_DIR=third_party/seed-tts-eval
export NVBENCH_DNSMOS_MODEL=checkpoints/dnsmos/sig_bak_ovr.onnx
```

---

## Acknowledgements

NV-Bench builds on several excellent open-source projects, and we thank their authors:

- [SenseVoice / FunASR](https://github.com/FunAudioLLM/SenseVoice) — backbone for the Multilingual-NVASR transcription model (Step 1).
- [seed-tts-eval](https://github.com/BytedanceSpeech/seed-tts-eval), [UniSpeech](https://github.com/microsoft/UniSpeech), and [s3prl](https://github.com/s3prl/s3prl) — WavLM-based speaker similarity (SIM, Step 3a).
- [DNSMOS](https://github.com/microsoft/DNS-Challenge/tree/master/DNSMOS) (via [Amphion](https://github.com/open-mmlab/Amphion)) — perceptual quality scorer (Step 3a); the scorer is vendored under `third_party/dnsmos/` (CC BY 4.0).
- [audioldm_eval](https://github.com/haoheliu/audioldm_eval) and [PANNs](https://github.com/qiuqiangkong/audioset_tagging_cnn) — distribution-level FAD / FD / KL metrics (Step 3b).

## Citation

If you use NV-Bench, please cite our paper ([arXiv:2603.15352](https://arxiv.org/abs/2603.15352)):

```bibtex
@article{ni2026nv,
  title={NV-Bench: Benchmark of Nonverbal Vocalization Synthesis for Expressive Text-to-Speech Generation},
  author={Ni, Qinke and Liao, Huan and Chen, Dekun and Wang, Yuxiang and Wu, Zhizheng},
  journal={arXiv preprint arXiv:2603.15352},
  year={2026}
}
```

## License

This project is for research purposes only. The vendored DNSMOS scorer
(`third_party/dnsmos/`) is licensed separately under CC BY 4.0 (see its header).
