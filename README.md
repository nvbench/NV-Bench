# NV-Bench

**NV-Bench: Benchmarking Nonverbal Vocalization Synthesis in Expressive Text-to-Speech Models**

[![Demo Page](https://img.shields.io/badge/Demo-Page-blue)](https://nvbench.github.io)
[![Dataset](https://img.shields.io/badge/Dataset-HuggingFace-yellow)](https://huggingface.co/datasets/AnonyData/NV-Bench)
[![Model](https://img.shields.io/badge/Model-HuggingFace-yellow)](https://huggingface.co/AnonyData/Multilingual-NVASR)

NV-Bench evaluates TTS models on their ability to generate **nonverbal vocalizations** (NVVs) — laughter, coughs, sighs, hesitations, and more — using a dual-dimensional evaluation protocol:

1. **Instruction Alignment** — Can the model produce the correct NVV events at the right positions? *(PCER / PWER)*
2. **Acoustic Fidelity** — How realistic and natural is the synthesized speech? *(DNSMOS, SIM, FAD / FD / KL)*

## Installation

```bash
git clone https://github.com/your-org/NV-Bench.git
cd NV-Bench
pip install -r requirements.txt
```

### Optional Dependencies

<details>
<summary><b>Speaker Similarity (SIM)</b></summary>

Requires [seed-tts-eval](https://github.com/BytedanceSpeech/seed-tts-eval) and a WavLM checkpoint:

```bash
git clone https://github.com/BytedanceSpeech/seed-tts-eval.git ./third_party/seed-tts-eval
# Download wavlm_large_finetune.pth into ./models/
```

Set in environment or `config.py`:
```bash
export NVBENCH_SIM_CKPT=/path/to/wavlm_large_finetune.pth
export NVBENCH_SEED_TTS_EVAL_DIR=/path/to/seed-tts-eval
```
</details>

<details>
<summary><b>DNSMOS</b></summary>

Requires the DNSMOS ONNX model from [Amphion](https://github.com/open-mmlab/Amphion):

```bash
export NVBENCH_DNSMOS_MODEL=/path/to/sig_bak_ovr.onnx
```
</details>

<details>
<summary><b>FAD / FD / KL</b></summary>

`audioldm_eval` has a broken transitive dependency. Install manually:

```bash
pip install ssr_eval --no-deps
pip install git+https://github.com/haoheliu/audioldm_eval --no-deps
pip install scipy scikit-image transformers torchlibrosa
```
</details>

## Quick Start

### Full Pipeline (NVASR Inference → Evaluation → Acoustic Metrics)

```bash
python run_pipeline.py \
    --input_json /path/to/testset.json \
    --model_dir /path/to/nvasr_model \
    --lang zh \
    --output_dir ./results/model_name \
    --steps infer,evaluate,acoustic
```

### Step-by-Step

```bash
# Step 1: NVASR Inference — transcribe TTS outputs
python infer.py \
    --input_json /path/to/testset.json \
    --model_dir /path/to/nvasr_model \
    --output_dir ./results/model_name \
    --lang auto

# Step 2: Instruction Alignment Evaluation
python evaluate.py \
    --input ./results/model_name/infer_results.json \
    --lang zh \
    --output ./results/model_name/eval_results.json

# Step 3a: Per-sample Acoustic Metrics (DNSMOS, SIM)
python compute_acoustic.py \
    --input_json /path/to/testset.json \
    --output_dir ./results/model_name \
    --metrics dnsmos,sim

# Step 3b: Distribution-level Metrics (FAD / FD / KL)
python compute_fad.py \
    --pairs gt_zh.json:gen_zh.json gt_en.json:gen_en.json \
    --output_json ./results/model_name/fad_results.json
```

> [!IMPORTANT]
> **FAD / FD / KL** are distribution-level metrics and must be computed using `compute_fad.py`.
> All subsets (zh + en) should be pooled together for stable estimates.

## Data Format

Each item in the benchmark JSON contains:

| Field | Description |
|-------|-------------|
| `wav_path` | Ground-truth reference audio path |
| `text` | Target text with NVV tags, e.g. `今天很开心[Laughter]` |
| `ref_wav_path` | Reference audio for speaker similarity |
| `ref_text` | Reference text for speaker prompt |
| `target_wav_path` | TTS-generated audio to evaluate |

## Evaluation Metrics

### Instruction Alignment (`evaluate.py`)

| Metric | Description |
|--------|-------------|
| **OCER / OWER** | Overall Character/Word Error Rate (text + NVV tags) |
| **PCER / PWER** | Paralinguistic CER/WER (NVV tags only) |
| **CER / WER** | Text-only error rate (NVV tags removed) |

### Acoustic Fidelity — Per-sample (`compute_acoustic.py`)

| Metric | Description |
|--------|-------------|
| **DNSMOS** | Perceptual speech quality (MOS prediction) |
| **SIM** | Speaker similarity via WavLM cosine similarity |

### Acoustic Fidelity — Distribution-level (`compute_fad.py`)

| Metric | Description |
|--------|-------------|
| **FAD** | Fréchet Audio Distance |
| **FD** | Fréchet Distance |
| **KL** | KL Divergence (sigmoid & softmax) |

## NVV Taxonomy

NVVs are organized into three functional levels:

| Level | Function | Categories |
|-------|----------|------------|
| **Level 1** | Vegetative | `[Laughter]`, `[Cough]`, `[Sigh]`, `[Breathing]` |
| **Level 2** | Affect Burst | `[Surprise-oh]`, `[Surprise-ah]`, `[Dissatisfaction-hnn]` |
| **Level 3** | Conversational Grunt | `[Uhm]`, `[Question-en/oh/ah/ei/huh]`, `[Confirmation-en]` |

> [!NOTE]
> Mandarin supports 13 NVV categories; English supports 7 categories.

## Project Structure

```
NV-Bench/
├── config.py              # Centralized configuration (env-overridable)
├── infer.py               # Step 1: NVASR inference
├── evaluate.py            # Step 2: Instruction Alignment (CER/WER/PCER/PWER)
├── compute_acoustic.py    # Step 3a: Per-sample metrics (DNSMOS, SIM)
├── compute_fad.py         # Step 3b: Distribution metrics (FAD, FD, KL)
├── run_pipeline.py        # Pipeline orchestrator (Steps 1-3a)
├── model.py               # SenseVoiceSmall NVASR model
├── data/
│   └── loader.py          # Benchmark JSON data loader
├── utils/
│   ├── postprocess.py     # NVASR output post-processing
│   ├── cer_accuracy.py    # CER/WER computation utilities
│   └── ctc_alignment.py   # CTC forced alignment
├── requirements.txt
└── README.md
```

## Configuration

All paths are configurable via environment variables:

```bash
export NVBENCH_NVASR_MODEL_DIR=/path/to/nvasr_model
export NVBENCH_DEVICE=cuda:0
export NVBENCH_DATA_DIR=/path/to/benchmark_data
export NVBENCH_OUTPUT_DIR=./results
export NVBENCH_SIM_CKPT=/path/to/wavlm_large_finetune.pth
export NVBENCH_SEED_TTS_EVAL_DIR=/path/to/seed-tts-eval
export NVBENCH_DNSMOS_MODEL=/path/to/sig_bak_ovr.onnx
```

## Citation

```bibtex
Coming Soon
}
```

## License

This project is for research purposes only.
