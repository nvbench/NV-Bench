"""
NV-Bench Acoustic Fidelity Evaluation
======================================
Compute per-sample acoustic fidelity metrics for TTS-generated speech.

Metrics:
    - DNSMOS — Perceptual speech quality (MOS prediction)
    - SIM — Speaker similarity (WavLM cosine similarity)

Note: For distribution-level metrics (FAD / FD / KL), use compute_fad.py instead.

Usage:
    python compute_acoustic.py --input_json /path/to/testset.json --output_dir ./results --metrics dnsmos,sim
"""

import os
import sys
import json
import argparse
import numpy as np
from tqdm import tqdm
from typing import List, Dict, Optional

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from config import (
    DNSMOS_MODEL_PATH,
    SIM_CKPT_PATH, SEED_TTS_EVAL_DIR,
    CORE_ENV, ACOUSTIC_ENV,
)


# ─────────────────────────────────────────────
# DNSMOS
# ─────────────────────────────────────────────
def compute_dnsmos(
    data: List[Dict],
    model_path: str = DNSMOS_MODEL_PATH,
    wav_key: str = "target_wav_path",
    sr: int = 16000,
) -> Dict:
    """Compute DNSMOS scores for all items.

    Args:
        data: List of benchmark data dicts.
        model_path: Path to DNSMOS ONNX model.
        wav_key: Key for audio path in data dicts.
        sr: Sample rate.

    Returns:
        Dict with 'avg_dnsmos', 'scores', 'scored_count', 'skipped_count'.
    """
    # DNSMOS scorer is vendored at third_party/dnsmos (no Amphion clone, and no
    # top-level `models` package to collide with seed-tts-eval's `models`).
    if not os.path.exists(model_path):
        print(f"[DNSMOS] ONNX model not found: {model_path}")
        print("[DNSMOS] Set NVBENCH_DNSMOS_MODEL or run scripts/setup.sh to fetch it.")
        return {"error": f"missing model: {model_path}"}

    from third_party.dnsmos import dnsmos

    try:
        import torch
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        device_name = "cpu"

    print(f"[DNSMOS] Loading model from {model_path} on {device_name}...")
    
    # Check available ONNX Runtime providers and use GPU if available
    import onnxruntime as ort
    available_providers = ort.get_available_providers()
    print(f"[DNSMOS] Available ONNX providers: {available_providers}")
    
    if device_name == "cuda" and "CUDAExecutionProvider" not in available_providers:
        print("[DNSMOS] Warning: CUDA is available but CUDAExecutionProvider not found in ONNX Runtime.")
        print("[DNSMOS] Please install onnxruntime-gpu: pip install onnxruntime-gpu")
        print("[DNSMOS] Falling back to CPU...")
        device_name = "cpu"
    
    dnsmos_compute_score = dnsmos.ComputeScore(model_path, device_name)

    scores = []
    skipped = 0

    for item in tqdm(data, desc="Computing DNSMOS"):
        wav_path = item.get(wav_key) or item.get("target_path") or item.get("wav_path")
        if not wav_path or not os.path.exists(wav_path):
            skipped += 1
            continue

        try:
            score_dict = dnsmos_compute_score(wav_path, sr, False)
            mos_ovrl = float(score_dict["OVRL"])
            item["dnsmos"] = mos_ovrl
            scores.append(mos_ovrl)
        except Exception as e:
            print(f"[DNSMOS] Error for {wav_path}: {e}")

    result = {
        "avg_dnsmos": float(np.mean(scores)) if scores else 0.0,
        "min_dnsmos": float(min(scores)) if scores else 0.0,
        "max_dnsmos": float(max(scores)) if scores else 0.0,
        "scored_count": len(scores),
        "skipped_count": skipped,
    }

    print(f"[DNSMOS] Average: {result['avg_dnsmos']:.4f} "
          f"(min={result['min_dnsmos']:.4f}, max={result['max_dnsmos']:.4f})")
    print(f"[DNSMOS] Scored: {len(scores)}, Skipped: {skipped}")
    return result


# ─────────────────────────────────────────────
# Speaker Similarity (SIM)
# ─────────────────────────────────────────────
def compute_sim(
    data: List[Dict],
    sim_ckpt: str = SIM_CKPT_PATH,
    device: str = "cuda:0",
    wav_key: str = "target_wav_path",
    ref_wav_key: str = "ref_wav_path",
) -> Dict:
    """Compute speaker similarity between generated and reference audio.

    Args:
        data: List of benchmark data dicts (must contain ref_wav_path).
        sim_ckpt: Path to WavLM checkpoint.
        device: Device for model.
        wav_key: Key for generated audio path.
        ref_wav_key: Key for reference audio path.

    Returns:
        Dict with 'avg_sim', 'scores', 'count'.
    """
    speaker_verification_path = os.path.join(SEED_TTS_EVAL_DIR, 'thirdparty/UniSpeech/downstreams/speaker_verification')
    if speaker_verification_path not in sys.path:
        sys.path.insert(0, speaker_verification_path)
    if SEED_TTS_EVAL_DIR not in sys.path:
        sys.path.insert(0, SEED_TTS_EVAL_DIR)

    # Patch torchaudio for compatibility with newer versions (>=2.1) where
    # set_audio_backend and sox_effects were removed. s3prl still depends on both.
    import torchaudio
    if not hasattr(torchaudio, 'set_audio_backend'):
        torchaudio.set_audio_backend = lambda backend: None
    if not hasattr(torchaudio, 'sox_effects'):
        import types
        _sox_shim = types.ModuleType('torchaudio.sox_effects')
        _sox_shim.apply_effects_tensor = lambda tensor, sr, effects=None: (tensor, sr)
        _sox_shim.apply_effects_file = lambda path, effects=None: (torchaudio.load(path))
        torchaudio.sox_effects = _sox_shim
        sys.modules['torchaudio.sox_effects'] = _sox_shim

    try:
        from verification import verification as _raw_verification
    except ImportError as e:
        print(f"[SIM] Error importing verification: {e}")
        return {"error": str(e)}

    ckpt = sim_ckpt if (sim_ckpt and os.path.exists(sim_ckpt)) else None
    verify_model = None
    scores = []
    skipped = 0

    for item in tqdm(data, desc="Computing SIM"):
        wav_path = item.get(wav_key) or item.get("target_path") or item.get("wav_path")
        ref_path = item.get(ref_wav_key, "")

        if not wav_path or not os.path.exists(wav_path):
            skipped += 1
            continue
        if not ref_path or not os.path.exists(ref_path):
            skipped += 1
            continue

        try:
            sim_tensor, verify_model = _raw_verification(
                'wavlm_large', ref_path, wav_path,
                checkpoint=ckpt, model=verify_model, device=device
            )
            score = sim_tensor.item()
            item["sim"] = score
            scores.append(score)
        except Exception as e:
            print(f"[SIM] Error for {wav_path}: {e}")

    result = {
        "avg_sim": float(np.mean(scores)) if scores else 0.0,
        "count": len(scores),
        "skipped": skipped,
    }
    print(f"[SIM] Average: {result['avg_sim']:.4f} ({len(scores)} pairs, skipped: {skipped})")
    return result


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="NV-Bench: Per-sample Acoustic Fidelity (DNSMOS, SIM)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input_json", type=str, required=True,
                        help="Path to benchmark testset JSON")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for results")
    parser.add_argument("--metrics", type=str, default="dnsmos,sim",
                        help="Comma-separated metrics: dnsmos, sim")
    parser.add_argument("--wav_key", type=str, default="target_wav_path",
                        help="JSON key for generated audio path")
    # parser.add_argument("--wav_key", type=str, default="wav_path",
    #                     help="JSON key for audio path to transcribe")
    parser.add_argument("--ref_wav_key", type=str, default="ref_wav_path",
                        help="JSON key for reference audio path (used by SIM)")
    parser.add_argument("--sim_ckpt", type=str, default=SIM_CKPT_PATH,
                        help="WavLM checkpoint for SIM")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Device for computation")

    args = parser.parse_args()

    metrics_to_compute = [m.strip().lower() for m in args.metrics.split(',')]
    os.makedirs(args.output_dir, exist_ok=True)

    # Fail fast with an env-aware message if the wrong conda env is active.
    # DNSMOS lives in `nvbench-core`; SIM lives in `nvbench-acoustic`.
    from utils.envcheck import require
    if "dnsmos" in metrics_to_compute:
        require(
            step="3a · DNSMOS", env=CORE_ENV,
            modules=["onnxruntime"],
            assets=[("DNSMOS ONNX model", DNSMOS_MODEL_PATH)],
            hint="pip install -r requirements/core.txt",
        )
    if "sim" in metrics_to_compute:
        require(
            step="3a · SIM", env=ACOUSTIC_ENV,
            modules=["s3prl", "torchaudio"],
            assets=[("WavLM checkpoint", args.sim_ckpt),
                    ("seed-tts-eval repo", SEED_TTS_EVAL_DIR)],
            hint="pip install -r requirements/acoustic.txt",
        )

    # Load data
    from data.loader import load_benchmark
    data = load_benchmark(args.input_json)
    print(f"[Acoustic] Loaded {len(data)} items from {args.input_json}")

    all_results = {}

    # DNSMOS
    if "dnsmos" in metrics_to_compute:
        print("\n" + "─" * 50)
        print("Computing DNSMOS...")
        print("─" * 50)
        dnsmos_result = compute_dnsmos(data, wav_key=args.wav_key)
        all_results["dnsmos"] = dnsmos_result

    # SIM
    if "sim" in metrics_to_compute:
        print("\n" + "─" * 50)
        print("Computing Speaker Similarity...")
        print("─" * 50)
        sim_result = compute_sim(data, args.sim_ckpt, args.device, args.wav_key, args.ref_wav_key)
        all_results["sim"] = sim_result

    # Save all results
    output_path = os.path.join(args.output_dir, "acoustic_results.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print(f"\n✓ Acoustic fidelity results saved to {output_path}")


if __name__ == "__main__":
    main()
