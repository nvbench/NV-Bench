"""
NV-Bench Full Pipeline
======================
One-command orchestrator for the complete NV-Bench evaluation pipeline.

Pipeline Steps (this orchestrator runs in the `nvbench-core` env):
    1. infer    — NVASR transcription of TTS-generated speech
    2. evaluate — Instruction Alignment metrics (CER/PCER/OCER or WER/PWER/OWER)
    3. acoustic — DNSMOS (per-sample perceptual quality)

Note on environments (see README "Environments"):
    DNSMOS runs here in `nvbench-core`. SIM and FAD/FD/KL need the
    `nvbench-acoustic` env (older torch + numpy<2), so run them separately:
        conda activate nvbench-acoustic
        python compute_acoustic.py --metrics sim  ...
        python compute_fad.py                     ...

Usage:
    python run_pipeline.py \\
        --input_json /path/to/testset.json \\
        --model_dir /path/to/nvasr_model \\
        --lang zh \\
        --output_dir ./results/cosyvoice3_zh \\
        --steps infer,evaluate,acoustic

    # Skip inference if results already exist:
    python run_pipeline.py \\
        --input_json /path/to/testset.json \\
        --infer_results ./results/cosyvoice3_zh/infer_results.json \\
        --lang zh \\
        --output_dir ./results/cosyvoice3_zh \\
        --steps evaluate
"""

import os
import sys
import json
import time
import argparse

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from config import NVASR_MODEL_DIR, DEVICE, SIM_CKPT_PATH, CORE_ENV, ACOUSTIC_ENV


def run_pipeline(args):
    """Run the full NV-Bench evaluation pipeline."""

    steps = [s.strip().lower() for s in args.steps.split(',')]
    os.makedirs(args.output_dir, exist_ok=True)
    t_start = time.time()

    infer_results_path = os.path.join(args.output_dir, "infer_results.json")
    eval_results_path = os.path.join(args.output_dir, "eval_results.json")
    acoustic_results_path = os.path.join(args.output_dir, "acoustic_results.json")

    print("=" * 60)
    print("  NV-Bench Evaluation Pipeline")
    print("=" * 60)
    print(f"  Input:      {args.input_json}")
    print(f"  Output:     {args.output_dir}")
    print(f"  Language:   {args.lang}")
    print(f"  Steps:      {', '.join(steps)}")
    print("=" * 60)

    # ─── Step 1: NVASR Inference ───
    if "infer" in steps:
        print("\n" + "━" * 60)
        print("  Step 1/3: NVASR Inference")
        print("━" * 60)

        from infer import load_nvasr_model, run_inference, save_results
        from data.loader import load_benchmark

        data = load_benchmark(args.input_json)
        print(f"Loaded {len(data)} items")

        model, kwargs = load_nvasr_model(args.model_dir, args.device)
        results = run_inference(
            model, kwargs, data,
            language=args.lang,
            wav_key=args.wav_key,
            text_key=args.text_key,
        )
        save_results(results, infer_results_path)

        # Use existing infer_results
        if args.infer_results:
            infer_results_path = args.infer_results
    elif args.infer_results:
        infer_results_path = args.infer_results
        print(f"\n[Pipeline] Using existing inference results: {infer_results_path}")

    # ─── Step 2: Instruction Alignment Evaluation ───
    if "evaluate" in steps:
        print("\n" + "━" * 60)
        print("  Step 2/3: Instruction Alignment Evaluation")
        print("━" * 60)

        if not os.path.exists(infer_results_path):
            print(f"[ERROR] Inference results not found: {infer_results_path}")
            print("Run with --steps infer first, or provide --infer_results")
            return

        from evaluate import evaluate
        evaluate(
            test_path=infer_results_path,
            output_path=eval_results_path,
            lang=args.lang,
            sim_ckpt=args.sim_ckpt,
            sim_device=args.sim_device,
        )

    # ─── Step 3: Acoustic Fidelity Evaluation ───
    if "acoustic" in steps:
        print("\n" + "━" * 60)
        print("  Step 3/3: Acoustic Fidelity Evaluation")
        print("━" * 60)

        from compute_acoustic import compute_dnsmos, compute_sim
        from data.loader import load_benchmark

        data = load_benchmark(args.input_json)
        all_results = {}

        # DNSMOS
        if "dnsmos" in args.acoustic_metrics:
            print("\nComputing DNSMOS...")
            dnsmos_result = compute_dnsmos(data, wav_key=args.wav_key)
            all_results["dnsmos"] = dnsmos_result

        # SIM
        if "sim" in args.acoustic_metrics:
            print("\nComputing Speaker Similarity...")
            sim_result = compute_sim(
                data, args.sim_ckpt, args.sim_device, args.wav_key, args.ref_wav_key
            )
            all_results["sim"] = sim_result

        with open(acoustic_results_path, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        print(f"\n✓ Acoustic results saved to {acoustic_results_path}")
        print("\n  Note: For FAD/FD/KL, run compute_fad.py separately.")

    # ─── Summary ───
    elapsed = time.time() - t_start
    print("\n" + "=" * 60)
    print(f"  Pipeline Complete — {elapsed:.1f}s total")
    print("=" * 60)

    output_files = []
    for path in [infer_results_path, eval_results_path, acoustic_results_path]:
        if os.path.exists(path):
            output_files.append(path)
    print(f"  Output files:")
    for f in output_files:
        print(f"    → {f}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="NV-Bench Full Evaluation Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Core arguments
    parser.add_argument("--input_json", type=str, required=True,
                        help="Path to benchmark testset JSON")
    parser.add_argument("--output_dir", type=str, default="./results",
                        help="Output directory for all results")
    parser.add_argument("--lang", type=str, required=True, choices=["zh", "en"],
                        help="Language: 'zh' or 'en'")
    parser.add_argument("--steps", type=str, default="infer,evaluate,acoustic",
                        help="Pipeline steps to run (comma-separated): infer, evaluate, acoustic")

    # Inference arguments
    parser.add_argument("--model_dir", type=str, default=NVASR_MODEL_DIR,
                        help="NVASR model directory")
    parser.add_argument("--device", type=str, default=DEVICE,
                        help="Device for inference and acoustic computation")
    parser.add_argument("--wav_key", type=str, default="target_wav_path",
                        help="JSON key for generated audio path")
    parser.add_argument("--text_key", type=str, default="text",
                        help="JSON key for ground-truth text")
    parser.add_argument("--infer_results", type=str, default=None,
                        help="Path to pre-existing inference results (skip infer step)")

    # Evaluation arguments
    parser.add_argument("--ref_wav_key", type=str, default="ref_wav_path",
                        help="JSON key for reference audio path (used by SIM)")
    parser.add_argument("--sim_ckpt", type=str, default=SIM_CKPT_PATH,
                        help="WavLM checkpoint for SIM")
    parser.add_argument("--sim_device", type=str, default="cuda:0",
                        help="Device for SIM model")

    # Acoustic arguments
    parser.add_argument("--acoustic_metrics", type=str, default="dnsmos",
                        help="Acoustic metrics (comma-separated): dnsmos, sim")

    args = parser.parse_args()
    args.acoustic_metrics = [m.strip().lower() for m in args.acoustic_metrics.split(',')]

    # This orchestrator runs in the core env (infer/evaluate/DNSMOS).
    steps = [s.strip().lower() for s in args.steps.split(',')]
    if "infer" in steps:
        from utils.envcheck import require
        require(
            step="1 · NVASR inference", env=CORE_ENV,
            modules=["funasr", "torch"],
            assets=[("NVASR model dir", args.model_dir)],
            hint="pip install -r requirements/core.txt",
        )
    if "sim" in args.acoustic_metrics:
        print(f"[Pipeline] WARNING: SIM needs the `{ACOUSTIC_ENV}` env and will likely be skipped here.")
        print(f"[Pipeline]          Run SIM separately: conda activate {ACOUSTIC_ENV}; "
              "python compute_acoustic.py --metrics sim ...")

    run_pipeline(args)


if __name__ == "__main__":
    main()
