"""
NV-Bench FAD / FD / KL Divergence Computation
===============================================
Standalone script for computing distribution-level acoustic metrics.

All subsets (zh + en, SFT + zero-shot) are pooled into a single evaluation
because FAD/FD/KL are distribution-level metrics and need sufficient data
for stable estimates.

This script:
  1. Takes one or more (gt_json, gen_json) pairs.
  2. Symlinks all ground-truth and generated audio into shared temp directories.
  3. Runs `audioldm_eval` once over the entire pooled dataset.
  4. Saves the results to a JSON file.

Usage:
    # Single subset
    python compute_fad.py \\
        --pairs gt1.json:gen1.json \\
        --output_json results/fad_results.json

    # Multiple subsets pooled together (recommended)
    python compute_fad.py \\
        --pairs gt_zh.json:gen_zh.json gt_en.json:gen_en.json \\
        --output_json results/fad_results.json

    # Custom keys and device
    python compute_fad.py \\
        --pairs gt.json:gen.json \\
        --gt_wav_key wav_path \\
        --gen_wav_key target_wav_path \\
        --device cuda:1 \\
        --sample_rate 16000 \\
        --output_json results/fad_results.json
"""

import os
import sys
import json
import glob
import shutil
import argparse
import traceback

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from config import AUDIOLDM_EVAL_BACKBONE, AUDIOLDM_EVAL_SAMPLE_RATE

# Fix PyTorch 2.6 compatibility: default weights_only changed to True,
# which breaks audioldm_eval's checkpoint loading (uses numpy globals).
import torch
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load


def main():
    parser = argparse.ArgumentParser(
        description="NV-Bench: Compute FAD / FD / KL divergence across all subsets",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--pairs", type=str, nargs="+", required=True,
        help=(
            "One or more gt:gen JSON pairs, separated by colon. "
            "e.g. gt_zh.json:gen_zh.json gt_en.json:gen_en.json"
        ),
    )
    parser.add_argument("--gt_wav_key", type=str, default="wav_path",
                        help="Key for ground-truth wav path in gt JSON")
    parser.add_argument("--gen_wav_key", type=str, default="target_wav_path",
                        help="Key for generated wav path in gen JSON")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Device for evaluation")
    parser.add_argument("--sample_rate", type=int, default=AUDIOLDM_EVAL_SAMPLE_RATE,
                        help="Sample rate for audioldm_eval")
    parser.add_argument("--backbone", type=str, default=AUDIOLDM_EVAL_BACKBONE,
                        help="Backbone model for audioldm_eval")
    parser.add_argument("--num_workers", type=int, default=2,
                        help="Number of parallel workers for EvaluationHelperParallel")
    parser.add_argument("--output_json", type=str, default="./results/fad_results.json",
                        help="Path to save metric results")

    args = parser.parse_args()

    # ─── Parse pairs ───
    pair_list = []
    for pair_str in args.pairs:
        parts = pair_str.split(":")
        if len(parts) != 2:
            print(f"[ERROR] Invalid pair format: '{pair_str}'. Expected gt.json:gen.json")
            sys.exit(1)
        gt_path, gen_path = parts
        if not os.path.exists(gt_path):
            print(f"[ERROR] Ground-truth JSON not found: {gt_path}")
            sys.exit(1)
        if not os.path.exists(gen_path):
            print(f"[ERROR] Generated JSON not found: {gen_path}")
            sys.exit(1)
        pair_list.append({"gt_json": gt_path, "gen_json": gen_path})

    print("=" * 60)
    print("  NV-Bench FAD / FD / KL Computation")
    print("=" * 60)
    print(f"  Pairs:       {len(pair_list)}")
    for i, p in enumerate(pair_list):
        print(f"    [{i}] GT:  {p['gt_json']}")
        print(f"        GEN: {p['gen_json']}")
    print(f"  Device:      {args.device}")
    print(f"  Sample rate: {args.sample_rate}")
    print(f"  Backbone:    {args.backbone}")
    print("=" * 60)

    # ─── Create temp directories ───
    temp_gt_dir = "./temp_gt_aligned"
    temp_gen_dir = "./temp_gen_aligned"

    for d in [temp_gt_dir, temp_gen_dir]:
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d, exist_ok=True)

    # Remove stale cache files
    for pattern in ["*_fad_feature_cache.npy", "*classifier_logits_feature_cache.pkl"]:
        for f in glob.glob(pattern):
            if "temp_gt_aligned" in f or "temp_gen_aligned" in f:
                try:
                    os.remove(f)
                    print(f"Removed cache file: {f}")
                except OSError:
                    pass

    # ─── Symlink all subsets into shared temp dirs ───
    print("\nPre-processing files...")
    total_files = 0
    skipped = 0

    for pi, pair in enumerate(pair_list):
        gt_json_path = pair["gt_json"]
        gen_json_path = pair["gen_json"]

        print(f"\n[Pair {pi}] GT: {gt_json_path}")
        print(f"[Pair {pi}] GEN: {gen_json_path}")

        with open(gt_json_path, 'r', encoding='utf-8') as f:
            gt_data = json.load(f)
        with open(gen_json_path, 'r', encoding='utf-8') as f:
            gen_data = json.load(f)

        n_items = min(len(gt_data), len(gen_data))
        if len(gt_data) != len(gen_data):
            print(f"  Warning: GT has {len(gt_data)} items, GEN has {len(gen_data)} items. Using {n_items}.")

        pair_count = 0
        for idx in range(n_items):
            gt_item = gt_data[idx]
            gen_item = gen_data[idx]

            # Ground-truth audio path
            target_path = gt_item.get(args.gt_wav_key, '')
            # Generated audio path
            gen_path = gen_item.get(args.gen_wav_key) or gen_item.get("target_path", '')

            if not target_path or not gen_path:
                reason = []
                if not target_path: reason.append("gt_path empty")
                if not gen_path: reason.append("gen_path empty")
                print(f"  [SKIP] Pair {pi}, idx {idx}: {', '.join(reason)}")
                skipped += 1
                continue
            if not os.path.exists(target_path):
                print(f"  [SKIP] Pair {pi}, idx {idx}: GT file not found: {target_path}")
                skipped += 1
                continue
            if not os.path.exists(gen_path):
                print(f"  [SKIP] Pair {pi}, idx {idx}: GEN file not found: {gen_path}")
                skipped += 1
                continue

            ext = os.path.splitext(gen_path)[1] or ".wav"
            unique_name = f"p{pi}_{idx}{ext}"

            os.symlink(target_path, os.path.join(temp_gt_dir, unique_name))
            os.symlink(gen_path, os.path.join(temp_gen_dir, unique_name))
            pair_count += 1
            total_files += 1

        print(f"  Aligned {pair_count} file pairs from this subset")

    print(f"\nTotal aligned files: {total_files} (skipped: {skipped})")

    if total_files == 0:
        print("[ERROR] No files found to align. Exiting.")
        return

    # ─── Run Evaluation ───
    print("\nStarting evaluation...")

    try:
        from audioldm_eval import EvaluationHelperParallel
        evaluator = EvaluationHelperParallel(
            args.sample_rate, args.num_workers, backbone=args.backbone
        )

        # evaluator.main() uses mp.spawn internally and returns None;
        # metrics are saved to a JSON file by the library itself.
        evaluator.main(
            temp_gen_dir,
            temp_gt_dir,
        )

        # Read metrics from the JSON file saved by audioldm_eval
        # (pattern: <timestamp>_temp_gen_aligned.json in cwd)
        metric_files = sorted(glob.glob("./*_temp_gen_aligned.json"), key=os.path.getmtime)
        if metric_files:
            latest_metric_file = metric_files[-1]
            with open(latest_metric_file, 'r') as f:
                metrics = json.load(f)

            print("\n" + "=" * 60)
            print("  FAD / FD / KL Results")
            print("=" * 60)
            for k, v in metrics.items():
                print(f"  {k}: {v}")
            print("=" * 60)

            # Save results
            os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
            with open(args.output_json, 'w', encoding='utf-8') as f:
                json.dump(metrics, f, indent=2, ensure_ascii=False)
            print(f"\n✓ Metrics saved to {args.output_json}")

            # Clean up the intermediate metric file
            os.remove(latest_metric_file)
        else:
            print("[ERROR] No metric JSON file found from audioldm_eval.")

    except ImportError:
        print("[ERROR] audioldm_eval not installed. Install with: pip install audioldm_eval")
    except Exception as e:
        print(f"[ERROR] Evaluation failed: {e}")
        traceback.print_exc()
    finally:
        print("\nCleaning up temporary directories...")
        for d in [temp_gt_dir, temp_gen_dir]:
            if os.path.exists(d):
                shutil.rmtree(d)


if __name__ == "__main__":
    main()
