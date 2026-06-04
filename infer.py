"""
NV-Bench NVASR Inference Pipeline
==================================
Step 1: Run NVASR transcription on TTS-generated speech.

Reads benchmark JSON → runs NVASR model → outputs structured JSON results.

Usage:
    python infer.py --input_json /path/to/testset.json --output_dir ./results --lang auto
"""

import os
import sys
import json
import argparse
import time
from typing import List, Dict

# Setup project paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from config import NVASR_MODEL_DIR, DEVICE, OUTPUT_DIR, CORE_ENV


def load_nvasr_model(model_dir: str, device: str):
    """Load the SenseVoiceSmall NVASR model.

    Args:
        model_dir: Path to NVASR model directory.
        device: Device string, e.g. 'cuda:0'.

    Returns:
        Tuple of (model, inference_kwargs).
    """
    from model import SenseVoiceSmall
    model, kwargs = SenseVoiceSmall.from_pretrained(model=model_dir, device=device)
    model.eval()
    print(f"[NVASR] Model loaded from {model_dir} on {device}")
    return model, kwargs


def run_inference(
    model,
    kwargs: dict,
    data: List[Dict],
    language: str = "auto",
    wav_key: str = "target_wav_path",
    text_key: str = "text",
) -> List[Dict]:
    """Run NVASR inference on a list of audio items.

    Args:
        model: Loaded SenseVoiceSmall model.
        kwargs: Inference kwargs from model loading.
        data: List of data dicts from benchmark JSON.
        language: Language code ('auto', 'zh', 'en', etc.).
        wav_key: Key to use for audio file path.
        text_key: Key to use for ground-truth text.

    Returns:
        List of result dicts with fields:
            - wav_path: path to the evaluated audio
            - target: ground-truth text
            - text: NVASR-predicted text
            - timestamp: CTC-aligned timestamps
    """
    from utils.postprocess import rich_transcription_postprocess

    results = []
    total = len(data)
    success = 0
    errors = 0

    for idx, item in enumerate(data):
        wav_path = item.get(wav_key, '') or item.get('target_path', '') or item.get('wav_path', '')
        gt_text = item.get(text_key, '')

        if not wav_path:
            print(f"[{idx+1}/{total}] Skipping: no wav_path found")
            errors += 1
            continue

        if not os.path.exists(wav_path):
            print(f"[{idx+1}/{total}] Skipping: file not found: {wav_path}")
            errors += 1
            continue

        try:
            res = model.inference(
                data_in=wav_path,
                language=language,
                use_itn=False,
                ban_emo_unk=False,
                output_timestamp=True,
                **kwargs,
            )

            timestamp = res[0][0].get("timestamp", [])
            text = rich_transcription_postprocess(res[0][0]["text"])

            result = {
                "wav_path": wav_path,
                "target": gt_text,
                "text": text,
                "timestamp": timestamp,
            }
            results.append(result)
            success += 1

            if (idx + 1) % 50 == 0 or idx == 0:
                print(f"[{idx+1}/{total}] {os.path.basename(wav_path)}: {text[:60]}...")

        except Exception as e:
            print(f"[{idx+1}/{total}] Error processing {wav_path}: {e}")
            errors += 1

    print(f"\n[NVASR] Inference complete: {success} success, {errors} errors out of {total} items")
    return results


def save_results(results: List[Dict], output_path: str):
    """Save inference results to JSON.

    Args:
        results: List of result dicts.
        output_path: Path for output JSON file.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"[NVASR] Results saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="NV-Bench Step 1: NVASR Inference",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input_json", type=str, required=True,
                        help="Path to benchmark testset JSON file")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for results (default: <NVBENCH_OUTPUT_DIR>/<input_name>)")
    parser.add_argument("--model_dir", type=str, default=NVASR_MODEL_DIR,
                        help="Path to NVASR model directory (default: config.NVASR_MODEL_DIR)")
    parser.add_argument("--device", type=str, default=DEVICE,
                        help="Device for inference")
    parser.add_argument("--lang", type=str, default="auto",
                        choices=["auto", "zh", "en", "yue", "ja", "ko"],
                        help="Language for NVASR")
    parser.add_argument("--wav_key", type=str, default="target_wav_path",
                        help="JSON key for audio path to transcribe")
    # parser.add_argument("--wav_key", type=str, default="wav_path",
    #                     help="JSON key for audio path to transcribe")
    parser.add_argument("--text_key", type=str, default="text",
                        help="JSON key for ground-truth text")

    args = parser.parse_args()

    # Fail fast if the wrong env is active or the NVASR model is missing.
    from utils.envcheck import require
    require(
        step="1 · NVASR inference", env=CORE_ENV,
        modules=["funasr", "torch"],
        assets=[("NVASR model dir", args.model_dir)],
        hint="pip install -r requirements/core.txt  +  download Multilingual-NVASR (see README)",
    )

    # Set CUDA device
    if args.device.startswith("cuda"):
        gpu_id = args.device.split(":")[-1] if ":" in args.device else "0"
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id

    # Determine output path (default: <NVBENCH_OUTPUT_DIR>/<input_name>)
    input_name = os.path.splitext(os.path.basename(args.input_json))[0]
    if args.output_dir is None:
        args.output_dir = os.path.join(OUTPUT_DIR, input_name)
    output_path = os.path.join(args.output_dir, "infer_results.json")

    # Load data
    from data.loader import load_benchmark
    data = load_benchmark(args.input_json)
    print(f"[NVASR] Loaded {len(data)} items from {args.input_json}")

    # Load model
    model, kwargs = load_nvasr_model(args.model_dir, args.device)

    # Run inference
    t0 = time.time()
    results = run_inference(
        model, kwargs, data,
        language=args.lang,
        wav_key=args.wav_key,
        text_key=args.text_key,
    )
    elapsed = time.time() - t0
    print(f"[NVASR] Total inference time: {elapsed:.1f}s ({elapsed/max(len(results),1):.2f}s/item)")

    # Save results
    save_results(results, output_path)
    print(f"\n✓ Inference complete. Results: {output_path}")
    print(f"  Next step: python evaluate.py --input {output_path} --lang {args.lang}")


if __name__ == "__main__":
    main()
