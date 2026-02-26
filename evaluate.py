"""
NV-Bench Instruction Alignment Evaluation
==========================================
Step 2: Evaluate NVASR transcription results against ground truth.

Supports both Chinese (CER-based) and English (WER-based) evaluation.

Metrics computed:
    - CER / WER — Standard text error rate (OCER / OWER)
    - PCER / PWER — Paralinguistic error rate (NVV tags only)
    - CER_text / WER_text — Text-only error rate (tags removed)
    - Speaker Similarity (SIM) — WavLM cosine similarity (optional)

Usage:
    python evaluate.py --input results/infer_results.json --lang zh --output results/eval_results.json
"""

import os
import sys
import json
import re
import argparse
import numpy as np
from typing import List, Dict, Tuple, Set, Optional

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from config import VOCAB_ZH, VOCAB_EN, PARA_GROUP_A, SIM_CKPT_PATH, SEED_TTS_EVAL_DIR

# ─────────────────────────────────────────────
# Speaker Similarity (SIM) — lazy import
# ─────────────────────────────────────────────
SIM_AVAILABLE = False
_raw_verification = None

def _setup_sim():
    global SIM_AVAILABLE, _raw_verification
    if SIM_AVAILABLE:
        return True
    try:
        sys.path.append(SEED_TTS_EVAL_DIR)
        sys.path.append(os.path.join(SEED_TTS_EVAL_DIR, 'thirdparty/UniSpeech/downstreams/speaker_verification'))
        from verification import verification as rv
        _raw_verification = rv
        SIM_AVAILABLE = True
        return True
    except ImportError:
        print("[Warning] 'verification' module not found. SIM computation disabled.")
        return False


# ─────────────────────────────────────────────
# CER accuracy utils — fallback if not found
# ─────────────────────────────────────────────
try:
    from utils.cer_accuracy import extract_paralingustic, normalize_text
except ImportError:
    def normalize_text(text: str) -> str:
        return text.strip().lower()

    def extract_paralingustic(text: str) -> List[str]:
        return re.findall(r"\[[^\]]+\]", text)


# ─────────────────────────────────────────────
# Regex Patterns
# ─────────────────────────────────────────────
_PUNC_PATTERN_ZH = re.compile(r"[\u3000-\u303f\uff00-\uffef]|[%/.,。，!！?？;…——·~•\u201c\u201d「」、：（）～《》『』；]")
TAG_SPLIT_RE = re.compile(r"(\[[^\[\]]+\])")
EN_RUN_RE = re.compile(r"[A-Za-z0-9%/.,;:…—·~\"'!\-]+")
CJK_CHAR_RE = re.compile(r"[\u4e00-\u9fff]")
PUNCS_EN = "，。！？；：\u201c\u201d''（）【】《》…,.?!;:'\"()[]{}-「」"


# ─────────────────────────────────────────────
# Tokenization
# ─────────────────────────────────────────────
def _remove_punc_zh(text: str) -> str:
    return re.sub(_PUNC_PATTERN_ZH, "", text)

def _replace_punc_en(s: str) -> str:
    if not s:
        return s
    for ch in PUNCS_EN:
        if ch == "'":
            continue
        s = s.replace(ch, " ")
    return s


def _is_tag(tok: str) -> bool:
    return bool(tok) and tok.startswith('[') and tok.endswith(']')


def tokenize_zh(text: str, remove_punc: bool = True) -> List[str]:
    """Tokenize Chinese text: CJK chars as single tokens, keep [Tags] intact."""
    if not text:
        return []
    parts = TAG_SPLIT_RE.split(text)
    tokens = []
    for part in parts:
        if not part:
            continue
        if TAG_SPLIT_RE.fullmatch(part):
            tokens.append(part)
            continue
        seg = _remove_punc_zh(part) if remove_punc else part
        seg = seg.lower()
        i = 0
        while i < len(seg):
            ch = seg[i]
            if ch.isspace():
                i += 1
                continue
            m = EN_RUN_RE.match(seg, i)
            if m:
                tokens.append(m.group())
                i = m.end()
                continue
            if CJK_CHAR_RE.fullmatch(ch):
                tokens.append(ch)
                i += 1
                continue
            tokens.append(ch)
            i += 1
    return tokens


def tokenize_en(text: str, remove_punc: bool = True) -> List[str]:
    """Tokenize English text: words split by whitespace, keep [Tags] intact."""
    if not text:
        return []
    parts = re.split(r'(\[.*?\])', text)
    tokens = []
    for part in parts:
        if not part:
            continue
        if re.fullmatch(r'\[.*?\]', part):
            tokens.append(part)
        else:
            part_low = part.lower()
            if remove_punc:
                part_low = _replace_punc_en(part_low)
            tokens.extend(part_low.split())
    return tokens


def filter_vocab_tags(tokens: List[str], vocab_set: Set[str]) -> List[str]:
    """Remove unknown tags; keep valid tags and normal words."""
    return [t for t in tokens if not _is_tag(t) or t in vocab_set]


def remove_all_tags(tokens: List[str]) -> List[str]:
    return [t for t in tokens if not _is_tag(t)]


def keep_only_group_tags(tokens: List[str], group: List[str], vocab_all: List[str]) -> List[str]:
    group_set = set(group)
    vocab_set = set(vocab_all)
    out = []
    for t in tokens:
        if t in vocab_set:
            if t in group_set:
                out.append(t)
        else:
            out.append(t)
    return out


# ─────────────────────────────────────────────
# Edit Distance / Error Rate
# ─────────────────────────────────────────────
def compute_err_rate(ref: List[str], hyp: List[str], norm_n: int) -> Tuple[float, float, float, float]:
    """Compute error rate with backtracking for S/D/I decomposition."""
    n_ref, m = len(ref), len(hyp)
    if norm_n == 0:
        return (0.0, 0.0, 0.0, 0.0) if n_ref == 0 and m == 0 else (1.0, 0.0, 0.0, 1.0)

    dp = [[0] * (m + 1) for _ in range(n_ref + 1)]
    bt = [[None] * (m + 1) for _ in range(n_ref + 1)]

    for i in range(1, n_ref + 1):
        dp[i][0] = i
        bt[i][0] = 'del'
    for j in range(1, m + 1):
        dp[0][j] = j
        bt[0][j] = 'ins'

    for i in range(1, n_ref + 1):
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
                bt[i][j] = 'ok'
            else:
                sub = dp[i - 1][j - 1] + 1
                dele = dp[i - 1][j] + 1
                inse = dp[i][j - 1] + 1
                best = min(sub, dele, inse)
                dp[i][j] = best
                if best == sub:
                    bt[i][j] = 'sub'
                elif best == dele:
                    bt[i][j] = 'del'
                else:
                    bt[i][j] = 'ins'

    i, j = n_ref, m
    S = D = I = 0
    while i > 0 or j > 0:
        op = bt[i][j] if bt[i][j] else ('del' if i > 0 else 'ins')
        if op == 'ok':
            i -= 1; j -= 1
        elif op == 'sub':
            S += 1; i -= 1; j -= 1
        elif op == 'del':
            D += 1; i -= 1
        else:
            I += 1; j -= 1

    return (S + D + I) / norm_n, S / norm_n, D / norm_n, I / norm_n


def compute_err_counts(ref: List[str], hyp: List[str]) -> Tuple[int, int, int, int]:
    """Compute raw S, D, I counts and reference length N."""
    n_ref, m = len(ref), len(hyp)
    dp = [[0] * (m + 1) for _ in range(n_ref + 1)]
    bt = [[None] * (m + 1) for _ in range(n_ref + 1)]

    for i in range(1, n_ref + 1):
        dp[i][0] = i; bt[i][0] = 'del'
    for j in range(1, m + 1):
        dp[0][j] = j; bt[0][j] = 'ins'

    for i in range(1, n_ref + 1):
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]; bt[i][j] = 'ok'
            else:
                sub = dp[i - 1][j - 1] + 1
                dele = dp[i - 1][j] + 1
                inse = dp[i][j - 1] + 1
                best = min(sub, dele, inse)
                dp[i][j] = best
                if best == sub: bt[i][j] = 'sub'
                elif best == dele: bt[i][j] = 'del'
                else: bt[i][j] = 'ins'

    i, j = n_ref, m
    S = D = I = 0
    while i > 0 or j > 0:
        op = bt[i][j] if bt[i][j] else ('del' if i > 0 else 'ins')
        if op == 'ok': i -= 1; j -= 1
        elif op == 'sub': S += 1; i -= 1; j -= 1
        elif op == 'del': D += 1; i -= 1
        else: I += 1; j -= 1

    return S, D, I, n_ref


# ─────────────────────────────────────────────
# Speaker Similarity
# ─────────────────────────────────────────────
def compute_similarity(wav1, wav2, verify_model=None, ckpt_path=None, device='cuda:0'):
    sim_tensor, verify_model = _raw_verification(
        'wavlm_large', wav1, wav2,
        checkpoint=ckpt_path, model=verify_model, device=device
    )
    return sim_tensor.item(), verify_model


# ─────────────────────────────────────────────
# Main Evaluation
# ─────────────────────────────────────────────
def evaluate(
    test_path: str,
    output_path: str,
    lang: str = "zh",
    sim_ckpt: Optional[str] = None,
    sim_device: str = "cuda:0",
    ref_wav_key: str = "ref_wav_path",
) -> Dict:
    """Run Instruction Alignment evaluation.

    Args:
        test_path: Path to inference results (JSON or TXT).
        output_path: Path to save evaluation results JSON.
        lang: Language ('zh' or 'en').
        sim_ckpt: Path to WavLM checkpoint.
        sim_device: Device for SIM model.
        ref_wav_key: Key for reference wav path in data items.

    Returns:
        Summary dict of all metrics.
    """
    from data.loader import load_infer_results

    is_english = (lang == "en")
    vocab = VOCAB_EN if is_english else VOCAB_ZH
    vocab_set = set(vocab)
    para_group_b = [t for t in vocab if t not in PARA_GROUP_A]
    tokenize_fn = tokenize_en if is_english else tokenize_zh

    test_data = load_infer_results(test_path)
    print(f"[Evaluate] Loaded {len(test_data)} items from {test_path}")
    print(f"[Evaluate] Language: {'English (WER)' if is_english else 'Chinese (CER)'}")

    # --- SIM Setup ---
    do_sim = _setup_sim()
    verify_model = None
    ckpt = sim_ckpt if (sim_ckpt and os.path.exists(sim_ckpt)) else None
    if do_sim:
        print(f"[SIM] SIM enabled, reading ref wav from '{ref_wav_key}' field")
    else:
        print("[SIM] verification module not available. Skipping SIM.")

    # Metric storage
    ocer_list = []          # OCER/OWER: CER/WER with all tags (no punc)
    cer_text_list = []      # CER/WER text-only (tags removed)
    ocer_A_list = []        # OCER/OWER with Group A tags only
    ocer_B_list = []        # OCER/OWER with Group B tags only

    pcer_macro_list = []
    para_S_total = para_D_total = para_I_total = para_N_total = 0

    sim_list = []
    valid_utt_cnt = 0

    # Per-utterance output
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    with open(output_path.replace('.json', '_detail.txt'), 'w', encoding='utf-8') as fout:
        for audio_info in test_data:
            valid_utt_cnt += 1
            wav_path = audio_info.get('wav_path', '')
            raw_text = audio_info.get('text', '')
            raw_target = audio_info.get('target', '')

            text = normalize_text(raw_text) if is_english else raw_text
            target = normalize_text(raw_target) if is_english else raw_target

            fout.write(f"wav_path: {wav_path}\n")
            fout.write(f"target: {target}\ntext: {text}\n")

            # 1. Paralinguistic extraction & PCER
            gt_para = extract_paralingustic(target)
            pred_para = extract_paralingustic(text)

            S_p, D_p, I_p, N_p = compute_err_counts(gt_para, pred_para)
            para_S_total += S_p
            para_D_total += D_p
            para_I_total += I_p
            para_N_total += N_p

            if N_p > 0:
                pcer_val = (S_p + D_p + I_p) / N_p * 100
                pcer_macro_list.append(pcer_val)
                fout.write(f"PCER(utt): {pcer_val:.2f}% (N={N_p})\n")

            # 2. OCER (overall, no punc)
            target_tok = tokenize_fn(target, remove_punc=True)
            text_tok = tokenize_fn(text, remove_punc=True)
            target_tok = filter_vocab_tags(target_tok, vocab_set)
            text_tok = filter_vocab_tags(text_tok, vocab_set)
            N_unified = len(target_tok)

            ocer, _, _, _ = compute_err_rate(target_tok, text_tok, N_unified)
            ocer_list.append(ocer * 100)

            # Group A
            t_A = keep_only_group_tags(target_tok, PARA_GROUP_A, vocab)
            p_A = keep_only_group_tags(text_tok, PARA_GROUP_A, vocab)
            c_A, _, _, _ = compute_err_rate(t_A, p_A, N_unified)
            ocer_A_list.append(c_A * 100)

            # Group B
            t_B = keep_only_group_tags(target_tok, para_group_b, vocab)
            p_B = keep_only_group_tags(text_tok, para_group_b, vocab)
            c_B, _, _, _ = compute_err_rate(t_B, p_B, N_unified)
            ocer_B_list.append(c_B * 100)

            # Text-only CER/WER (tags removed)
            t_text = remove_all_tags(target_tok)
            p_text = remove_all_tags(text_tok)
            c_text, _, _, _ = compute_err_rate(t_text, p_text, N_unified)
            cer_text_list.append(c_text * 100)

            fout.write(f"OCER: {ocer * 100:.2f}%\n")

            # 3. Speaker Similarity (SIM)
            if do_sim:
                ref_wav = audio_info.get(ref_wav_key, '')
                if ref_wav and os.path.exists(ref_wav) and wav_path and os.path.exists(wav_path):
                    try:
                        sim_score, verify_model = compute_similarity(
                            ref_wav, wav_path,
                            verify_model=verify_model,
                            ckpt_path=ckpt,
                            device=sim_device,
                        )
                        sim_list.append(sim_score)
                        fout.write(f"SIM: {sim_score:.4f}\n")
                    except Exception as e:
                        fout.write(f"SIM: Error ({e})\n")

            fout.write("\n")

    # ─── Summary ───
    metric_prefix = "WER" if is_english else "CER"
    pcer_prefix = "PWER" if is_english else "PCER"

    pcer_macro_avg = float(np.mean(pcer_macro_list)) if pcer_macro_list else 0.0
    pcer_micro_avg = ((para_S_total + para_D_total + para_I_total) / para_N_total * 100) if para_N_total > 0 else 0.0

    summary = {
        f"O{metric_prefix}": float(np.mean(ocer_list)) if ocer_list else 0.0,
        f"O{metric_prefix}_A (Vegetative)": float(np.mean(ocer_A_list)) if ocer_A_list else 0.0,
        f"O{metric_prefix}_B (Conversational)": float(np.mean(ocer_B_list)) if ocer_B_list else 0.0,
        f"{metric_prefix} (text only)": float(np.mean(cer_text_list)) if cer_text_list else 0.0,
        f"{pcer_prefix}_Macro": pcer_macro_avg,
        f"{pcer_prefix}_Micro": pcer_micro_avg,
        "num_utterances": valid_utt_cnt,
    }

    if do_sim and sim_list:
        summary["Speaker_SIM_Avg"] = float(np.mean(sim_list))
        summary["Speaker_SIM_Count"] = len(sim_list)

    # Print summary
    print("\n" + "=" * 55)
    print(f"  NV-Bench Instruction Alignment — {lang.upper()}")
    print("=" * 55)
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"  {k:<35}: {v:.2f}{'%' if 'SIM' not in k else ''}")
        else:
            print(f"  {k:<35}: {v}")
    print("=" * 55)

    # Save JSON summary
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n✓ Evaluation results saved to {output_path}")

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="NV-Bench Step 2: Instruction Alignment Evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", type=str, required=True,
                        help="Path to inference results (JSON or TXT)")
    parser.add_argument("--output", type=str, default=None,
                        help="Path for evaluation results JSON")
    parser.add_argument("--lang", type=str, required=True, choices=["zh", "en"],
                        help="Language: 'zh' for CER-based, 'en' for WER-based")
    parser.add_argument("--ref_wav_key", type=str, default="ref_wav_path",
                        help="JSON key for reference audio path (for SIM)")
    parser.add_argument("--sim_ckpt", type=str, default=SIM_CKPT_PATH,
                        help="WavLM checkpoint for SIM")
    parser.add_argument("--sim_device", type=str, default="cuda:0",
                        help="Device for SIM model")

    args = parser.parse_args()

    if args.output is None:
        base = os.path.splitext(args.input)[0]
        args.output = f"{base}_eval.json"

    evaluate(
        test_path=args.input,
        output_path=args.output,
        lang=args.lang,
        sim_ckpt=args.sim_ckpt,
        sim_device=args.sim_device,
        ref_wav_key=args.ref_wav_key,
    )


if __name__ == "__main__":
    main()
