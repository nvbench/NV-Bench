"""
NV-Bench environment doctor
============================
Reports whether the *currently active* conda env can run a given role, and
which assets are present. Safe to run anytime; never mutates anything.

    python scripts/check_env.py core       # Steps 1, 2, 3a-DNSMOS
    python scripts/check_env.py acoustic   # Steps 3a-SIM, 3b-FAD
    python scripts/check_env.py            # both roles

Exit code is non-zero if the requested role has any missing requirement, so it
can gate CI / setup.sh.
"""

import importlib
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import config as c  # noqa: E402


def _check(label, ok):
    print(f"  {'OK ' if ok else '!! '} {label}")
    return ok


def _mod(name):
    try:
        importlib.import_module(name)
        return True
    except Exception:
        return False


def check_core():
    print("[core]  Steps 1 (infer) · 2 (evaluate) · 3a-DNSMOS")
    good = True
    for m in ["torch", "funasr", "onnxruntime", "numpy"]:
        good &= _check(f"module {m}", _mod(m))
    good &= _check(f"DNSMOS onnx  {c.DNSMOS_MODEL_PATH}", os.path.exists(c.DNSMOS_MODEL_PATH))
    good &= _check(f"NVASR model  {c.NVASR_MODEL_DIR}", os.path.isdir(c.NVASR_MODEL_DIR))
    return good


def check_acoustic():
    print("[acoustic]  Steps 3a-SIM · 3b-FAD/FD/KL")
    good = True
    for m in ["torch", "torchaudio", "s3prl", "audioldm_eval"]:
        good &= _check(f"module {m}", _mod(m))
    import numpy as np
    good &= _check(f"numpy<2 (got {np.__version__})", int(np.__version__.split('.')[0]) < 2)
    good &= _check(f"WavLM ckpt   {c.SIM_CKPT_PATH}", os.path.exists(c.SIM_CKPT_PATH))
    good &= _check(f"seed-tts-eval {c.SEED_TTS_EVAL_DIR}", os.path.isdir(c.SEED_TTS_EVAL_DIR))
    for p in [c.FAD_CNN14_16K, c.FAD_CNN14]:
        good &= _check(f"FAD ckpt {p}", os.path.exists(p))
    return good


def main():
    role = sys.argv[1] if len(sys.argv) > 1 else "both"
    print(f"interpreter: {sys.executable}\n")
    ok = True
    if role in ("core", "both"):
        ok &= check_core()
        print()
    if role in ("acoustic", "both"):
        ok &= check_acoustic()
        print()
    print("RESULT:", "ready ✓" if ok else "missing requirements ✗")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
