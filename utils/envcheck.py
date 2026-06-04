"""
NV-Bench environment self-check
===============================
Every step depends on a *specific* conda env (see README "Environments").
Getting the wrong env is the #1 deployment papercut, so instead of crashing
deep inside a third-party import we check up front and print exactly what is
missing and which env to use.

Usage (top of each step script):

    from utils.envcheck import require
    require(
        step="3b · FAD/FD/KL",
        env="nvbench-acoustic",
        modules=["audioldm_eval", "torch"],
        assets=[("Cnn14 16k checkpoint", os.path.join(PROJECT_ROOT, "ckpt", "Cnn14_16k_mAP=0.438.pth"))],
        hint="conda activate nvbench-acoustic   # see scripts/setup.sh",
    )
"""

import importlib.util
import os
import sys
from typing import List, Sequence, Tuple


def _module_available(mod: str) -> bool:
    """True if ``mod`` is importable, WITHOUT importing it.

    Uses find_spec so the pre-flight check stays cheap — we don't want to pay
    the multi-second cost of actually importing torch/funasr/audioldm_eval just
    to fail fast. (scripts/check_env.py does a real import on purpose, as a
    doctor that also catches broken installs / ABI mismatches.)
    """
    try:
        return importlib.util.find_spec(mod) is not None
    except (ImportError, ValueError):
        return False


def require(
    step: str,
    env: str,
    modules: Sequence[str] = (),
    assets: Sequence[Tuple[str, str]] = (),
    hint: str = "",
    fatal: bool = True,
) -> List[str]:
    """Verify the current env can run ``step``.

    Args:
        step:    Human label, e.g. "3b · FAD/FD/KL".
        env:     Conda env this step expects, e.g. "nvbench-acoustic".
        modules: Python modules that must import.
        assets:  (label, path) pairs that must exist on disk.
        hint:    Extra one-line remediation hint.
        fatal:   Exit(1) on failure (default). If False, return the problem list.

    Returns:
        List of problem strings (empty if everything is satisfied).
    """
    problems = []
    for mod in modules:
        if not _module_available(mod):
            problems.append(f"missing python module: {mod}")
    for label, path in assets:
        if not path or not os.path.exists(path):
            problems.append(f"missing asset: {label} -> {path}")

    if not problems:
        return []

    msg = [
        "",
        "═" * 70,
        f"  NV-Bench: environment not ready for Step {step}",
        "═" * 70,
        f"  Expected conda env : {env}",
        f"  Active interpreter : {sys.executable}",
        "",
        "  Problems:",
    ]
    msg += [f"    ✗ {p}" for p in problems]
    msg += [
        "",
        "  Fix:",
        f"    conda activate {env}",
    ]
    if hint:
        msg.append(f"    {hint}")
    msg += [
        "    # or run scripts/setup.sh to build both envs from scratch",
        "═" * 70,
        "",
    ]
    sys.stderr.write("\n".join(msg) + "\n")

    if fatal:
        sys.exit(1)
    return problems
