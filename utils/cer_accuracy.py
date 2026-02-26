# CER/Accuracy utilities for NV-Bench
# Adapted from nvasr/utils/cer_accuracy.py

import numpy as np
import re

CONTRACTIONS = {
    r"\bI'm\b": "I am",
    r"\bI've\b": "I have",
    r"\bI'd\b": "I would",
    r"\bI'll\b": "I will",
    r"\bHe's\b": "He is",
    r"\bShe's\b": "She is",
    r"\bIt's\b": "It is",
    r"\bWe're\b": "We are",
    r"\bThey're\b": "They are",
    r"\bcan't\b": "cannot",
    r"\bdon't\b": "do not",
    r"\bdont\b": "do not",
    r"\bcant\b": "cannot",
}


def _normalize_segment(seg: str) -> str:
    for pattern, repl in CONTRACTIONS.items():
        seg = re.sub(pattern, repl, seg, flags=re.IGNORECASE)
    seg = seg.lower()
    seg = re.sub(r"[^a-z0-9\s]", "", seg)
    seg = re.sub(r"\s+", " ", seg).strip()
    return seg


def normalize_text(text: str) -> str:
    """Normalize text while preserving [...] tags."""
    parts = re.split(r'(\[[^\]]+\])', text)
    normalized_parts = []
    for part in parts:
        if not part:
            continue
        if part.startswith("[") and part.endswith("]"):
            normalized_parts.append(part)
        else:
            normalized_parts.append(_normalize_segment(part))
    return " ".join(p for p in normalized_parts if p)


def extract_paralingustic(text):
    """Extract paralinguistic tags [...] from text."""
    matches = re.findall(r"\[[^\]]+\]", text)
    return matches


def split_by_angle_brackets_and_language(text):
    pattern = re.compile(
        r'([.*?])'
        r"|([A-Za-z0-9%/.,;…—·~\u201c\u201d「」、：（）～《》『』；'!-]+)"
        r'|([\u4e00-\u9fff])'
    )
    tokens = []
    for br, en, zh in pattern.findall(text):
        if br:
            tokens.append(br)
        elif en:
            tokens.append(en)
        elif zh:
            tokens.append(zh)
    return tokens


def edit_distance(s1, s2):
    if type(s1) is str:
        s1 = split_by_angle_brackets_and_language(s1)
    if type(s2) is str:
        s2 = split_by_angle_brackets_and_language(s2)
    m, n = len(s1), len(s2)
    dp = np.zeros((m+1, n+1), dtype=int)
    for i in range(m+1):
        dp[i][0] = i
    for j in range(n+1):
        dp[0][j] = j
    for i in range(1, m+1):
        for j in range(1, n+1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = min(dp[i-1][j-1], dp[i-1][j], dp[i][j-1]) + 1
    return dp[m][n], len(s1)


def cer(ground_truth, transcription):
    distance, length = edit_distance(ground_truth, transcription)
    return distance / length


def accuracy(ground_truth, transcription):
    correct_chars = sum(1 for gt_char, tr_char in zip(ground_truth, transcription) if gt_char == tr_char)
    return correct_chars / len(ground_truth)


def remove_punctuation(text):
    pattern = re.compile(r"[\u3000-\u303f\uff00-\uffef]|[%/.,。，!！?？;…——·~•\u201c\u201d「」、：（）～《》『』；]")
    return re.sub(pattern, "", text)


def remove_paralingustic(text):
    pattern = re.compile(r"\[[^\]]+\]")
    return re.sub(pattern, "", text)
