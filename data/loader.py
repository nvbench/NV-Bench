"""
NV-Bench Data Loader
====================
Unified loader for NV-Bench benchmark JSON files.
"""

import json
import os
from typing import List, Dict, Optional


def load_benchmark(json_path: str) -> List[Dict]:
    """Load a benchmark JSON file and return standardized list of dicts.

    Each dict contains:
        - wav_path: original reference audio
        - text: target text with NVV tags
        - target_wav_path: TTS-generated audio to evaluate
        - ref_wav_path: reference wav for speaker similarity (optional)
        - ref_text: reference text for SIM prompt (optional)

    Args:
        json_path: Path to the benchmark JSON file.

    Returns:
        List of standardized data dictionaries.
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if isinstance(data, dict):
        # Handle dict-format JSON (wav_path as key)
        items = []
        for wav_path, info in data.items():
            item = {"wav_path": wav_path}
            item.update(info)
            items.append(item)
        return items

    return data


def load_infer_results(result_path: str) -> List[Dict]:
    """Load inference results from either JSON or TXT format.

    Args:
        result_path: Path to inference results (JSON or TXT).

    Returns:
        List of dicts with wav_path, target, text fields.
    """
    if result_path.endswith('.json'):
        return _read_json_results(result_path)
    else:
        return _read_txt_results(result_path)


def _read_json_results(json_path: str) -> List[Dict[str, str]]:
    """Read JSON inference results."""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    data_list = []
    if isinstance(data, dict):
        for wav_path, info in data.items():
            data_list.append({
                "wav_path": wav_path,
                "target": info.get("gt_text", "") or info.get("target", ""),
                "text": info.get("predicted_text", "") or info.get("text", ""),
            })
    elif isinstance(data, list):
        for item in data:
            data_list.append({
                "wav_path": item.get('wav_path', ''),
                "target": item.get('target', '') or item.get('gt_text', ''),
                "text": item.get('text', '') or item.get('predicted_text', ''),
            })
    return data_list


def _read_txt_results(file_path: str) -> List[Dict[str, str]]:
    """Read TXT inference results (legacy format).

    Expected format:
        wav_path: /path/to/audio.wav
        target: ground truth text
        text: predicted text
        timestamp: [...]

        (blank line separator)
    """
    data_list, data = [], {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line.startswith('wav_path:'):
                if data:
                    data_list.append(data)
                data = {'wav_path': line.split(': ', 1)[1]}
            elif not line:
                if data:
                    data_list.append(data)
                    data = {}
            else:
                parts = line.split(': ')
                if len(parts) >= 2:
                    key = parts[0]
                    value = ': '.join(parts[1:])
                    data[key] = value
    if data:
        data_list.append(data)
    return data_list
