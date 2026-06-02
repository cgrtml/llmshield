"""Policy: tespitlerden karar uretir (blokla / isaretle / izin ver) + gerekce.

Esik tabanli ve seffaf: en yuksek skoru esiklerle karsilastirir.
"""
from __future__ import annotations

from typing import List, Tuple

from llmshield.types import Detection


def decide(detections: List[Detection],
           block_threshold: float = 0.8,
           flag_threshold: float = 0.5) -> Tuple[str, List[Detection]]:
    """Donus: (action, tetikleyen_tespitler). action in {allow, flag, block}."""
    if not detections:
        return "allow", []
    # Dedektorun KENDI kalibre esigi (triggered) blok karari verir.
    triggered = [d for d in detections if d.triggered]
    if triggered:
        return "block", triggered
    flagged = [d for d in detections if d.score >= flag_threshold]
    if flagged:
        return "flag", flagged
    return "allow", []
