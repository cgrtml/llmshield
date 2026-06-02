"""Siniflandirma metrikleri. y: 1=saldiri, 0=iyi-huylu."""
from __future__ import annotations

from typing import Dict, List


def report(y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    recall = tp / (tp + fn) if (tp + fn) else 0.0        # tespit orani (saldirilar)
    fpr = fp / (fp + tn) if (fp + tn) else 0.0           # yanlis-alarm orani
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    acc = (tp + tn) / len(y_true) if y_true else 0.0
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "recall": recall, "fpr": fpr, "precision": precision, "f1": f1, "accuracy": acc}


def pretty(m: Dict[str, float]) -> str:
    return (
        f"  Tespit orani (recall)   : %{100*m['recall']:.1f}  ({m['tp']}/{m['tp']+m['fn']} saldiri yakalandi)\n"
        f"  Yanlis-alarm (FPR)      : %{100*m['fpr']:.1f}  ({m['fp']}/{m['fp']+m['tn']} iyi-huylu yanlis bloklandi)\n"
        f"  Precision               : %{100*m['precision']:.1f}\n"
        f"  F1                      : {m['f1']:.3f}\n"
        f"  Dogruluk (accuracy)     : %{100*m['accuracy']:.1f}\n"
        f"  Karisiklik: TP={m['tp']} FP={m['fp']} TN={m['tn']} FN={m['fn']}"
    )
