"""Provenance dedektoru — KOKEN-temelli prior (ayirici DEGIL).

Indirect injection'in gercek imzasi metin degil BICIM + KOKEN: guvensiz
kaynaktan gelen TALIMAT-FORMU icerik. "Write a script..." kullanici promptunda
normal istek, retrieve edilen e-postada injection — ayni string, ayiran tek sey
koken. Bu dedektor metni AYIRMAYA calismaz (bkz. _notes/spec_17_provenance.md:
metin-seviyesi tavan olculdu); guvensiz kaynaktaki talimat-formuna TAVANLI,
triggered=False bir prior skor verir ki noisy-OR fuzyonunda diger sinyallerle
birlesince anlamli karar uretsin.

DEGER OLCUMU: sinyal-AUC DEGIL, karar-seviyesi karma-guven (eval/diag17_provenance.py).
CAP_PROV sabit DEGIL — karar-seviyesi FPR'den turetilir (preset disiplini).
"""
from __future__ import annotations

from reasongate.detectors.imperative import extract_imperative_spans
from reasongate.types import Detection, Segment

# Kaynak guven-agirligi: web en az guvenilir, retrieved orta. user=0 (susar).
_SOURCE_WEIGHT = {
    "web": 1.0, "tool": 0.9, "file": 0.85, "retrieved": 0.8,
    "user": 0.0, "trusted": 0.0,
}


class ProvenanceDetector:
    name = "provenance"
    stage = "context"

    def __init__(self, cap: float = 0.5):
        # CAP_PROV: bu sinyalin TAVANI. <block_threshold (tek basina block YOK).
        # Gercek deger karar-seviyesi karma-guven FPR'den kalibre edilir.
        self.cap = cap

    def scan_segment(self, seg: Segment) -> Detection:
        # Guvenilir koken (kullanicinin kendi verdigi) -> SUS (wedge korumasi).
        if seg.trust == "trusted" or seg.source == "user":
            return Detection(self.name, False, 0.0,
                             "guvenilir koken (user) — provenance sinyali yok", [])
        spans = extract_imperative_spans(seg.text)
        if not spans:
            return Detection(self.name, False, 0.0,
                             "guvensiz koken ama talimat-formu yok", [])
        w = _SOURCE_WEIGHT.get(seg.source, 0.8)
        score = round(self.cap * w, 3)         # triggered=False, score<=cap her zaman
        dom = f" ({seg.domain})" if seg.domain else ""
        reason = (f"guvensiz '{seg.source}'{dom} kokeninde asistana-yonelik "
                  f"talimat-formu ({len(spans)} span) — prior, tek basina block ETMEZ")
        return Detection(self.name, False, score, reason, spans[:3])
