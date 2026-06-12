"""Teshis #17 — de-konfaund probe: 0.96'lik (a) gercek 'injection-ozu' mu,
yoksa KONU-ayrimi mi? Pozitifleri aile-2 / aile-1-kalinti diye ayirip
ICERIK-eslesmeli negatife (notinject — ayni dilbilimsel siniftan masum
gorev/teknik istekleri) karsi koyar. Headline'i tek satira gommeden once.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import eval.diag17_separability as d

DATA = d.DATA
FAMILY2 = d.FAMILY2


def main():
    attacks = json.load(open(os.path.join(DATA, "bipia_text_attacks.json")))
    det = d.IndirectInjectionDetector()
    f2, f1 = [], []
    for cat, atks in attacks.items():
        for a in atks:
            if not (det.scan(a).matches and det.scan(a).score >= det.block_score):
                (f2 if cat in FAMILY2 else f1).append(a)

    recipes = [x["text"] for x in json.load(open(os.path.join(DATA, "benign_instructional.json")))
               if x.get("src") == "recipe"]
    chn = json.load(open(os.path.join(DATA, "clean_hardneg.json")))
    ni = [x["prompt"] for x in json.load(open(os.path.join(DATA, "notinject.json")))]

    Ef2 = d.embed(f2, "document")
    Ef1 = d.embed(f1, "document")
    Erec = d.embed(recipes, "document")
    Echn = d.embed(chn, "document")
    Eni = d.embed(ni, "document")

    print("=" * 70)
    print("PROBE — (a)'yi ICERIK-eslesmeli negatife karsi DE-KONFAUND")
    print(f"  pozitif: aile-2={len(f2)}  aile-1-kalinti={len(f1)}")
    print("  notinject = ayni siniftan masum gorev/teknik istekleri (asil zor-neg)")
    print("=" * 70)
    rows = [
        ("aile-2  vs notinject (ICERIK-eslesmeli)", Ef2, Eni),
        ("aile-2  vs clean_hardneg (Python how-to)", Ef2, Echn),
        ("aile-2  vs recipes (uzun, alakasiz konu)", Ef2, Erec),
        ("aile-1  vs notinject (ICERIK-eslesmeli)", Ef1, Eni),
        ("aile-1  vs recipes (uzun, alakasiz konu)", Ef1, Erec),
        ("TUM-50  vs notinject (ICERIK-eslesmeli)", np.vstack([Ef2, Ef1]), Eni),
    ]
    for name, P, N in rows:
        print(d.fmt(name, d.lr_oof(P, N)))
    print("=" * 70)
    print("OKUMA: aile-2-vs-notinject COKERSE -> 0.96 KONU-ayrimiydi, injection-ozu")
    print("DEGIL. Ayni dilbilimsel sinifta family-2 metin-seviyesinde ayrilamaz;")
    print("'app-layer provenance sart' olcumle desteklenir.")
    print("Aile-1 yuksek kalirsa: pattern-residue'nun lurid konulari (scam/cipher)")
    print("ayriliyor — ama o zaten saldiri-ozu degil konu; build'i yaniltir.")


if __name__ == "__main__":
    main()
