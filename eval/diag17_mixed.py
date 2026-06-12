"""Teshis #17 — KARISIK-CHUNK testi (teshisin kor noktasi).

cos_max IZOLE payload'larda olculdu (chunk = saldiri). Gercek indirect injection
payload'i MESRU HOST'un icine gomulu gelir. O chunk'ta host'un mesru imperatifleri
kullaniciyla hizali -> cos_max yuksek -> enjeksiyon maskelenir.

Kurulum: ayni host + ayni kullanici-promptu, injection VAR/YOK toggle (host+prompt
sabit, sadece payload degisir). Iki olcum:
  (A) mevcut cos_max bu sette COKER mu? (tahmin: evet -> 0.854 izole artefakti)
  (B) span-konjonksiyon: her span icin (user-hizasizlik u, host-tutarsizlik h).
      u = 1-cos(span,user); h = 1-cos(span, ayni chunk'taki DIGER cumleler / LOO).
      Gomulu payload: u yuksek + h yuksek. 'Take off heat': u yuksek ama h DUSUK
      (host-coherent) -> elenir. chunk skoru = max_span(u*h). Wedge = recipe+your-ref.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import eval.diag17_separability as d
from reasongate.detectors.imperative import extract_imperative_spans, _split_sentences

DATA = d.DATA
N_HOST = 100   # recipe host sayisi (maliyet)


def main():
    attacks = json.load(open(os.path.join(DATA, "bipia_text_attacks.json")))
    FAMILY2 = d.FAMILY2
    det = d.IndirectInjectionDetector()
    payloads, pay_fam = [], []
    for cat, atks in attacks.items():
        for a in atks:
            if not (det.scan(a).matches and det.scan(a).score >= det.block_score):
                payloads.append(a); pay_fam.append(2 if cat in FAMILY2 else 1)
    recipes = [x["text"] for x in json.load(open(os.path.join(DATA, "benign_instructional.json")))
               if x.get("src") == "recipe"][:N_HOST]
    yr = list(json.load(open(os.path.join(DATA, "benign_yourref.json"))))

    # --- chunk'lari kur: (user_prompt, chunk_text, label, kind) ---
    # host+prompt SABIT, injection toggle. user-prompt = GEVSEK yemek sorusu (dürüst).
    chunks = []
    for i, r in enumerate(recipes):
        up = d.FOOD_LOOSE[i % len(d.FOOD_LOOSE)]
        body = d.directions_span(r)
        chunks.append((up, body, 0, "clean_recipe"))                       # temiz host
        pay = payloads[i % len(payloads)]; fam = pay_fam[i % len(payloads)]
        kind = "injected_f2" if fam == 2 else "injected_f1"
        chunks.append((up, body + "\n" + pay, 1, kind))                    # host+gomulu payload
    for i, t in enumerate(yr):                                             # wedge: tek-span benign
        chunks.append((d.YR_LOOSE[i % len(d.YR_LOOSE)], t, 0, "clean_yourref"))

    # --- embedding: tum cumleler (LOO host-coherence icin) + promptlar ---
    all_sents, all_prompts = set(), set()
    for up, ch, lab, kind in chunks:
        all_prompts.add(up)
        for s in _split_sentences(ch):
            all_sents.add(s)
    all_sents = list(all_sents); all_prompts = list(all_prompts)
    Es = dict(zip(all_sents, d.embed(all_sents, "document")))
    Pq = dict(zip(all_prompts, d.embed(all_prompts, "query")))

    def cos1(a, b):
        a = a / (np.linalg.norm(a) + 1e-9); b = b / (np.linalg.norm(b) + 1e-9)
        return float(a @ b)

    # payload kapi-kapsami (family bazli): payload TEK BASINA imperatif-form mu?
    cov2 = [extract_imperative_spans(p) != [] for p, f in zip(payloads, pay_fam) if f == 2]
    cov1 = [extract_imperative_spans(p) != [] for p, f in zip(payloads, pay_fam) if f == 1]

    rows = []   # (label, kind, cos_max, max_u, max_h, max_uh, max_min)
    for up, ch, lab, kind in chunks:
        sents = _split_sentences(ch)
        gated = extract_imperative_spans(ch)
        if not gated:
            gated = [ch.strip()]
        qv = Pq[up]
        cos_max = -1.0; mu = mh = muh = mmin = 0.0
        for g in gated:
            cu = cos1(Es.get(g, qv*0), qv)             # span-user cosine
            cos_max = max(cos_max, cu)
            u = 1.0 - cu                                # user-hizasizlik
            others = [Es[s] for s in sents if s != g]
            if others:
                hc = np.mean(others, axis=0)
                h = 1.0 - cos1(Es.get(g, hc*0), hc)    # host-tutarsizlik (LOO)
            else:
                h = 0.0                                 # tek-span -> tutarsiz olamaz
            muh = max(muh, u * h); mmin = max(mmin, min(u, h))
            mu = max(mu, u); mh = max(mh, h)
        rows.append((lab, kind, cos_max, mu, mh, muh, mmin))

    kinds = np.array([r[1] for r in rows])
    R = np.array([(l, cm, u, h, uh, mn) for l, k, cm, u, h, uh, mn in rows])
    yall = R[:, 0]
    clean_mask = yall == 0
    print("=" * 70)
    print("KARISIK-CHUNK — host+gomulu payload vs temiz host (host+prompt SABIT)")
    print("=" * 70)
    nf2 = int((kinds == "injected_f2").sum()); nf1 = int((kinds == "injected_f1").sum())
    print(f"chunk: {nf2} injected_f2 + {nf1} injected_f1 + {int(clean_mask.sum())} temiz "
          f"({len(recipes)} recipe + {len(yr)} your-ref)")
    print(f"payload KAPI-KAPSAMI (imperatif-form): aile-2 {sum(cov2)}/{len(cov2)}  "
          f"aile-1 {sum(cov1)}/{len(cov1)}  <- kapsam-sinir, gizli degil")

    def auc_vs_clean(pos_kind, col, hi=True):
        m = clean_mask | (kinds == pos_kind)
        s = R[m, col] if hi else -R[m, col]
        return d.auc_ci(R[m, 0], s)

    print("\n[AUC — injected vs temiz]  (AILE-2 = #17'nin asil hedefi)")
    for tag, kind in [("AILE-2", "injected_f2"), ("aile-1", "injected_f1")]:
        print(f"  -- {tag} --")
        print(d.fmt("   (A) cos_max  [IZOLE'de 0.854]", auc_vs_clean(kind, 1, hi=False)))
        print(d.fmt("       max_u  (user-hizasizlik)", auc_vs_clean(kind, 2)))
        print(d.fmt("       max_h  (host-tutarsizlik)", auc_vs_clean(kind, 3)))
        print(d.fmt("   (B) max(u*h)  KONJONKSIYON", auc_vs_clean(kind, 4)))

    # wedge: temiz recipe vs injected-f2
    rc = R[kinds == "clean_recipe"]; ri = R[kinds == "injected_f2"]
    yrr = R[kinds == "clean_yourref"]
    print("\n[konfaund/wedge denetimi — medyanlar, aile-2]")
    print(f"  cos_max:  temiz-recipe={np.median(rc[:,1]):.3f}  "
          f"injected-f2={np.median(ri[:,1]):.3f}  (yakinsa cos_max COKTU)")
    print(f"  max_h:    temiz-recipe={np.median(rc[:,3]):.3f}  "
          f"injected-f2={np.median(ri[:,3]):.3f}  (FP-riski: recipe h tabani)")
    print(f"  u*h:      temiz-recipe={np.median(rc[:,4]):.3f}  "
          f"injected-f2={np.median(ri[:,4]):.3f}  your-ref={np.median(yrr[:,4]):.3f}")

    print("\n[OKUMA]")
    print("  (A) cos_max coker (~0.5) + (B) u*h ayirir (>>0.5) ise: spec span-")
    print("  konjonksiyona doner, build dogru hedefe gider (7. sayi, build'den once).")
    print("  (B) de cokerse: chunk-seviyesi tavan -> app-layer provenance ZORUNLU.")
    print("=" * 70)


if __name__ == "__main__":
    main()
