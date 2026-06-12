"""Teshis #17 — PROVENANCE karar-seviyesi karma-guven degerlendirmesi.

Provenance bir AYIRICI degil PRIOR -> sinyal-AUC ANLAMSIZ (bkz. spec). Deger
yalniz KARAR-SEVIYESI (allow/flag/block) ile gosterilir. Uc bilesen:
  1. user-trusted imperatif doku (WEDGE): provenance SUSMALI -> ~%100 allow.
  2. untrusted benign retrieved (over-defense maliyeti): tam fuzyon yiginindan
     gecince allow/flag/block? -> CAP_PROV bu satirdan TURETILIR.
  3. untrusted injected (fayda): cap=0'a gore block-orani artiyor mu?

CAP_PROV SWEEP: off/0.3/0.35/0.5. Embedding YOK, offline, hizli.
Cikti: karar-seviyesi tablo -> Cagri ile CAP_PROV ortak kalibrasyonu.
"""
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from reasongate.shield import Shield
from reasongate.types import Segment
from reasongate.detectors.indirect import IndirectInjectionDetector

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
FAMILY2 = {"Task Automation", "Business Intelligence", "Conversational Agent",
           "Research Assistance", "Sentiment Analysis"}
CAPS = [0.0, 0.30, 0.35, 0.50]


def load():
    bi = json.load(open(os.path.join(DATA, "benign_instructional.json")))
    benign = [x["text"] for x in bi]                       # 305 (tarif+how-to)
    yr = list(json.load(open(os.path.join(DATA, "benign_yourref.json"))))   # 20 wedge
    attacks = json.load(open(os.path.join(DATA, "bipia_text_attacks.json")))
    det = IndirectInjectionDetector()
    payloads, fam = [], []
    for cat, atks in attacks.items():
        for a in atks:
            if not (det.scan(a).matches and det.scan(a).score >= det.block_score):
                payloads.append(a); fam.append(2 if cat in FAMILY2 else 1)
    recipes = [x["text"] for x in bi if x.get("src") == "recipe"]
    return benign, yr, payloads, fam, recipes


def rates(shield, segments):
    """Her segment'i tek-tek scan_context'ten gecir, action dagilimi."""
    c = Counter(shield.scan_context([s]).action for s in segments)
    n = sum(c.values())
    return {k: 100.0 * c.get(k, 0) / n for k in ("allow", "flag", "block")}, n


def fmt_row(label, r):
    return (f"  {label:34} allow {r['allow']:5.1f}  flag {r['flag']:5.1f}  "
            f"block {r['block']:5.1f}")


def main():
    benign, yr, payloads, fam, recipes = load()
    # segment setleri
    # 1) WEDGE: user kendi imperatif dokusunu yapistirmis (ayni tarif metni, trust=user)
    trusted = [Segment(t, source="user", trust="trusted") for t in recipes]
    # 2) over-defense: benign retrieved (tarif+how-to) + your-ref, untrusted
    untrusted_benign = [Segment(t, source="retrieved", trust="untrusted") for t in benign]
    untrusted_yr = [Segment(t, source="retrieved", trust="untrusted") for t in yr]
    # 3) fayda: payload mesru recipe-host'a gomulu, untrusted (gercek indirect senaryo)
    injected = [Segment(recipes[i % len(recipes)] + "\n" + p, source="retrieved",
                        trust="untrusted")
                for i, p in enumerate(payloads)]
    inj_f2 = [s for s, f in zip(injected, fam) if f == 2]
    inj_f1 = [s for s, f in zip(injected, fam) if f == 1]

    print("=" * 74)
    print("PROVENANCE — karar-seviyesi karma-guven (CAP_PROV sweep, retrieved kaynak)")
    print("  retrieved agirligi 0.8 -> efektif skor = CAP x 0.8")
    print("=" * 74)

    for cap in CAPS:
        sh = Shield(provenance_cap=cap)
        tag = "OFF (baseline)" if cap == 0.0 else f"CAP={cap}"
        print(f"\n--- {tag} ---")
        r, n = rates(sh, trusted)
        print(fmt_row(f"1 WEDGE user-trusted imp ({n})", r) +
              ("   <- ~100 allow olmali" if cap > 0 else ""))
        r, _ = rates(sh, untrusted_benign)
        print(fmt_row(f"2 untrusted benign tarif/howto", r) + "   <- over-defense maliyeti")
        r, _ = rates(sh, untrusted_yr)
        print(fmt_row(f"2 untrusted your-ref wedge", r))
        rf2, _ = rates(sh, inj_f2)
        rf1, _ = rates(sh, inj_f1)
        print(fmt_row(f"3 injected AILE-2 ({len(inj_f2)})", rf2) + "   <- fayda (asil hedef)")
        print(fmt_row(f"3 injected aile-1 ({len(inj_f1)})", rf1))

    # --- NEDEN DUZ: ikinci-sinyal frekansi (provenance fuzyon icin es-sinyal arar) ---
    from reasongate.policy import fuse, FUSION_FLOOR
    idet = IndirectInjectionDetector()

    def s2_freq(segs):
        sc = [idet.scan(s.text).score for s in segs]
        return (100.0 * sum(x >= FUSION_FLOOR for x in sc) / len(sc),
                100.0 * sum(x >= 0.5 for x in sc) / len(sc), max(sc))

    print("\n" + "=" * 74)
    print("NEDEN DUZ — IKINCI-SINYAL frekansi (IndirectInjectionDetector, canli yigin)")
    print("=" * 74)
    for name, segs in [("untrusted benign", untrusted_benign),
                       ("injected AILE-2", inj_f2), ("injected aile-1", inj_f1)]:
        f30, f50, mx = s2_freq(segs)
        print(f"  {name:20}: score>=0.30 %{f30:.1f}  >=0.50 %{f50:.1f}  max={mx:.2f}")
    print("  -> residue (aile-2 ozellikle) ikinci sinyal tripslemiyor; provenance'in")
    print("     fuzyonda birlesecek es-sinyali YOK -> tek basina sub-flag -> ATIL.")

    print("\n" + "=" * 74)
    print("ETKILESIM TABLOSU — fuse(provenance=CAPx0.8, ikinci_sinyal s2) -> aksiyon")
    print("  (provenance'in DEGERI burada: es-sinyal VARSA kararı nasil kaydirir)")
    print("=" * 74)

    def act(f):
        return "BLOCK" if f >= 0.8 else ("flag" if f >= 0.5 else "allow")
    s2s = [0.0, 0.30, 0.50, 0.60, 0.70]
    print(f"  {'CAP(eff)':>10} | " + " ".join(f"s2={s:>4}" for s in s2s))
    for cap in CAPS:
        eff = round(cap * 0.8, 2)
        cells = []
        for s2 in s2s:
            scores = [x for x in (eff, s2) if x >= FUSION_FLOOR]
            cells.append(f"{act(fuse([eff, s2])):>6}")
        print(f"  {eff:>10} | " + " ".join(cells))
    print("  OKUMA: es-sinyal yoksa (s2=0) tum CAP'lerde allow (over-defense YOK).")
    print("  es-sinyal 0.6-0.7 varsa provenance kararı flag/BLOCK'a kaydirir = fayda.")
    print("  Kalibrasyon: benign'de s2>=0.3 nadir (yukarida) -> arka-kapi riski dusuk;")
    print("  CAP_PROV es-sinyalli GERCEK saldiriyi block'a tasiyacak ama benign-coflag")
    print("  yapmayacak en yuksek deger. Host-incoherence (arsiv) dogal es-sinyal adayi.")
    print("=" * 74)


if __name__ == "__main__":
    main()
