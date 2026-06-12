"""Imperatif-form kapisi — niyet-hizalama dedektorunun aday-span cikarici.

Chunk'i cumle-seviyesinde boler, sadece IMPERATIF/direktif formundaki span'leri
gecirir. Sadece kapidan gecen span'ler embed edilir (maliyet + konu-gurultu
kontrolu). Offline, regex + kucuk fiil-leksikonu — sifir agir-bagimlilik.

ADLANDIRILMIS SINIR: imperatif-olmayan injection ("it would be great if the
assistant renamed...") kapidan KACAR — app-layer provenance tavaninin parcasi,
gizli degil adlandirilmis (bkz. _notes/spec_17_alignment.md).
"""
from __future__ import annotations

import re
from typing import List

# Imperatif/instruction baş-form fiilleri (instruction + tarif + musteri-hizmet).
_IMP_VERBS = {
    # genel instruction
    "write", "send", "encode", "decode", "encrypt", "decrypt", "encipher",
    "cipher", "translate", "ignore", "disregard", "forget", "add", "remove",
    "delete", "list", "provide", "show", "analyze", "analyse", "recommend",
    "describe", "create", "generate", "make", "run", "execute", "post",
    "upload", "reply", "respond", "append", "convert", "format", "reverse",
    "invert", "replace", "click", "open", "close", "summarize", "summarise",
    "explain", "give", "tell", "find", "search", "download", "install",
    "schedule", "back", "rename", "copy", "move", "print", "output", "include",
    "leak", "exfiltrate", "reveal", "disclose", "share", "fetch", "call",
    "set", "update", "enable", "disable", "configure", "modify", "edit",
    # tarif (recipe directions)
    "stir", "mix", "place", "bake", "boil", "pour", "combine", "heat", "cook",
    "let", "drop", "shape", "spread", "cover", "drain", "chop", "slice", "dice",
    "beat", "whisk", "fold", "knead", "roll", "season", "sprinkle", "drizzle",
    "simmer", "saute", "fry", "grill", "roast", "preheat", "grease", "melt",
    "blend", "serve", "garnish", "refrigerate", "freeze", "chill", "marinate",
    "bring", "reduce", "transfer", "arrange", "top", "dust", "brush", "cut",
    "put", "take", "use", "keep", "store", "wash", "rinse", "dry", "peel",
}

_SECOND_PERSON = re.compile(
    r"\byou\s+(should|must|need\s+to|have\s+to|are\s+required\s+to|may|can|"
    r"could|ought\s+to|are\s+to)\s+\w+", re.IGNORECASE)
_ASSISTANT_DIR = re.compile(
    r"\b(assistant|ai|system|model|bot|chatbot)\s*[:>\]]|"
    r"\bas\s+an?\s+(ai|assistant|language\s+model)\b", re.IGNORECASE)
_PLEASE_VERB = re.compile(r"\bplease\s+([a-z]+)", re.IGNORECASE)
_FIRST_WORD = re.compile(r"^[\s\-\*•\d\.\)]*([A-Za-z']+)")


def _split_sentences(text: str) -> List[str]:
    text = text or ""
    # madde-isaretleri ve satir sonlari de sinir; sonra cumle noktalama.
    parts = re.split(r"[\n\r]+|(?<=[.!?])\s+|(?:^|\s)[\-\*•]\s+", text)
    return [p.strip(" \t\-\*•").strip() for p in parts if p and p.strip(" \t\-\*•").strip()]


def is_imperative(span: str) -> bool:
    """Span imperatif/direktif formunda mi?"""
    s = span.strip()
    if not s:
        return False
    if _SECOND_PERSON.search(s) or _ASSISTANT_DIR.search(s):
        return True
    m = _PLEASE_VERB.search(s)
    if m and m.group(1).lower() in _IMP_VERBS:
        return True
    fw = _FIRST_WORD.match(s)
    if fw and fw.group(1).lower() in _IMP_VERBS:
        return True
    return False


def extract_imperative_spans(text: str, max_spans: int = 12) -> List[str]:
    """Chunk'tan imperatif-form aday span'leri (cumle-seviyesi). Kapidan gecmeyen
    her sey elenir. Bos liste = chunk'ta imperatif-form yok (provenance-tavani sinir)."""
    spans = [s for s in _split_sentences(text) if is_imperative(s)]
    # tekrarlari koru-sirali tekille
    seen, out = set(), []
    for s in spans:
        if s.lower() not in seen:
            seen.add(s.lower()); out.append(s)
        if len(out) >= max_spans:
            break
    return out
