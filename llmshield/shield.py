"""Shield — model-bagimsiz guvenlik kalkani.

Herhangi bir LLM fonksiyonunu (prompt:str -> str) sarar:
  1) Girdiyi input-dedektorlerden gecirir; bloklanirsa LLM'i HIC cagirmaz.
  2) Izin varsa LLM'i cagirir.
  3) Ciktiyi output-dedektorlerden gecirir; bloklanirsa cikti yerine gerekce doner.

Her karar bir ShieldResult.explain() ile "neden" aciklamasi tasir.
"""
from __future__ import annotations

from typing import Callable, List, Optional

from llmshield import policy
from llmshield.detectors.base import Detector
from llmshield.detectors import InjectionDetector, LeakageDetector
from llmshield.types import ShieldResult


class Shield:
    def __init__(self,
                 input_detectors: Optional[List[Detector]] = None,
                 output_detectors: Optional[List[Detector]] = None,
                 block_threshold: float = 0.8,
                 flag_threshold: float = 0.5):
        # Varsayilan: makul bir baslangic seti
        self.input_detectors = input_detectors if input_detectors is not None else [InjectionDetector()]
        self.output_detectors = output_detectors if output_detectors is not None else [LeakageDetector()]
        self.block_threshold = block_threshold
        self.flag_threshold = flag_threshold

    def scan_input(self, prompt: str) -> ShieldResult:
        dets = [d.scan(prompt) for d in self.input_detectors]
        action, _ = policy.decide(dets, self.block_threshold, self.flag_threshold)
        return ShieldResult(action=action, stage="input", detections=dets)

    def scan_output(self, text: str) -> ShieldResult:
        dets = [d.scan(text) for d in self.output_detectors]
        action, _ = policy.decide(dets, self.block_threshold, self.flag_threshold)
        return ShieldResult(action=action, stage="output", detections=dets, output=text)

    def protect(self, prompt: str, llm_fn: Callable[[str], str]) -> ShieldResult:
        """Tek cagri: girdi tara -> (gerekirse) LLM cagir -> cikti tara."""
        inp = self.scan_input(prompt)
        if inp.action == "block":
            return inp  # LLM hic cagrilmadi
        raw = llm_fn(prompt)
        out = self.scan_output(raw)
        # girdi 'flag' ise ve cikti temizse, flag bilgisini koru
        if inp.action == "flag" and out.action == "allow":
            out.action = "flag"
            out.detections = inp.detections + out.detections
        return out

    def guard(self, llm_fn: Callable[[str], str]) -> Callable[[str], ShieldResult]:
        """Herhangi bir LLM fonksiyonunu korumali bir surumune cevirir."""
        def wrapped(prompt: str) -> ShieldResult:
            return self.protect(prompt, llm_fn)
        return wrapped
