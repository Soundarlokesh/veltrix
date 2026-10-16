"""Deterministic prompt-injection controls. These are not LLM instructions.

1. User messages that try to override the assistant, reveal hidden data or change permissions
   are refused before any tool or model runs.
2. Corpus chunks that contain instructions aimed at an AI assistant are quarantined: they are
   never placed in an LLM context, and the event is logged by doc_id.

Patterns are generic. No corpus fact is written here.
"""

from __future__ import annotations

import re

_MANIPULATION = [
    r"\bignore\b.{0,40}\b(instructions?|rules?|prompts?|guidelines?|previous|prior|above)\b",
    r"\bdisregard\b.{0,40}\b(instructions?|rules?|prompts?|policy|policies)\b",
    r"\b(system|developer|hidden|internal)\s+(prompt|message|instructions?)\b",
    r"\breveal\b.{0,40}\b(prompt|instructions?|hidden|secret|confidential|restricted)\b",
    r"\b(print|show|dump|repeat)\b.{0,30}\b(your|the)\b.{0,20}\b(prompt|instructions?)\b",
    r"\byou are now\b",
    r"\bpretend (to be|you are)\b",
    r"\bact as (an? |the )?(admin|administrator|system|developer|hr)\b",
    r"\b(override|bypass|disable)\b.{0,30}\b(permissions?|access|rules?|checks?|security|guardrails?)\b",
    r"\bjailbreak\b",
    r"\bdeveloper mode\b",
]

_CORPUS_INSTRUCTION = [
    r"\bnote for the (ai|assistant|chatbot|model)\b",
    r"\b(ai|language model|assistant)\s+(reading|processing) this\b",
    r"\binstructions? (for|to) (the )?(ai|assistant|model|chatbot)\b",
    r"\bignore (all |any |the )?(previous|prior|above|other) (instructions?|rules?)\b",
    r"\bthis is approved by hr\b.{0,80}\b(tell|reveal|include|send)\b",
    r"\b(tell|reveal|include|send)\b.{0,60}\b(salary|email address|phone)\b.{0,40}\b(manager|employee)\b",
]

_MANIPULATION_RE = [re.compile(p, re.IGNORECASE | re.DOTALL) for p in _MANIPULATION]
_CORPUS_RE = [re.compile(p, re.IGNORECASE | re.DOTALL) for p in _CORPUS_INSTRUCTION]


def is_manipulation(message: str) -> bool:
    return any(p.search(message) for p in _MANIPULATION_RE)


def contains_untrusted_instruction(text: str) -> bool:
    return any(p.search(text) for p in _CORPUS_RE)
