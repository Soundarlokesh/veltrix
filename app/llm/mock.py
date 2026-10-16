"""Deterministic providers.

ExtractiveMockProvider is the default when no API key is configured. It does NOT understand
policies. It returns the first in-effect passage from the context, quoted, with its citation,
and never returns structured rules. Use it for deterministic tests and demos only.

ScriptedProvider returns replies you give it. Tests use it to check how the application
handles model output (validation, precedence, citations, restricted content), not how well
a model answers.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable

from app.llm.provider import LLMError

BLOCK_RE = re.compile(
    r"<<CHUNK id=(?P<id>\S+) doc=(?P<doc>\S+) status=(?P<status>\S+) effective=\S+ locator=(?P<locator>.*?)>>\n"
    r"(?P<text>.*?)\n<</CHUNK>>",
    re.DOTALL,
)
SNIPPET_CHARS = 400
MOCK_CAVEAT = (
    "[No language model is configured, so this is an unverified passage from the cited document. "
    "It may be superseded or incomplete. Confirm with the HR team.] "
)


class ExtractiveMockProvider:
    name = "mock-extractive"

    def complete(self, system: str, user: str, timeout: float) -> str:
        blocks = [m.groupdict() for m in BLOCK_RE.finditer(user)]
        in_effect = [b for b in blocks if b["status"] == "IN_EFFECT"]
        if not in_effect:
            return json.dumps({"status": "no_answer", "answer": "", "cited_chunk_ids": [], "rules": []})
        first = in_effect[0]
        snippet = " ".join(first["text"].split())[:SNIPPET_CHARS]
        return json.dumps(
            {
                "status": "answer",
                "answer": MOCK_CAVEAT + snippet,
                "cited_chunk_ids": [first["id"]],
                "rules": [],
            }
        )


class ScriptedProvider:
    """Replies from a list or a function. Records every (system, user) prompt it receives."""

    name = "scripted"

    def __init__(self, replies: list[str] | Callable[[str, str], str] | None = None, error: Exception | None = None) -> None:
        self._replies = list(replies) if isinstance(replies, list) else replies
        self._error = error
        self.calls: list[tuple[str, str]] = []

    def complete(self, system: str, user: str, timeout: float) -> str:
        self.calls.append((system, user))
        if self._error is not None:
            raise self._error
        if callable(self._replies):
            return self._replies(system, user)
        if not self._replies:
            raise LLMError("no scripted reply left")
        return self._replies.pop(0)
