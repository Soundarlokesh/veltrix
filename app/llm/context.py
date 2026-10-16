"""Build the corpus context for one LLM call.

Only EligibleChunk or UpcomingChunk objects are accepted (app/policy/filtering.py). The context
is limited to 3,000 tokens of corpus text (TASK.pdf section 4.3). The system prompt and the
question are not corpus text and are not counted here.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

from app.policy.filtering import EligibleChunk, UpcomingChunk
from app.tokens import count_tokens

logger = logging.getLogger(__name__)

CORPUS_TOKEN_BUDGET = 3000


@dataclass(frozen=True)
class CorpusContext:
    text: str
    tokens: int
    doc_ids: tuple[str, ...]
    chunk_ids: tuple[str, ...]
    dropped: int


def _block(item: EligibleChunk | UpcomingChunk) -> str:
    chunk = item.chunk
    return (
        f"<<CHUNK id={chunk.chunk_id} doc={chunk.doc_id} status={item.status} "
        f"effective={chunk.effective_date} locator={chunk.locator}>>\n{chunk.text}\n<</CHUNK>>"
    )


def build_context(items: Sequence[object], max_tokens: int = CORPUS_TOKEN_BUDGET) -> CorpusContext:
    """Pack blocks in the given order (most relevant first). A block that does not fit is skipped
    whole, never cut. Raises TypeError for anything that did not pass the policy filter."""
    for item in items:
        if not isinstance(item, (EligibleChunk, UpcomingChunk)):
            raise TypeError("only policy-filtered chunks may be sent to the LLM")

    blocks: list[str] = []
    doc_ids: list[str] = []
    chunk_ids: list[str] = []
    dropped = 0
    for item in items:
        assert isinstance(item, (EligibleChunk, UpcomingChunk))
        candidate = "\n\n".join(blocks + [_block(item)])
        if count_tokens(candidate) > max_tokens:
            dropped += 1
            continue
        blocks.append(_block(item))
        chunk_ids.append(item.chunk.chunk_id)
        if item.chunk.doc_id not in doc_ids:
            doc_ids.append(item.chunk.doc_id)
    text = "\n\n".join(blocks)
    tokens = count_tokens(text) if text else 0
    logger.info("llm context corpus_tokens=%d chunks=%d docs=%d dropped=%d", tokens, len(chunk_ids), len(doc_ids), dropped)
    return CorpusContext(text=text, tokens=tokens, doc_ids=tuple(doc_ids), chunk_ids=tuple(chunk_ids), dropped=dropped)
