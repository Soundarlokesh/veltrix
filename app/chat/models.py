"""Result types for one chat turn. The route rules of TASK.pdf section 8.1 are enforced here."""

from __future__ import annotations

from dataclasses import dataclass, field

ROUTES = ("rag", "tool", "hybrid", "clarify", "no_answer", "refuse")
TOOL_STATUSES = ("pending_confirmation", "success", "error")


@dataclass(frozen=True)
class Citation:
    doc_id: str
    locator: str


@dataclass(frozen=True)
class ToolCallRecord:
    name: str
    args: dict
    status: str


@dataclass(frozen=True)
class ChatResult:
    route: str
    answer: str
    citations: tuple[Citation, ...] = ()
    tool_calls: tuple[ToolCallRecord, ...] = ()
    pending_confirmation: bool = False
    extra: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.route not in ROUTES:
            raise ValueError(f"invalid route {self.route!r}")
        for call in self.tool_calls:
            if call.status not in TOOL_STATUSES:
                raise ValueError(f"invalid tool status {call.status!r}")
        has_citations = bool(self.citations)
        has_tools = bool(self.tool_calls)
        if self.route == "rag" and not (has_citations and not has_tools):
            raise ValueError("rag needs citations and no tool calls")
        if self.route == "tool" and not (has_tools and not has_citations):
            raise ValueError("tool needs tool calls and no citations")
        if self.route == "hybrid" and not (has_citations and has_tools):
            raise ValueError("hybrid needs citations and tool calls")
        if self.route in ("clarify", "no_answer", "refuse") and (has_citations or has_tools):
            raise ValueError(f"{self.route} must have no citations or tool calls")
        if self.pending_confirmation and not any(c.status == "pending_confirmation" for c in self.tool_calls):
            raise ValueError("pending_confirmation needs a pending_confirmation tool call")
