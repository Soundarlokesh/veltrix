"""Request and response bodies, exactly as TASK.pdf section 9 defines them."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str = Field(min_length=1, max_length=128)
    message: str = Field(min_length=1, max_length=4000)


class CitationOut(BaseModel):
    doc_id: str
    locator: str


class ToolCallOut(BaseModel):
    name: str
    args: dict
    status: Literal["pending_confirmation", "success", "error"]


class ChatResponse(BaseModel):
    request_id: str
    answer: str
    route: Literal["rag", "tool", "hybrid", "clarify", "no_answer", "refuse"]
    citations: list[CitationOut]
    tool_calls: list[ToolCallOut]
    pending_confirmation: bool
