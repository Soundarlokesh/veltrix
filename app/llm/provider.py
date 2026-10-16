"""Pluggable chat-model interface. Business rules never depend on which provider is used."""

from __future__ import annotations

from typing import Protocol


class LLMError(Exception):
    """The provider failed. The caller answers with a safe, citation-free message."""


class LLMTimeoutError(LLMError):
    pass


class LLMConfigError(LLMError):
    """The provider is selected but not configured (missing key or model ID)."""


class ChatProvider(Protocol):
    name: str

    def complete(self, system: str, user: str, timeout: float) -> str:
        """Return the model's text reply. Must raise LLMTimeoutError or LLMError."""
        ...


def make_provider(
    provider_name: str,
    api_key: str | None,
    model: str,
    timeout: float,
    gemini_api_key: str | None = None,
    gemini_model: str = "",
) -> ChatProvider:
    """Groq uses api_key and model. Gemini uses gemini_api_key and gemini_model."""
    provider = (provider_name or "mock").strip().lower()
    if provider == "mock":
        from app.llm.mock import ExtractiveMockProvider

        return ExtractiveMockProvider()
    if provider == "groq":
        from app.llm.groq_provider import GroqProvider

        return GroqProvider(api_key=api_key, model=model)
    if provider == "gemini":
        from app.llm.gemini_provider import GeminiProvider

        return GeminiProvider(api_key=gemini_api_key, model=gemini_model)
    raise LLMConfigError(f"Unknown LLM_PROVIDER {provider_name!r}. Use mock, groq or gemini.")
