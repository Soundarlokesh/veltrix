"""Groq chat-completions provider.

The model ID must be set in GROQ_MODEL. There is no default, because no model ID has been
verified against Groq's current catalogue from this environment. Nothing is sent until both
GROQ_API_KEY and GROQ_MODEL are set.
"""

from __future__ import annotations

from app.llm.provider import LLMConfigError, LLMError, LLMTimeoutError


class GroqProvider:
    name = "groq"

    def __init__(self, api_key: str | None, model: str) -> None:
        if not api_key:
            raise LLMConfigError("GROQ_API_KEY is not set. Set it in .env, or use LLM_PROVIDER=mock.")
        if not model:
            raise LLMConfigError(
                "GROQ_MODEL is not set. Set it to a model ID verified in your Groq console."
            )
        self._api_key = api_key
        self._model = model

    def complete(self, system: str, user: str, timeout: float) -> str:
        import groq

        client = groq.Groq(api_key=self._api_key, timeout=timeout, max_retries=0)
        try:
            response = client.chat.completions.create(
                model=self._model,
                temperature=0,
                max_tokens=700,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
        except groq.APITimeoutError as exc:
            raise LLMTimeoutError("the model did not respond in time") from exc
        except groq.APIError as exc:
            raise LLMError("the model request failed") from exc
        content = response.choices[0].message.content
        if not content:
            raise LLMError("the model returned an empty reply")
        return content
