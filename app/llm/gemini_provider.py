"""Google Gemini provider using the official google-genai SDK (package `google-genai`).

GEMINI_MODEL must be set. There is no default, because no model ID has been verified from this
environment. Nothing is sent until GEMINI_API_KEY and GEMINI_MODEL are both set.
SDK retries are disabled so a request cannot exceed the assessment's 30-second limit.
"""

from __future__ import annotations

import logging

from app.llm.provider import LLMConfigError, LLMError, LLMTimeoutError

logger = logging.getLogger(__name__)
# 700 was too small: replies were cut off mid-JSON in live tests (see PROGRESS.txt section 13).
MAX_OUTPUT_TOKENS = 2048


class GeminiProvider:
    name = "gemini"

    def __init__(self, api_key: str | None, model: str) -> None:
        if not api_key:
            raise LLMConfigError("GEMINI_API_KEY is not set. Set it in .env, or use LLM_PROVIDER=mock.")
        if not model:
            raise LLMConfigError("GEMINI_MODEL is not set. Set it to a model ID you verified in Google AI Studio.")
        self._api_key = api_key
        self._model = model

    def complete(self, system: str, user: str, timeout: float) -> str:
        import httpx
        from google import genai
        from google.genai import errors, types

        client = genai.Client(
            api_key=self._api_key,
            http_options=types.HttpOptions(
                timeout=int(timeout * 1000),  # the SDK expects milliseconds
                retry_options=types.HttpRetryOptions(attempts=1),
            ),
        )
        try:
            response = client.models.generate_content(
                model=self._model,
                contents=user,
                config=types.GenerateContentConfig(
                    system_instruction=system,
                    temperature=0,
                    max_output_tokens=MAX_OUTPUT_TOKENS,
                    response_mime_type="application/json",
                ),
            )
        except httpx.TimeoutException as exc:
            raise LLMTimeoutError("the model did not respond in time") from exc
        except errors.APIError as exc:
            raise LLMError(f"the model request failed (status {exc.code})") from exc
        except httpx.HTTPError as exc:
            raise LLMError("the model request failed") from exc

        try:
            text = response.text
        except Exception as exc:  # the SDK raises when a reply is blocked or has no text
            raise LLMError("the model returned no usable text") from exc
        finish = _finish_reason(response)
        logger.info("gemini reply finish_reason=%s chars=%d", finish, len(text or ""))
        if finish and finish.upper().endswith("MAX_TOKENS"):
            logger.warning("gemini reply hit the output limit; the reply may be truncated")
        if not text:
            raise LLMError("the model returned an empty reply")
        return text


def _finish_reason(response) -> str | None:
    """Why the model stopped (for example STOP or MAX_TOKENS). Logged only; never the reply text."""
    try:
        reason = response.candidates[0].finish_reason
    except (AttributeError, IndexError, TypeError):
        return None
    return getattr(reason, "name", None) or (str(reason) if reason is not None else None)
