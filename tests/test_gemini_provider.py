"""Gemini provider tests. No network and no paid API: google.genai.Client is replaced by a fake.

The fake key below is a placeholder used only inside these tests; it is not a credential."""

import dataclasses
import json

import httpx
import pytest
from google import genai
from google.genai import errors

from app.config import Settings
from app.llm.gemini_provider import GeminiProvider
from app.llm.mock import ExtractiveMockProvider
from app.llm.provider import LLMConfigError, LLMError, LLMTimeoutError, make_provider
from app.main import create_app

FAKE_KEY = "fake-key-for-tests-only"
FAKE_MODEL = "fake-model-for-tests"


class _Response:
    def __init__(self, text):
        self.text = text


class _FakeModels:
    def __init__(self, owner):
        self._owner = owner

    def generate_content(self, **kwargs):
        self._owner.calls.append(kwargs)
        outcome = self._owner.outcome
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class FakeClient:
    """Stands in for google.genai.Client. Records how it was built and called."""

    init_kwargs: dict = {}
    calls: list = []

    def __init__(self, **kwargs):
        FakeClient.init_kwargs = kwargs
        self.models = _FakeModels(self)

    @property
    def outcome(self):
        return FakeClient._outcome

    @outcome.setter
    def outcome(self, value):
        FakeClient._outcome = value


@pytest.fixture
def fake_sdk(monkeypatch):
    FakeClient.calls = []
    FakeClient.init_kwargs = {}
    FakeClient._outcome = _Response("reply text")
    monkeypatch.setattr(genai, "Client", FakeClient)
    return FakeClient


def test_make_provider_selects_gemini():
    provider = make_provider("gemini", None, "", 5, gemini_api_key=FAKE_KEY, gemini_model=FAKE_MODEL)
    assert isinstance(provider, GeminiProvider) and provider.name == "gemini"


def test_mock_and_groq_selection_is_unchanged():
    assert isinstance(make_provider("mock", None, "", 5), ExtractiveMockProvider)
    with pytest.raises(LLMConfigError):
        make_provider("groq", None, "some-model", 5)  # still needs its own key


def test_unknown_provider_is_rejected():
    with pytest.raises(LLMConfigError, match="gemini"):
        make_provider("openai-typo", None, "", 5)


def test_missing_gemini_key_is_a_configuration_error_not_a_fallback():
    # A Groq key must never be used as a Gemini key, and the mock must never silently replace Gemini.
    with pytest.raises(LLMConfigError, match="GEMINI_API_KEY"):
        make_provider("gemini", "groq-key-not-for-gemini", "groq-model", 5, gemini_api_key=None, gemini_model=FAKE_MODEL)


def test_missing_gemini_model_is_a_configuration_error():
    with pytest.raises(LLMConfigError, match="GEMINI_MODEL"):
        make_provider("gemini", None, "", 5, gemini_api_key=FAKE_KEY, gemini_model="")


def test_successful_call_sends_the_expected_request(fake_sdk):
    provider = GeminiProvider(FAKE_KEY, FAKE_MODEL)
    reply = provider.complete("SYSTEM TEXT", "QUESTION AND CONTEXT", timeout=20)
    assert reply == "reply text"
    assert fake_sdk.init_kwargs["api_key"] == FAKE_KEY
    http = fake_sdk.init_kwargs["http_options"]
    assert http.timeout == 20000  # milliseconds
    assert http.retry_options.attempts == 1  # no SDK retries, so the 30 s limit holds
    call = fake_sdk.calls[0]
    assert call["model"] == FAKE_MODEL
    assert call["contents"] == "QUESTION AND CONTEXT"
    config = call["config"]
    assert config.system_instruction == "SYSTEM TEXT"
    assert config.temperature == 0
    assert config.max_output_tokens == 2048
    assert config.response_mime_type == "application/json"


def test_timeout_maps_to_timeout_error(fake_sdk):
    fake_sdk._outcome = httpx.ReadTimeout("slow")
    with pytest.raises(LLMTimeoutError):
        GeminiProvider(FAKE_KEY, FAKE_MODEL).complete("s", "u", timeout=5)


def test_api_error_maps_to_provider_error_without_leaking_the_key(fake_sdk):
    fake_sdk._outcome = errors.APIError(500, {"error": {"message": "internal"}})
    with pytest.raises(LLMError) as info:
        GeminiProvider(FAKE_KEY, FAKE_MODEL).complete("s", "u", timeout=5)
    assert "status 500" in str(info.value)
    assert FAKE_KEY not in str(info.value)


def test_network_error_maps_to_provider_error(fake_sdk):
    fake_sdk._outcome = httpx.ConnectError("refused")
    with pytest.raises(LLMError):
        GeminiProvider(FAKE_KEY, FAKE_MODEL).complete("s", "u", timeout=5)


def test_empty_reply_is_an_error(fake_sdk):
    fake_sdk._outcome = _Response("")
    with pytest.raises(LLMError, match="empty"):
        GeminiProvider(FAKE_KEY, FAKE_MODEL).complete("s", "u", timeout=5)


def test_blocked_reply_without_text_is_an_error(fake_sdk):
    class Blocked:
        @property
        def text(self):
            raise ValueError("no text: response was blocked")

    fake_sdk._outcome = Blocked()
    with pytest.raises(LLMError, match="no usable text"):
        GeminiProvider(FAKE_KEY, FAKE_MODEL).complete("s", "u", timeout=5)


def test_settings_read_gemini_variables(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", FAKE_KEY)
    monkeypatch.setenv("GEMINI_MODEL", FAKE_MODEL)
    settings = Settings.from_env()
    assert settings.gemini_api_key == FAKE_KEY
    assert settings.gemini_model == FAKE_MODEL


def test_startup_fails_closed_when_gemini_is_selected_without_a_key(tmp_path):
    settings = dataclasses.replace(
        Settings.from_env(),
        llm_provider="gemini",
        gemini_api_key=None,
        gemini_model=FAKE_MODEL,
        db_path=tmp_path / "missing.db",
        state_db=tmp_path / "state.db",
        vector_dir=tmp_path,
    )
    from fastapi.testclient import TestClient

    app = create_app(settings, embedder=None, provider=None, start_background=False)
    with TestClient(app) as client:
        assert client.get("/health").status_code == 503
        assert "GEMINI_API_KEY" in (app.state.startup_error or "")
