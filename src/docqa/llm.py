"""Groq LLM client: chat completion, JSON completion, language detect, translate."""
import json
from groq import Groq
from langdetect import detect, LangDetectException
from .config import get_groq_key, GROQ_MODEL


class LLMError(Exception):
    """Raised when the LLM call fails or returns unparseable output."""


_client: Groq | None = None


def _get_client() -> Groq:
    global _client
    if _client is None:
        _client = Groq(api_key=get_groq_key())
    return _client


def detect_language(text: str) -> str:
    """Best-effort ISO 639-1 language code; defaults to English on failure."""
    try:
        return detect(text)
    except LangDetectException:
        return "en"


def complete(system: str, user: str, temperature: float = 0.1) -> str:
    try:
        resp = _get_client().chat.completions.create(
            model=GROQ_MODEL,
            temperature=temperature,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
        )
        return resp.choices[0].message.content or ""
    except Exception as e:  # groq raises various API errors; surface uniformly.
        raise LLMError(f"LLM request failed: {e}") from e


def complete_json(system: str, user: str) -> dict:
    try:
        resp = _get_client().chat.completions.create(
            model=GROQ_MODEL,
            temperature=0.1,
            response_format={"type": "json_object"},
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
        )
        content = resp.choices[0].message.content or "{}"
        return json.loads(content)
    except json.JSONDecodeError as e:
        raise LLMError(f"LLM returned invalid JSON: {e}") from e
    except Exception as e:
        raise LLMError(f"LLM request failed: {e}") from e


def translate(text: str, target_lang: str) -> str:
    """Translate text into target_lang (ISO 639-1). Returns text unchanged for
    empty input."""
    if not text.strip():
        return text
    system = (
        "You are a professional translator. Translate the user's text into the "
        f"language with ISO 639-1 code '{target_lang}'. Preserve meaning and "
        "any quoted phrases. Output ONLY the translation, no preamble."
    )
    return complete(system, text)
