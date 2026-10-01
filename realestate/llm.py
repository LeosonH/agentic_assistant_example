"""Thin wrapper around the Claude API. Returns None whenever the caller should fall back
to the offline template answer (LLM disabled, API error, or refusal)."""

import logging
import os

import anthropic

log = logging.getLogger(__name__)

MODEL = os.environ.get("REA_MODEL", "claude-opus-5-5")
MAX_HISTORY_MESSAGES = 10

_client: anthropic.Anthropic | None = None


def llm_enabled() -> bool:
    """REA_LLM=on|off|auto (default auto: on when Anthropic credentials are in the environment)."""
    mode = os.environ.get("REA_LLM", "auto").lower()
    if mode in ("on", "off"):
        return mode == "on"
    return bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))


def get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic(timeout=60.0)
    return _client


def clean_history(history: list[dict]) -> list[dict]:
    """Keep only well-formed user/assistant text turns, starting with a user turn."""
    turns = [
        {"role": m["role"], "content": m["content"]}
        for m in history
        if m.get("role") in ("user", "assistant") and isinstance(m.get("content"), str) and m["content"].strip()
    ][-MAX_HISTORY_MESSAGES:]
    while turns and turns[0]["role"] != "user":
        turns.pop(0)
    return turns


def complete(system: str, user_message: str, history: list[dict] | None = None) -> str | None:
    if not llm_enabled():
        return None

    messages = clean_history(history or []) + [{"role": "user", "content": user_message}]
    try:
        response = get_client().beta.messages.create(
            model=MODEL,
            max_tokens=16000,
            system=system,
            messages=messages,
            output_config={"effort": "low"},  # short, grounded chat answers
            # If a safety classifier declines, the API retries on a fallback model in the same call.
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )
    except anthropic.RateLimitError:
        log.warning("Claude rate limit hit; using offline answer")
        return None
    except anthropic.APIStatusError as e:
        log.warning("Claude API error %s; using offline answer", e.status_code)
        return None
    except anthropic.APIConnectionError:
        log.warning("Could not reach the Claude API; using offline answer")
        return None

    if response.stop_reason == "refusal":
        return None
    text = "".join(block.text for block in response.content if block.type == "text").strip()
    return text or None
