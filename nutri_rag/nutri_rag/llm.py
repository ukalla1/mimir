"""Thin OpenAI-compatible chat client using requests.

Used by the standalone pipeline and demo scripts.
The lm-evaluation-harness integration uses its own model backend.
"""

from __future__ import annotations

import json
import requests

from nutri_rag.config import LLM_BASE_URL, LLM_MODEL


def chat_completion(
    messages: list[dict[str, str]],
    *,
    temperature: float = 0.0,
    max_tokens: int = 4096,
    base_url: str = LLM_BASE_URL,
    model: str = LLM_MODEL,
) -> str:
    """Send a chat completion request and return the assistant reply."""
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    resp = requests.post(base_url, json=payload, timeout=120)
    resp.raise_for_status()

    data = resp.json()
    content = data["choices"][0]["message"]["content"]

    # Strip Qwen3.5 <think>...</think> blocks if present
    import re as _re
    content = _re.sub(r"<think>.*?</think>", "", content, flags=_re.DOTALL).strip()

    return content


def chat_completion_json(
    messages: list[dict[str, str]],
    **kwargs,
) -> dict:
    """Send a chat completion and parse the reply as JSON.

    Tries to extract JSON from the response even if the model wraps it
    in markdown code fences or adds surrounding text.
    """
    raw = chat_completion(messages, **kwargs)

    # Try direct parse
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Try extracting from code fences
    import re
    m = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", raw, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass

    # Try finding first { ... } block
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass

    # Handle truncated JSON (missing closing braces) — common with LLM output
    first_brace = raw.find("{")
    if first_brace >= 0:
        fragment = raw[first_brace:]
        # Count unmatched braces and append closing ones
        depth = 0
        for ch in fragment:
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
        if depth > 0:
            fragment += "}" * depth
            try:
                return json.loads(fragment)
            except json.JSONDecodeError:
                pass

    raise ValueError(f"Could not parse JSON from LLM response:\n{raw}")
