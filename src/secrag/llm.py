"""Thin client for any OpenAI-compatible chat endpoint (vLLM, Ollama, hosted APIs)."""
from __future__ import annotations

import json
import re
from typing import Any, Protocol

_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


class ChatModel(Protocol):
    def chat_json(self, system: str, user: str, max_tokens: int | None = None) -> Any: ...


def parse_json(text: str) -> Any:
    t = _FENCE_RE.sub("", text.strip()).strip()
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        pass
    starts = [i for i in (t.find("{"), t.find("[")) if i != -1]
    if not starts:
        raise ValueError(f"no JSON found in model output: {text[:200]!r}")
    s = min(starts)
    e = t.rfind("}" if t[s] == "{" else "]")
    if e <= s:
        raise ValueError(f"unterminated JSON in model output: {text[:200]!r}")
    return json.loads(t[s : e + 1])


class OpenAICompatLLM:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        timeout: float = 180.0,
    ):
        from openai import OpenAI

        self.client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._json_mode = True
        self.calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def chat(self, system: str, user: str, max_tokens: int | None = None, json_mode: bool = False,
             temperature: float | None = None) -> str:
        from openai import BadRequestError

        kwargs: dict[str, Any] = dict(
            model=self.model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=self.temperature if temperature is None else temperature,
            max_tokens=max_tokens or self.max_tokens,
        )
        if json_mode and self._json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        try:
            resp = self.client.chat.completions.create(**kwargs)
        except BadRequestError:
            if "response_format" not in kwargs:
                raise
            self._json_mode = False  # server doesn't support it; fall back to prompt-only JSON
            kwargs.pop("response_format")
            resp = self.client.chat.completions.create(**kwargs)
        self.calls += 1
        if resp.usage:
            self.prompt_tokens += resp.usage.prompt_tokens or 0
            self.completion_tokens += resp.usage.completion_tokens or 0
        return resp.choices[0].message.content or ""

    def chat_json(self, system: str, user: str, max_tokens: int | None = None) -> Any:
        from openai import APIStatusError

        last: Exception | None = None
        temperature = self.temperature
        for attempt in range(2):
            prompt = user if attempt == 0 else user + "\n\nYour previous reply was not valid JSON. Reply with ONLY a JSON object."
            try:
                text = self.chat(system, prompt, max_tokens, json_mode=True, temperature=temperature)
                return parse_json(text)
            except (ValueError, APIStatusError) as e:
                # Greedy decoding (temperature 0) can get stuck in a repetition loop; servers
                # such as Ollama then abort with a 500. Retrying with a little randomness unsticks it.
                # Passed per call (not stored on self), so concurrent eval workers don't interfere.
                last = e
                temperature = max(self.temperature, 0.4)
        assert last is not None
        raise last
