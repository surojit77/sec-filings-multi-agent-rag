"""Real OpenAICompatLLM against a mocked transport (no server needed)."""
import httpx
import pytest
from openai import APIStatusError

from secrag.llm import OpenAICompatLLM, parse_json


def make_llm(chat):
    llm = OpenAICompatLLM.__new__(OpenAICompatLLM)
    llm.temperature = 0.0
    llm.chat = chat
    return llm


def test_retry_on_server_error_raises_temperature_without_mutating_state():
    temps = []

    def chat(system, user, max_tokens=None, json_mode=False, temperature=None):
        temps.append(temperature)
        if len(temps) == 1:
            raise APIStatusError("loop", response=httpx.Response(500, request=httpx.Request("POST", "http://x")),
                                 body=None)
        return '{"ok": 1}'

    llm = make_llm(chat)
    assert llm.chat_json("s", "u") == {"ok": 1}
    assert temps == [0.0, 0.4] and llm.temperature == 0.0


def test_parse_json_variants():
    assert parse_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json('Sure! {"a": [1, 2]} hope that helps') == {"a": [1, 2]}
    with pytest.raises(ValueError):
        parse_json("no json here")


def test_real_client_constructs():
    llm = OpenAICompatLLM("http://localhost:1/v1", "EMPTY", "m")
    assert llm.model == "m" and llm.temperature == 0.0
