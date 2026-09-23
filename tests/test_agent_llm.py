"""Проверка запроса OpenAI-агента без обращения к внешнему API."""

from unittest.mock import patch
from types import SimpleNamespace

import pytest

from src.agent.llm import openai_chat_loop, pick_backend
from src.agent.llm import anthropic_chat_loop


@pytest.mark.parametrize("key_name", ["LLM_API_KEY", "OPENAI_API_KEY"])
def test_luna_executes_tool_cycle_with_supported_chat_parameters(key_name):
    requests = []
    executed = []

    class Response:
        status_code = 200

        def __init__(self, message):
            self.message = message

        def raise_for_status(self):
            pass

        def json(self):
            return {"choices": [{"message": self.message}]}

    def post(url, headers, json, timeout):
        requests.append(json)
        assert url == "https://api.openai.com/v1/chat/completions"
        assert headers == {"Authorization": "Bearer test-key"}
        if len(requests) == 1:
            message = {"role": "assistant", "content": None, "tool_calls": [{
                "id": "call-1", "type": "function",
                "function": {"name": "write_outputs", "arguments": '{"analysis":"Готово"}'},
            }]}
        else:
            raise AssertionError("Агент сделал лишний запрос после write_outputs")
        return Response(message)

    def run_tool(name, args):
        executed.append((name, args))
        return {"saved": True}

    tools = [{"name": "write_outputs", "description": "Сохранить отчёт",
              "input_schema": {"type": "object", "properties": {
                  "analysis": {"type": "string"}}, "required": ["analysis"]}}]
    with patch.dict("os.environ", {key_name: "test-key", "LLM_BASE_URL": "https://api.openai.com/v1"}, clear=True), \
         patch("httpx.post", side_effect=post):
        assert pick_backend() == "openai"
        result = openai_chat_loop("План", tools, "Запусти день", run_tool)

    assert result == "ok"
    assert executed == [("write_outputs", {"analysis": "Готово"})]
    assert requests[0]["model"] == "gpt-6-luna"
    assert requests[0]["reasoning_effort"] == "none"
    assert requests[0]["max_completion_tokens"] == 2500
    assert "max_tokens" not in requests[0]


@pytest.mark.parametrize("backend", ["openai", "anthropic"])
def test_adapters_reject_batches_and_continue_after_rejected_write(monkeypatch, backend):
    """Сначала пакет, затем преждевременная запись, затем успешная запись."""
    requests, executions = [], []
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    def run_tool(name, args):
        executions.append(name)
        return {"error": "ещё не выполнена валидация"} if len(executions) == 1 else {"saved": True}

    def response():
        requests.append(True)
        names = ["fetch_weather", "write_outputs"] if len(requests) == 1 else ["write_outputs"]
        return names

    if backend == "openai":
        def post(base, key, payload):
            names = response()
            calls = [{"id": str(i), "type": "function", "function": {
                "name": name, "arguments": '{"analysis":"Отчёт"}'}}
                for i, name in enumerate(names)]
            return {"choices": [{"message": {"role": "assistant", "tool_calls": calls}}]}
        monkeypatch.setattr("src.agent.llm._openai_post", post)
        run = openai_chat_loop
    else:
        def create(**kwargs):
            calls = [SimpleNamespace(type="tool_use", id=str(i), name=name,
                                     input={"analysis": "Отчёт"}) for i, name in enumerate(response())]
            return SimpleNamespace(content=calls)
        monkeypatch.setattr("anthropic.Anthropic", lambda **kwargs:
                            SimpleNamespace(messages=SimpleNamespace(create=create)))
        run = anthropic_chat_loop

    assert run("Система", [], "Запуск", run_tool, max_steps=4) == "ok"
    assert len(requests) == 3
    assert executions == ["write_outputs", "write_outputs"]
