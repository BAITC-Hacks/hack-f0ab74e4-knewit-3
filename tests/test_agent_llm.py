"""Проверка запроса OpenAI-агента без обращения к внешнему API."""

from unittest.mock import patch

import pytest

from src.agent.llm import openai_chat_loop, pick_backend


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
