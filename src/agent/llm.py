"""Переключаемые LLM-бэкенды агента.

- anthropic: Claude через Anthropic API (ANTHROPIC_API_KEY)
- openai-совместимый: любой эндпоинт с tool calling — OpenAI, NVIDIA NIM
  (LLM_BASE_URL, LLM_API_KEY, LLM_MODEL). NVIDIA: base_url=https://integrate.api.nvidia.com/v1

Оба гоняют один и тот же цикл над одними тулами; выбор — деталь среды, не архитектуры.
"""
from __future__ import annotations

import json
import os
import time


def anthropic_chat_loop(system: str, tool_defs: list[dict], user_msg: str,
                        run_tool, max_steps: int = 16) -> str | None:
    """Возвращает None, если агент так и не вызвал write_outputs."""
    import anthropic
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    model = os.environ.get("LLM_MODEL", "claude-sonnet-5")
    messages = [{"role": "user", "content": user_msg}]
    for _ in range(max_steps):
        resp = client.messages.create(model=model, max_tokens=2500, system=system,
                                      tools=tool_defs, messages=messages)
        calls = [b for b in resp.content if b.type == "tool_use"]
        if not calls:
            return None
        messages.append({"role": "assistant", "content": resp.content})
        results, done = [], False
        for call in calls:
            out = run_tool(call.name, call.input)
            done = done or call.name == "write_outputs"
            results.append({"type": "tool_result", "tool_use_id": call.id,
                            "content": json.dumps(out, ensure_ascii=False)})
        messages.append({"role": "user", "content": results})
        if done:
            return "ok"
    return None


def _openai_post(base: str, key: str, payload: dict, retries: int = 3) -> dict:
    """POST с повтором на 429/5xx. Лимит токенов называется по-разному в поколениях
    моделей (max_tokens у GPT-4-линейки, max_completion_tokens у новых) — пробуем оба."""
    import httpx
    last = None
    for attempt in range(retries):
        r = httpx.post(f"{base}/chat/completions",
                       headers={"Authorization": f"Bearer {key}"},
                       json=payload, timeout=180)
        if r.status_code == 400 and "max_tokens" in r.text and "max_completion_tokens" in r.text:
            payload = {k: v for k, v in payload.items() if k != "max_tokens"}
            payload["max_completion_tokens"] = 2500
            continue
        if r.status_code == 429 or r.status_code >= 500:
            last = r
            time.sleep(2 ** attempt)
            continue
        r.raise_for_status()
        return r.json()
    if last is not None:
        last.raise_for_status()
    raise RuntimeError("не удалось получить ответ от LLM")


def openai_chat_loop(system: str, tool_defs: list[dict], user_msg: str,
                     run_tool, max_steps: int = 16) -> str | None:
    """OpenAI-совместимый эндпоинт (OpenAI / NVIDIA NIM) через httpx, без SDK."""
    base = os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    key = os.environ["LLM_API_KEY"]
    model = os.environ.get("LLM_MODEL", "gpt-4.1-mini")
    oa_tools = [{"type": "function",
                 "function": {"name": t["name"], "description": t["description"],
                              "parameters": t["input_schema"]}} for t in tool_defs]
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": user_msg}]
    for _ in range(max_steps):
        data = _openai_post(base, key, {"model": model, "messages": messages,
                                        "tools": oa_tools, "max_tokens": 2500})
        msg = data["choices"][0]["message"]
        calls = msg.get("tool_calls") or []
        if not calls:
            return None
        messages.append(msg)
        done = False
        for call in calls:
            args = json.loads(call["function"]["arguments"] or "{}")
            out = run_tool(call["function"]["name"], args)
            done = done or call["function"]["name"] == "write_outputs"
            messages.append({"role": "tool", "tool_call_id": call["id"],
                             "content": json.dumps(out, ensure_ascii=False)})
        if done:
            return "ok"
    return None


def pick_backend() -> str:
    """Явный LLM_BACKEND > LLM_API_KEY > ANTHROPIC_API_KEY > none.

    LLM_API_KEY выигрывает намеренно: это имя задаётся только в .env проекта, тогда как
    ANTHROPIC_API_KEY часто висит в окружении разработчика от сторонних инструментов —
    иначе настроенный OpenAI-ключ молча игнорировался бы.
    """
    explicit = os.environ.get("LLM_BACKEND", "").strip().lower()
    if explicit in ("openai", "anthropic", "none"):
        return explicit
    if os.environ.get("LLM_API_KEY"):
        return "openai"
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic"
    return "none"
