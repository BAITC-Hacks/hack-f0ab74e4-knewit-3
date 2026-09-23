"""Переключаемые LLM-бэкенды агента.

- anthropic: Claude через Anthropic API (ANTHROPIC_API_KEY)
- openai-совместимый: любой эндпоинт с tool calling — OpenAI, NVIDIA NIM
  (LLM_BASE_URL, OPENAI_API_KEY или LLM_API_KEY, LLM_MODEL).
  NVIDIA: base_url=https://integrate.api.nvidia.com/v1

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
    """POST с повтором на 429/5xx и правкой несовместимых параметров.

    Правка параметра не расходует попытки повтора: это разовая подстройка под поколение
    модели, а не сбой. Лимит токенов называется по-разному (max_tokens у GPT-4-линейки,
    max_completion_tokens у новых), а reasoning_effort принимают не все эндпоинты.
    """
    import httpx
    last, attempt, fixes = None, 0, 0
    while attempt < retries:
        r = httpx.post(f"{base}/chat/completions",
                       headers={"Authorization": f"Bearer {key}"},
                       json=payload, timeout=180)
        if r.status_code == 400 and fixes < 2:
            body = r.text
            if "max_tokens" in body and "max_completion_tokens" in body:
                payload.pop("max_tokens", None)
                payload["max_completion_tokens"] = 2500
                fixes += 1
                continue
            if "reasoning_effort" in body and "reasoning_effort" in payload:
                payload.pop("reasoning_effort")
                fixes += 1
                continue
        if r.status_code == 429 or r.status_code >= 500:
            last = r
            time.sleep(2 ** attempt)
            attempt += 1
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
    key = os.environ.get("LLM_API_KEY") or os.environ["OPENAI_API_KEY"]
    model = os.environ.get("LLM_MODEL", "gpt-6-luna")
    oa_tools = [{"type": "function",
                 "function": {"name": t["name"], "description": t["description"],
                              "parameters": t["input_schema"]}} for t in tool_defs]
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": user_msg}]
    for step in range(max_steps):
        payload = {"model": model, "messages": messages, "tools": oa_tools}
        if model.startswith(("gpt-6-luna", "gpt-6-sol")):
            # У GPT-6 в Chat Completions тулы доступны только без reasoning.
            payload.update(reasoning_effort="none", max_completion_tokens=2500)
        else:
            payload["max_tokens"] = 2500
        data = _openai_post(base, key, payload)
        msg = data["choices"][0]["message"]
        calls = msg.get("tool_calls") or []
        if not calls:
            if step == 0:
                # Модель не вызвала ни одного тула на первом шаге — почти наверняка
                # tool calling не работает на этой конфигурации (см. reasoning_effort
                # у GPT-6 в Chat Completions). Молчать нельзя: день тихо уйдёт в --no-llm.
                print(f"    ВНИМАНИЕ: модель {model} не вызвала тулы на первом шаге. "
                      f"Ответ: {str(msg.get('content'))[:160]!r}")
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
    """Явный LLM_BACKEND > ключ OpenAI > ANTHROPIC_API_KEY > none.

    Ключ OpenAI выигрывает намеренно: он задаётся в .env проекта, тогда как
    ANTHROPIC_API_KEY часто висит в окружении разработчика от сторонних инструментов —
    иначе настроенный OpenAI-ключ молча игнорировался бы.
    """
    explicit = os.environ.get("LLM_BACKEND", "").strip().lower()
    if explicit in ("openai", "anthropic", "none"):
        return explicit
    if os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY"):
        return "openai"
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic"
    return "none"
