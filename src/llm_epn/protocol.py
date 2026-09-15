from __future__ import annotations

import time
import uuid


def messages_to_prompt(messages: list[dict]) -> str:
    lines: list[str] = []
    for message in messages:
        role = message.get("role", "user")
        content = message.get("content", "")
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") in {"text", "input_text"}:
                    parts.append(str(item.get("text", "")))
                elif isinstance(item, str):
                    parts.append(item)
            content = "\n".join(parts)
        lines.append(f"{role}: {content}")
    lines.append("assistant:")
    return "\n".join(lines)


def responses_input_to_prompt(value) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        messages = []
        for item in value:
            if isinstance(item, dict) and "role" in item:
                messages.append(item)
            elif isinstance(item, dict) and item.get("type") in {"message", "input_text"}:
                messages.append({"role": item.get("role", "user"), "content": item.get("content", item.get("text", ""))})
        return messages_to_prompt(messages) if messages else str(value)
    return str(value)


def chat_completion(model: str, text: str) -> dict:
    created = int(time.time())
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


def chat_completion_chunk(model: str, text: str, finish: bool = False) -> dict:
    created = int(time.time())
    choice: dict
    if finish:
        choice = {"index": 0, "delta": {}, "finish_reason": "stop"}
    else:
        choice = {"index": 0, "delta": {"content": text}, "finish_reason": None}
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [choice],
    }


def response_object(
    model: str,
    text: str,
    response_id: str | None = None,
    item_id: str | None = None,
    content_id: str | None = None,
) -> dict:
    response_id = response_id or f"resp_{uuid.uuid4().hex}"
    item_id = item_id or f"msg_{uuid.uuid4().hex}"
    content_id = content_id or f"out_{uuid.uuid4().hex}"
    return {
        "id": response_id,
        "object": "response",
        "created_at": int(time.time()),
        "status": "completed",
        "model": model,
        "output_text": text,
        "output": [
            {
                "id": item_id,
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [
                    {
                        "id": content_id,
                        "type": "output_text",
                        "text": text,
                        "annotations": [],
                    }
                ],
            }
        ],
        "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
    }
