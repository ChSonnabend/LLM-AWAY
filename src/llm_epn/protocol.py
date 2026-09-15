from __future__ import annotations

import html
import json
import re
import time
import uuid


def models_list(model: str) -> dict:
    return {
        "object": "list",
        "data": [
            {
                "id": model,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "llm-epn",
            }
        ],
    }


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


def content_to_text(content) -> str:
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") in {"text", "input_text", "output_text"}:
                parts.append(str(item.get("text", "")))
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts)
    return str(content)


def responses_input_to_prompt(value) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        lines = []
        for item in value:
            if isinstance(item, dict) and "role" in item:
                role = item.get("role", "user")
                lines.append(f"{role}: {content_to_text(item.get('content', ''))}")
            elif isinstance(item, dict) and item.get("type") in {"message", "input_text"}:
                role = item.get("role", "user")
                lines.append(f"{role}: {content_to_text(item.get('content', item.get('text', '')))}")
            elif isinstance(item, dict) and item.get("type") == "function_call":
                lines.append(f"assistant tool_call {item.get('name', '')}: {item.get('arguments', '')}")
            elif isinstance(item, dict) and item.get("type") == "function_call_output":
                lines.append(f"tool {item.get('call_id', '')}: {content_to_text(item.get('output', ''))}")
        if lines:
            lines.append("assistant:")
            return "\n".join(lines)
        return str(value)
    return str(value)


def normalize_tool_arguments(arguments) -> dict:
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            return {"input": arguments}
    return arguments if isinstance(arguments, dict) else {"input": arguments}


def parse_tool_parameters(body: str) -> dict:
    params = {}
    for param_match in re.finditer(
        r"<parameter=(?P<name>[A-Za-z_][\w.-]*)>(?P<value>.*?)</parameter>",
        body,
        flags=re.DOTALL,
    ):
        params[param_match.group("name")] = html.unescape(param_match.group("value").strip())
    if not params:
        params["input"] = html.unescape(body.strip())
    return params


def parse_json_tool_call(body: str) -> dict | None:
    try:
        data = json.loads(body)
    except json.JSONDecodeError:
        return None
    name = data.get("name") or data.get("function")
    arguments = data.get("arguments") or data.get("parameters") or {}
    if not name:
        return None
    return {"name": str(name), "arguments": normalize_tool_arguments(arguments)}


def parse_xml_tool_call(body: str) -> dict | None:
    nested = re.search(
        r"<function=(?P<name>[A-Za-z_][\w.-]*)>\s*(?P<body>.*?)</function>",
        body,
        flags=re.DOTALL,
    )
    if nested:
        return {"name": nested.group("name"), "arguments": parse_tool_parameters(nested.group("body"))}

    inline = re.search(
        r"function=(?P<name>[A-Za-z_][\w.-]*)\s*(?P<body>.*?)(?:</function>)?\s*$",
        body,
        flags=re.DOTALL,
    )
    if inline:
        return {"name": inline.group("name"), "arguments": parse_tool_parameters(inline.group("body"))}

    return None


def parse_tool_calls(text: str) -> list[dict]:
    calls: list[dict] = []
    for match in re.finditer(r"<tool_call>\s*(?P<body>.*?)\s*</tool_call>", text, flags=re.DOTALL):
        body = match.group("body").strip()
        if body.startswith("{"):
            call = parse_json_tool_call(body)
        else:
            call = parse_xml_tool_call(body)
        if call:
            calls.append(call)

    if calls:
        return calls

    bare = re.search(
        r"<tool_call>\s*function=(?P<name>[A-Za-z_][\w.-]*)\s*(?P<body>.*?)(?:</function>|$)",
        text,
        flags=re.DOTALL,
    )
    if bare:
        calls.append({"name": bare.group("name"), "arguments": parse_tool_parameters(bare.group("body"))})
    return calls


def parse_tool_call(text: str) -> dict | None:
    calls = parse_tool_calls(text)
    return calls[0] if calls else None


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
    tool_calls = parse_tool_calls(text)
    if tool_calls:
        return {
            "id": response_id,
            "object": "response",
            "created_at": int(time.time()),
            "status": "completed",
            "model": model,
            "output_text": "",
            "output": [
                {
                    "id": f"fc_{uuid.uuid4().hex}",
                    "type": "function_call",
                    "status": "completed",
                    "call_id": f"call_{uuid.uuid4().hex}",
                    "name": tool_call["name"],
                    "arguments": json.dumps(tool_call["arguments"]),
                }
                for tool_call in tool_calls
            ],
            "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
        }

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
