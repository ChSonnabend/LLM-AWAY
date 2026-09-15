from __future__ import annotations

import html
import json
import re
import shlex
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


def tool_name(tool: dict) -> str:
    if not isinstance(tool, dict):
        return ""
    if tool.get("name"):
        return str(tool["name"])
    function = tool.get("function")
    if isinstance(function, dict) and function.get("name"):
        return str(function["name"])
    if tool.get("type") in {"shell", "apply_patch"}:
        return str(tool["type"])
    return ""


def available_tool_names(tools) -> set[str]:
    if not isinstance(tools, list):
        return set()
    return {name for tool in tools if (name := tool_name(tool))}


def tools_to_prompt(tools) -> str:
    lines = [
        "Tool calling:",
        "Use only the tool names listed here. Emit tool calls as:",
        "<tool_call><function=tool_name><parameter=param_name>value</parameter></function></tool_call>",
        "Do not invent tools such as read_file unless they are listed. For file reads, directory listing, and search, prefer the shell tool with commands like cat, sed, ls, and rg.",
        "If the user asks to inspect, explain, diagnose, review, summarize, or tell what code does, use read-only commands and then answer; do not modify files.",
        "Only edit files when the user explicitly asks for a change. When editing, use apply_patch instead of shell heredocs or redirection.",
    ]

    names = available_tool_names(tools)
    if not names:
        lines.append("- exec_command: run a shell command. Parameters: cmd")
        return "\n".join(lines)

    for tool in tools:
        name = tool_name(tool)
        if not name:
            continue
        description = tool.get("description")
        function = tool.get("function")
        if isinstance(function, dict):
            description = description or function.get("description")
        params = tool.get("parameters")
        if isinstance(function, dict):
            params = params or function.get("parameters")
        rendered = f"- {name}"
        if description:
            rendered += f": {description}"
        if params:
            rendered += f" Parameters: {json.dumps(params, sort_keys=True)}"
        lines.append(rendered)
    return "\n".join(lines)


def responses_request_to_prompt(payload: dict) -> str:
    parts = []
    instructions = payload.get("instructions")
    if instructions:
        parts.append(f"instructions: {content_to_text(instructions)}")
    parts.append(tools_to_prompt(payload.get("tools", [])))
    parts.append(responses_input_to_prompt(payload.get("input", "")))
    return "\n\n".join(part for part in parts if part)


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


def tool_call_blocks(text: str) -> list[tuple[int, int, dict]]:
    blocks: list[tuple[int, int, dict]] = []
    for match in re.finditer(r"<tool_call>\s*(?P<body>.*?)\s*</tool_call>", text, flags=re.DOTALL):
        body = match.group("body").strip()
        if body.startswith("{"):
            call = parse_json_tool_call(body)
        else:
            call = parse_xml_tool_call(body)
        if call:
            blocks.append((match.start(), match.end(), call))

    if blocks:
        return blocks

    bare = re.search(
        r"<tool_call>\s*function=(?P<name>[A-Za-z_][\w.-]*)\s*(?P<body>.*?)(?:</function>|$)",
        text,
        flags=re.DOTALL,
    )
    if bare:
        blocks.append(
            (
                bare.start(),
                bare.end(),
                {"name": bare.group("name"), "arguments": parse_tool_parameters(bare.group("body"))},
            )
        )
    return blocks


def parse_tool_calls(text: str) -> list[dict]:
    return [call for _, _, call in tool_call_blocks(text)]


def visible_tool_text(text: str, blocks: list[tuple[int, int, dict]], display_calls: list[dict] | None = None) -> str:
    if not blocks:
        return text

    chunks = []
    offset = 0
    for start, end, _ in blocks:
        chunks.append(text[offset:start])
        offset = end
    chunks.append(text[offset:])
    visible = re.sub(r"\n{3,}", "\n\n", "".join(chunks)).strip()
    if visible:
        return visible

    calls = display_calls if display_calls is not None else [call for _, _, call in blocks]
    names = [call["name"] for call in calls]
    unique_names = list(dict.fromkeys(names))
    if len(blocks) == 1:
        return f"Calling {names[0]}."
    if len(unique_names) == 1:
        return f"Calling {unique_names[0]} ({len(blocks)} calls)."
    return f"Calling tools ({len(blocks)} calls): {', '.join(unique_names)}."


def message_output(text: str, item_id: str | None = None, content_id: str | None = None) -> dict:
    return {
        "id": item_id or f"msg_{uuid.uuid4().hex}",
        "type": "message",
        "status": "completed",
        "role": "assistant",
        "content": [
            {
                "id": content_id or f"out_{uuid.uuid4().hex}",
                "type": "output_text",
                "text": text,
                "annotations": [],
            }
        ],
    }


def parse_tool_call(text: str) -> dict | None:
    calls = parse_tool_calls(text)
    return calls[0] if calls else None


def shell_tool_name(allowed_names: set[str] | None = None) -> str:
    allowed_names = allowed_names or set()
    if "exec_command" in allowed_names:
        return "exec_command"
    if "exec" in allowed_names:
        return "exec"
    return "exec_command"


def command_tool_call(command: str, allowed_names: set[str] | None = None) -> dict:
    name = shell_tool_name(allowed_names)
    if name == "exec":
        return {"name": "exec", "arguments": {"command": command}}
    return {"name": "exec_command", "arguments": {"cmd": command}}


def rewrite_tool_call(call: dict, allowed_names: set[str] | None = None) -> dict:
    allowed_names = allowed_names or set()
    if call["name"] == "exec" and "exec" not in allowed_names:
        args = call.get("arguments", {})
        command = args.get("command") or args.get("cmd")
        if command:
            return command_tool_call(str(command), allowed_names)
    if not allowed_names or call["name"] in allowed_names:
        return call

    shell_name = shell_tool_name(allowed_names)
    if shell_name not in allowed_names and allowed_names:
        return call

    args = call.get("arguments", {})
    if call["name"] == "read_file" and args.get("path"):
        return command_tool_call(f"sed -n '1,240p' -- {shlex.quote(str(args['path']))}", allowed_names)
    if call["name"] in {"list_directory", "list_files"}:
        path = args.get("path") or args.get("directory") or "."
        return command_tool_call(f"ls -la -- {shlex.quote(str(path))}", allowed_names)
    if call["name"] in {"search_files", "grep"} and args.get("pattern"):
        path = args.get("path") or args.get("directory") or "."
        return command_tool_call(
            f"rg -n -- {shlex.quote(str(args['pattern']))} {shlex.quote(str(path))}",
            allowed_names,
        )

    return call


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
    tools=None,
) -> dict:
    response_id = response_id or f"resp_{uuid.uuid4().hex}"
    blocks = tool_call_blocks(text)
    if blocks:
        allowed_names = available_tool_names(tools) or {"exec_command"}
        rewritten_calls = [rewrite_tool_call(tool_call, allowed_names) for _, _, tool_call in blocks]
        output_text = visible_tool_text(text, blocks, rewritten_calls)
        output = [message_output(output_text, item_id=item_id, content_id=content_id)] if output_text else []
        output.extend(
            [
                {
                    "id": f"fc_{uuid.uuid4().hex}",
                    "type": "function_call",
                    "status": "completed",
                    "call_id": f"call_{uuid.uuid4().hex}",
                    "name": rewritten["name"],
                    "arguments": json.dumps(rewritten["arguments"]),
                }
                for rewritten in rewritten_calls
            ]
        )
        return {
            "id": response_id,
            "object": "response",
            "created_at": int(time.time()),
            "status": "completed",
            "model": model,
            "output_text": output_text,
            "output": output,
            "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
        }

    return {
        "id": response_id,
        "object": "response",
        "created_at": int(time.time()),
        "status": "completed",
        "model": model,
        "output_text": text,
        "output": [message_output(text, item_id=item_id, content_id=content_id)],
        "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
    }
