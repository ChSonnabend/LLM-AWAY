from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import signal
import sys
import threading
import time
import uuid

from .backends import Backend, BackendError, InferenceRequest
from .config import AppConfig
from .protocol import (
    chat_completion,
    chat_completion_chunk,
    messages_to_prompt,
    models_list,
    normalize_chat_messages,
    response_object,
    responses_request_to_prompt,
    responses_request_to_messages,
)


class ProviderHandler(BaseHTTPRequestHandler):
    backend: Backend
    config: AppConfig

    def do_GET(self) -> None:
        if self.path == "/health":
            self.write_json({"status": "ok", "model": self.config.model.name})
            return
        if self.path == "/v1/models":
            self.write_json(models_list("away"))
            return
        self.write_json({"error": "not found"}, status=404)

    def do_POST(self) -> None:
        try:
            self._do_POST()
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            # Escape/Ctrl+C in Codex closes this request, not the gateway.
            self.close_connection = True

    def _do_POST(self) -> None:
        try:
            payload = self.read_json()
            if self.path == "/v1/chat/completions":
                self.handle_chat(payload)
            elif self.path == "/v1/responses":
                self.handle_responses(payload)
            else:
                self.write_json({"error": "not found"}, status=404)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            raise
        except BackendError as exc:
            self.write_json({"error": {"message": str(exc), "type": "backend_error"}}, status=502)
        except Exception as exc:
            self.write_json({"error": {"message": str(exc), "type": "server_error"}}, status=500)

    def handle_chat(self, payload: dict) -> None:
        model = payload.get("model") or self.config.model.name
        raw_prompt = messages_to_prompt(payload.get("messages", []))
        prompt = self.compact_prompt(raw_prompt)
        messages = self.compact_messages(normalize_chat_messages(payload.get("messages", [])))
        text = self.backend.infer(InferenceRequest(prompt=prompt, model=model, raw_prompt_chars=len(raw_prompt), messages=messages))
        if payload.get("stream"):
            self.write_chat_stream(model, text)
            return
        self.write_json(chat_completion(model, text))

    def handle_responses(self, payload: dict) -> None:
        model = payload.get("model") or self.config.model.name
        raw_prompt = responses_request_to_prompt(payload)
        prompt = self.compact_prompt(raw_prompt)
        messages = self.compact_messages(responses_request_to_messages(payload))
        if payload.get("stream"):
            self.handle_responses_stream(model, prompt, len(raw_prompt), payload.get("tools", []), messages)
            return
        response = self.infer_response(model, prompt, len(raw_prompt), payload.get("tools", []), messages)
        self.write_json(response)

    def read_json(self) -> dict:
        length = int(self.headers.get("content-length", "0"))
        raw = self.rfile.read(length).decode("utf-8")
        return json.loads(raw) if raw else {}

    def compact_messages(self, messages: list[dict]) -> list[dict]:
        """Apply the head/tail budget without losing message role boundaries."""
        limit = self.config.gateway.max_prompt_chars
        total = sum(len(message["content"]) for message in messages)
        if limit <= 0 or total <= limit:
            return messages
        marker = "\n[llm-away: earlier context omitted]\n"
        if limit <= 2 * len(marker):
            return [{**messages[-1], "content": messages[-1]["content"][-limit:]}]
        # At most two boundary messages need a truncation marker.
        available = max(0, limit - 2 * len(marker))
        latest_size = min(len(messages[-1]["content"]), available)
        head = min(max(0, self.config.gateway.prompt_keep_head_chars), available - latest_size)
        tail_start = total - (available - head)
        compacted = []
        offset = 0
        for message in messages:
            content = message["content"]
            end = offset + len(content)
            prefix = content[:max(0, head - offset)] if offset < head else ""
            suffix = content[max(0, tail_start - offset):] if end > tail_start else ""
            kept = prefix + suffix
            if kept:
                if len(kept) < len(content):
                    kept = prefix + marker + suffix
                compacted.append({**message, "content": kept})
            offset = end
        return compacted

    def compact_prompt(self, prompt: str) -> str:
        limit = self.config.gateway.max_prompt_chars
        if limit <= 0 or len(prompt) <= limit:
            return prompt

        head_chars = max(0, self.config.gateway.prompt_keep_head_chars)
        tail_chars = max(0, self.config.gateway.prompt_keep_tail_chars)
        if head_chars + tail_chars >= limit:
            tail_chars = max(0, limit - head_chars)
        omitted = len(prompt) - head_chars - tail_chars
        marker = f"\n\n[llm-away: compacted {omitted} characters from the middle of the Codex context]\n\n"
        if len(marker) >= limit:
            return prompt[-limit:]

        available = limit - len(marker)
        kept_head = min(head_chars, available)
        kept_tail = max(0, available - kept_head)
        return prompt[:kept_head] + marker + (prompt[-kept_tail:] if kept_tail else "")

    def write_json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def begin_sse(self) -> None:
        self.close_connection = True
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.send_header("cache-control", "no-cache")
        self.send_header("connection", "close")
        self.end_headers()

    def write_sse(self, event: str, payload: dict | str) -> None:
        self.wfile.write(f"event: {event}\n".encode("utf-8"))
        data = payload if isinstance(payload, str) else json.dumps(payload)
        self.wfile.write(f"data: {data}\n\n".encode("utf-8"))
        self.wfile.flush()

    def write_chat_stream(self, model: str, text: str) -> None:
        self.begin_sse()
        self.write_sse("message", chat_completion_chunk(model, "", finish=False))
        if text:
            self.write_sse("message", chat_completion_chunk(model, text, finish=False))
        self.write_sse("message", chat_completion_chunk(model, "", finish=True))
        self.write_sse("message", "[DONE]")

    def infer_response(self, model, prompt, raw_prompt_chars=None, tools=None, messages=None, response_id=None):
        """Allow one format correction; never execute or guess malformed commands."""
        request = InferenceRequest(prompt=prompt, model=model, raw_prompt_chars=raw_prompt_chars, messages=messages)
        for attempt in range(2):
            text = self.backend.infer(request)
            try:
                return response_object(model, text, response_id=response_id, tools=tools)
            except ValueError as exc:
                if attempt:
                    return response_object(
                        model, "The model produced an invalid tool call twice. No tool from these attempts was executed. "
                        + str(exc).rstrip(".") + ".", response_id=response_id,
                    )
                correction = (
                    "Your previous tool call was rejected before execution: " + str(exc) + "\n"
                    "Regenerate the intended next call using only a listed tool and its required parameters. "
                    "Use <tool_call><function=NAME><parameter=KEY>literal value</parameter></function></tool_call>. "
                    "Put shell commands or patches directly in XML parameter text: no JSON string wrapper, "
                    "no extra escaping of quotes. Keep shell commands short. Close all tags, then stop."
                )
                previous = text[:6000] + ("\n[invalid output truncated]" if len(text) > 6000 else "")
                retry_messages = None if messages is None else [
                    *messages, {"role": "assistant", "content": previous},
                    {"role": "user", "content": correction},
                ]
                request = InferenceRequest(
                    prompt=prompt + "\nassistant: " + previous + "\nuser: " + correction + "\nassistant:",
                    model=model, raw_prompt_chars=raw_prompt_chars, messages=retry_messages,
                )

    def handle_responses_stream(
        self, model: str, prompt: str, raw_prompt_chars: int | None = None, tools=None, messages=None
    ) -> None:
        response_id = f"resp_{uuid.uuid4().hex}"
        created = {
            "id": response_id,
            "object": "response",
            "created_at": int(time.time()),
            "status": "in_progress",
            "model": model,
            "output": [],
            "output_text": "",
            "usage": None,
        }

        self.begin_sse()
        self.write_sse(
            "response.created",
            {"type": "response.created", "sequence_number": 0, "response": created},
        )
        try:
            response = self.infer_response(model, prompt, raw_prompt_chars, tools, messages, response_id)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            raise
        except Exception as exc:
            failed = dict(created)
            failed["status"] = "failed"
            failed["error"] = {"message": str(exc), "code": "server_error"}
            self.write_sse(
                "response.failed",
                {"type": "response.failed", "sequence_number": 1, "response": failed},
            )
            return
        self.write_response_stream(response, sequence_number=1, include_created=False)

    def write_response_stream(self, response: dict, sequence_number: int = 0, include_created: bool = True) -> None:
        def event_payload(event: str, payload: dict) -> dict:
            nonlocal sequence_number
            payload = {"type": event, "sequence_number": sequence_number, **payload}
            sequence_number += 1
            return payload

        created = dict(response)
        created["status"] = "in_progress"
        created["output"] = []
        created["output_text"] = ""

        if include_created:
            self.begin_sse()
            self.write_sse("response.created", event_payload("response.created", {"response": created}))
        for output_index, output in enumerate(response["output"]):
            if output.get("type") in {"function_call", "custom_tool_call"}:
                self.write_function_call_stream(response, output, output_index, event_payload)
            else:
                self.write_message_stream(response, output, output_index, event_payload)
        self.write_sse("response.completed", event_payload("response.completed", {"response": response}))

    def write_message_stream(self, response: dict, output: dict, output_index: int, event_payload) -> None:
        content = output["content"][0]
        text = content.get("text", "")
        common = {
            "response_id": response["id"],
            "output_index": output_index,
            "item_id": output["id"],
            "content_index": 0,
        }
        self.write_sse(
            "response.output_item.added",
            event_payload(
                "response.output_item.added",
                {
                    "response_id": response["id"],
                    "output_index": output_index,
                    "item": {
                        "id": output["id"],
                        "type": "message",
                        "status": "in_progress",
                        "role": "assistant",
                        "content": [],
                    },
                },
            ),
        )
        self.write_sse(
            "response.content_part.added",
            event_payload(
                "response.content_part.added",
                {**common, "part": {"type": "output_text", "text": "", "annotations": []}},
            ),
        )
        if text:
            self.write_sse(
                "response.output_text.delta",
                event_payload("response.output_text.delta", {**common, "delta": text}),
            )
        self.write_sse(
            "response.output_text.done",
            event_payload("response.output_text.done", {**common, "text": text, "logprobs": []}),
        )
        self.write_sse(
            "response.content_part.done",
            event_payload("response.content_part.done", {**common, "part": content}),
        )
        self.write_sse(
            "response.output_item.done",
            event_payload(
                "response.output_item.done",
                {"response_id": response["id"], "output_index": output_index, "item": output},
            ),
        )

    def write_function_call_stream(self, response: dict, output: dict, output_index: int, event_payload) -> None:
        added_item = dict(output)
        added_item["status"] = "in_progress"
        field = "input" if output["type"] == "custom_tool_call" else "arguments"
        event_kind = "custom_tool_call_input" if field == "input" else "function_call_arguments"
        added_item[field] = ""
        arguments = output.get(field, "")
        common = {
            "response_id": response["id"],
            "item_id": output["id"],
            "output_index": output_index,
            "call_id": output["call_id"],
            "name": output["name"],
        }
        self.write_sse(
            "response.output_item.added",
            event_payload(
                "response.output_item.added",
                {"response_id": response["id"], "output_index": output_index, "item": added_item},
            ),
        )
        if arguments:
            self.write_sse(
                f"response.{event_kind}.delta",
                event_payload(f"response.{event_kind}.delta", {**common, "delta": arguments}),
            )
        self.write_sse(
            f"response.{event_kind}.done",
            event_payload(f"response.{event_kind}.done", {**common, field: arguments}),
        )
        self.write_sse(
            "response.output_item.done",
            event_payload(
                "response.output_item.done",
                {"response_id": response["id"], "output_index": output_index, "item": output},
            ),
        )

    def log_message(self, fmt: str, *args) -> None:
        sys.stderr.write(f"[llm-away] {self.address_string()} {fmt % args}\n")


def warm_backend(config: AppConfig, backend: Backend) -> None:
    ensure_ready = getattr(backend, "ensure_ready", None)
    if ensure_ready is None:
        return
    try:
        print(f"llm-away: warming backend for model {config.model.name}", file=sys.stderr, flush=True)
        ensure_ready(config.model.name)
        completion = getattr(backend, "completion", None)
        if completion is not None:
            completion("Reply with OK.", max_tokens=1)
        print(f"llm-away: backend ready for model {config.model.name}", file=sys.stderr, flush=True)
    except Exception as exc:
        print(f"llm-away: backend warmup failed: {exc}", file=sys.stderr, flush=True)


def serve(config: AppConfig, backend: Backend, warm: bool = False) -> None:
    class Handler(ProviderHandler):
        pass

    Handler.backend = backend
    Handler.config = config
    httpd = ThreadingHTTPServer((config.server.host, config.server.port), Handler)
    previous_handlers = {}
    shutdown_started = False

    def stop(signum, frame):
        nonlocal shutdown_started
        if shutdown_started:
            return
        shutdown_started = True
        print("llm-away: shutting down", file=sys.stderr, flush=True)
        threading.Thread(target=httpd.shutdown, daemon=True).start()

    for sig in (signal.SIGINT, signal.SIGTERM):
        previous_handlers[sig] = signal.getsignal(sig)
        signal.signal(sig, stop)

    print(f"llm-away provider listening on http://{config.server.host}:{config.server.port}", flush=True)
    if warm:
        threading.Thread(target=warm_backend, args=(config, backend), daemon=True).start()
    try:
        httpd.serve_forever()
    finally:
        close = getattr(backend, "close", None)
        if close:
            close()
        httpd.server_close()
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)
