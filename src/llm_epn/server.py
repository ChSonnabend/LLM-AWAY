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
    response_object,
    responses_input_to_prompt,
)


class ProviderHandler(BaseHTTPRequestHandler):
    backend: Backend
    config: AppConfig

    def do_GET(self) -> None:
        if self.path == "/health":
            self.write_json({"status": "ok", "model": self.config.model.name})
            return
        if self.path == "/v1/models":
            self.write_json(models_list(self.config.model.name))
            return
        self.write_json({"error": "not found"}, status=404)

    def do_POST(self) -> None:
        try:
            payload = self.read_json()
            if self.path == "/v1/chat/completions":
                self.handle_chat(payload)
            elif self.path == "/v1/responses":
                self.handle_responses(payload)
            else:
                self.write_json({"error": "not found"}, status=404)
        except BackendError as exc:
            self.write_json({"error": {"message": str(exc), "type": "backend_error"}}, status=502)
        except Exception as exc:
            self.write_json({"error": {"message": str(exc), "type": "server_error"}}, status=500)

    def handle_chat(self, payload: dict) -> None:
        model = payload.get("model") or self.config.model.name
        prompt = self.compact_prompt(messages_to_prompt(payload.get("messages", [])))
        text = self.backend.infer(InferenceRequest(prompt=prompt, model=model))
        if payload.get("stream"):
            self.write_chat_stream(model, text)
            return
        self.write_json(chat_completion(model, text))

    def handle_responses(self, payload: dict) -> None:
        model = payload.get("model") or self.config.model.name
        prompt = self.compact_prompt(responses_input_to_prompt(payload.get("input", "")))
        if payload.get("stream"):
            self.handle_responses_stream(model, prompt)
            return
        text = self.backend.infer(InferenceRequest(prompt=prompt, model=model))
        response = response_object(model, text)
        self.write_json(response)

    def read_json(self) -> dict:
        length = int(self.headers.get("content-length", "0"))
        raw = self.rfile.read(length).decode("utf-8")
        return json.loads(raw) if raw else {}

    def compact_prompt(self, prompt: str) -> str:
        limit = self.config.gateway.max_prompt_chars
        if limit <= 0 or len(prompt) <= limit:
            return prompt

        head_chars = max(0, self.config.gateway.prompt_keep_head_chars)
        tail_chars = max(0, self.config.gateway.prompt_keep_tail_chars)
        if head_chars + tail_chars >= limit:
            tail_chars = max(0, limit - head_chars)
        omitted = len(prompt) - head_chars - tail_chars
        return (
            prompt[:head_chars]
            + f"\n\n[llm-epn: compacted {omitted} characters from the middle of the Codex context]\n\n"
            + prompt[-tail_chars:]
        )

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

    def handle_responses_stream(self, model: str, prompt: str) -> None:
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
            text = self.backend.infer(InferenceRequest(prompt=prompt, model=model))
        except Exception as exc:
            failed = dict(created)
            failed["status"] = "failed"
            failed["error"] = {"message": str(exc), "type": "backend_error"}
            self.write_sse(
                "response.failed",
                {"type": "response.failed", "sequence_number": 1, "response": failed},
            )
            return
        response = response_object(model, text, response_id=response_id)
        self.write_response_stream(response, sequence_number=1, include_created=False)

    def write_response_stream(self, response: dict, sequence_number: int = 0, include_created: bool = True) -> None:
        output = response["output"][0]

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
        if output.get("type") == "function_call":
            self.write_function_call_stream(response, output, sequence_number, event_payload)
            return
        self.write_message_stream(response, output, event_payload)
        self.write_sse("response.completed", event_payload("response.completed", {"response": response}))

    def write_message_stream(self, response: dict, output: dict, event_payload) -> None:
        content = output["content"][0]
        text = content.get("text", "")
        common = {
            "response_id": response["id"],
            "output_index": 0,
            "item_id": output["id"],
            "content_index": 0,
        }
        self.write_sse(
            "response.output_item.added",
            event_payload(
                "response.output_item.added",
                {
                    "response_id": response["id"],
                    "output_index": 0,
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
                {"response_id": response["id"], "output_index": 0, "item": output},
            ),
        )

    def write_function_call_stream(self, response: dict, output: dict, sequence_number: int, event_payload) -> None:
        added_item = dict(output)
        added_item["status"] = "in_progress"
        added_item["arguments"] = ""
        arguments = output.get("arguments", "")
        common = {
            "response_id": response["id"],
            "item_id": output["id"],
            "output_index": 0,
            "call_id": output["call_id"],
            "name": output["name"],
        }
        self.write_sse(
            "response.output_item.added",
            event_payload(
                "response.output_item.added",
                {"response_id": response["id"], "output_index": 0, "item": added_item},
            ),
        )
        if arguments:
            self.write_sse(
                "response.function_call_arguments.delta",
                event_payload("response.function_call_arguments.delta", {**common, "delta": arguments}),
            )
        self.write_sse(
            "response.function_call_arguments.done",
            event_payload("response.function_call_arguments.done", {**common, "arguments": arguments}),
        )
        self.write_sse(
            "response.output_item.done",
            event_payload(
                "response.output_item.done",
                {"response_id": response["id"], "output_index": 0, "item": output},
            ),
        )
        self.write_sse("response.completed", event_payload("response.completed", {"response": response}))

    def log_message(self, fmt: str, *args) -> None:
        sys.stderr.write(f"[llm-epn] {self.address_string()} {fmt % args}\n")


def warm_backend(config: AppConfig, backend: Backend) -> None:
    ensure_ready = getattr(backend, "ensure_ready", None)
    if ensure_ready is None:
        return
    try:
        print(f"llm-epn: warming backend for model {config.model.name}", file=sys.stderr, flush=True)
        ensure_ready(config.model.name)
        completion = getattr(backend, "completion", None)
        if completion is not None:
            completion("Reply with OK.", max_tokens=1)
        print(f"llm-epn: backend ready for model {config.model.name}", file=sys.stderr, flush=True)
    except Exception as exc:
        print(f"llm-epn: backend warmup failed: {exc}", file=sys.stderr, flush=True)


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
        print("llm-epn: shutting down", file=sys.stderr, flush=True)
        threading.Thread(target=httpd.shutdown, daemon=True).start()

    for sig in (signal.SIGINT, signal.SIGTERM):
        previous_handlers[sig] = signal.getsignal(sig)
        signal.signal(sig, stop)

    print(f"llm-epn provider listening on http://{config.server.host}:{config.server.port}", flush=True)
    if warm:
        threading.Thread(target=warm_backend, args=(config, backend), daemon=True).start()
    try:
        httpd.serve_forever()
    finally:
        close = getattr(backend, "close", None)
        if close:
            close()
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)
