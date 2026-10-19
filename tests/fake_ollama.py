"""A tiny HTTP server that speaks Ollama's REST API (/api/tags, /api/generate).

Used by the tests to exercise the real OllamaLLM client end-to-end over HTTP
(payload, structured-output schema, retries, fallbacks, guardrails) without
downloading a model. Its answers come from the simulated LLM, optionally
corrupted on purpose to test each failure path.
"""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from agents.llm import MockLLM


class FakeOllama:
    def __init__(self, mode: str = "good", model: str = "llama3.2:latest"):
        self.mode = mode
        self.model = model
        self.requests: list[dict] = []
        self.extraction_calls = 0
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):  # silence
                pass

            def _send(self, code, obj):
                body = json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                if self.path == "/api/tags":
                    self._send(200, {"models": [{"name": outer.model}]})
                else:
                    self._send(404, {"error": "not found"})

            def do_POST(self):
                payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                outer.requests.append(payload)
                self._send(*outer.respond(payload))

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.host = f"http://127.0.0.1:{self.server.server_port}"
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def respond(self, payload: dict):
        if self.mode == "old_server" and isinstance(payload.get("format"), dict):
            return 400, {"error": "invalid format"}
        prompt = payload["prompt"]
        text = MockLLM().generate(prompt, payload.get("system")).text
        is_extraction = "### Task" in prompt
        if is_extraction:
            self.extraction_calls += 1
            if self.mode == "always_bad" or (self.mode == "bad_then_good" and self.extraction_calls == 1):
                text = "Sure! Here is the JSON: {procedure_key: hernia"
            elif self.mode == "hallucinate":
                obj = json.loads(text)
                obj["chronic_conditions"].append({"condition": "asthma", "duration_years": 5, "evidence_sentence": 3})
                text = json.dumps(obj)
        elif self.mode == "unfaithful_letter":
            text = text + "\nAs a goodwill gesture we will also pay ₹9,99,999."
        elif self.mode == "echo_markers":  # seen with qwen2.5:7b: the answer repeats the prompt's markers
            text = f"<<<LETTER\n{text}\nLETTER>>>"
        return 200, {"model": payload["model"], "response": text, "done": True,
                     "prompt_eval_count": len(prompt) // 4, "eval_count": len(text) // 4}

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.server.shutdown()
        self.server.server_close()
