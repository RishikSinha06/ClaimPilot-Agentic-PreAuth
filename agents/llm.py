"""LLM backends.

Default open-source model: Meta Llama 3.2 3B Instruct, served locally by Ollama
(`ollama pull llama3.2`). Ollama runs the model with llama.cpp on CPU or GPU
from a 4-bit quantised GGUF file (~2 GB), so no paid API and no internet
connection are needed at run time.

Two backends share one interface, generate(prompt, system, schema, purpose):

* OllamaLLM - real model over Ollama's REST API (/api/generate).
    - temperature 0 and a fixed seed  -> repeatable answers
    - `format` = our JSON schema      -> grammar-constrained decoding (Ollama >= 0.5);
                                         falls back to plain JSON mode on older servers
    - response cache keyed by SHA-256 of (model, system, prompt, schema)
    - token counts and latency are taken from Ollama's response metadata

* MockLLM - offline stand-in so the demo never breaks. It reads the SAME
    prompt the real model would get and answers with rule-based NLP. It is
    clearly labelled "simulated" everywhere in the UI.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass

import requests

from .nlp.preprocess import analyze_numbered_text, estimate_tokens, facts_from_entities
from .nlp.prompts import TASK_MARKER
from .trace import record

RECOMMENDED_MODELS = {
    "llama3.2": "Meta Llama 3.2 3B Instruct · default · ~2.0 GB",
    "llama3.2:1b": "Meta Llama 3.2 1B Instruct · fastest · ~1.3 GB",
    "qwen2.5:3b": "Alibaba Qwen 2.5 3B Instruct · ~1.9 GB",
    "qwen2.5:7b": "Alibaba Qwen 2.5 7B Instruct · benchmarked in results/ · ~4.7 GB",
    "phi3.5": "Microsoft Phi-3.5 mini 3.8B · ~2.2 GB",
    "mistral": "Mistral 7B Instruct · best quality, slower · ~4.1 GB",
}


@dataclass
class LLMResult:
    text: str
    model: str
    prompt_tokens: int | None
    output_tokens: int | None
    latency_ms: float
    cached: bool = False
    simulated: bool = False
    format_used: str = ""


class MockLLM:
    name = "Mock LLM (simulated, offline)"
    model = "mock"
    is_mock = True

    def available(self) -> tuple[bool, str]:
        return True, "Built-in simulated LLM (no model download needed)"

    def generate(self, prompt: str, system: str | None = None, schema: dict | None = None,
                 purpose: str = "") -> LLMResult:
        t0 = time.perf_counter()
        if TASK_MARKER in prompt:
            task = prompt.split(TASK_MARKER, 1)[1]
            m = re.search(r"Request date:\s*(\d{4})", task)
            sentences, ents = analyze_numbered_text(task, int(m.group(1)) if m else 2026)
            facts = facts_from_entities(sentences, ents)
            for c in facts["chronic_conditions"]:
                c.pop("stated_as_declared", None)
            text = json.dumps(facts)
        elif "<<<LETTER" in prompt:
            text = prompt.split("<<<LETTER", 1)[1].split("LETTER>>>", 1)[0].strip()
        else:
            text = ""
        ms = (time.perf_counter() - t0) * 1000
        res = LLMResult(text=text, model=self.model, prompt_tokens=estimate_tokens((system or "") + prompt),
                        output_tokens=estimate_tokens(text), latency_ms=round(ms, 2), simulated=True,
                        format_used="json-schema (simulated)")
        record("llm", self.name, f"{purpose}: ~{res.prompt_tokens}→{res.output_tokens} tokens (estimated)")
        return res


_CACHE: dict[str, LLMResult] = {}


class OllamaLLM:
    is_mock = False

    def __init__(self, model: str = "llama3.2", host: str = "http://localhost:11434", temperature: float = 0.0,
                 seed: int = 42, num_ctx: int = 4096, num_predict: int = 512, timeout: int = 180,
                 use_cache: bool = True):
        self.model = model
        self.host = host.rstrip("/")
        self.options = {"temperature": temperature, "seed": seed, "num_ctx": num_ctx, "num_predict": num_predict}
        self.timeout = timeout
        self.use_cache = use_cache

    @property
    def name(self) -> str:
        return f"Ollama · {self.model}"

    def list_models(self) -> list[str]:
        r = requests.get(f"{self.host}/api/tags", timeout=3)
        r.raise_for_status()
        return [m.get("name", "") for m in r.json().get("models", [])]

    def available(self) -> tuple[bool, str]:
        try:
            names = self.list_models()
        except requests.RequestException:
            return False, f"Ollama is not running at {self.host}. Start it with `ollama serve`."
        wanted = {self.model, f"{self.model}:latest"}
        if not any(n in wanted for n in names):
            return False, f"Model '{self.model}' is not downloaded. Run `ollama pull {self.model}`."
        return True, f"Connected · {self.model}"

    def _post(self, payload: dict) -> requests.Response:
        return requests.post(f"{self.host}/api/generate", json=payload, timeout=self.timeout)

    def generate(self, prompt: str, system: str | None = None, schema: dict | None = None,
                 purpose: str = "") -> LLMResult:
        key = hashlib.sha256(json.dumps([self.model, self.options, system, prompt, schema], sort_keys=True).encode()).hexdigest()
        if self.use_cache and key in _CACHE:
            hit = _CACHE[key]
            record("llm", self.name, f"{purpose}: cache hit")
            return LLMResult(**{**hit.__dict__, "cached": True})

        payload = {"model": self.model, "prompt": prompt, "stream": False, "options": self.options}
        if system:
            payload["system"] = system
        fmt = ""
        if schema is not None:
            payload["format"] = schema
            fmt = "json-schema"
        t0 = time.perf_counter()
        r = self._post(payload)
        if r.status_code >= 400 and schema is not None:
            payload["format"] = "json"  # older Ollama: no structured outputs
            fmt = "json"
            r = self._post(payload)
        r.raise_for_status()
        data = r.json()
        ms = (time.perf_counter() - t0) * 1000
        res = LLMResult(text=data.get("response", ""), model=self.model,
                        prompt_tokens=data.get("prompt_eval_count"), output_tokens=data.get("eval_count"),
                        latency_ms=round(ms, 1), format_used=fmt or "text")
        record("llm", self.name, f"{purpose}: {res.prompt_tokens}→{res.output_tokens} tokens, {ms:.0f} ms")
        if self.use_cache:
            _CACHE[key] = res
        return res


def make_llm(backend: str, model: str = "llama3.2", host: str = "http://localhost:11434", timeout: int = 180):
    """Factory used by the UI and scripts. Returns (llm, status_message)."""
    if backend == "ollama":
        llm = OllamaLLM(model=model, host=host, timeout=timeout)
        ok, msg = llm.available()
        if ok:
            return llm, msg
        return MockLLM(), msg + " Using the simulated LLM instead."
    return MockLLM(), MockLLM().available()[1]
