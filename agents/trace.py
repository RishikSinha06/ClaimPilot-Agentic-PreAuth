"""Call recorder.

The orchestrator opens a recording scope for every agent step. Tools, NLP
stages and LLM calls made inside that step register themselves here, so the
UI can draw an exact sequence diagram of what happened (agent -> tool,
agent -> LLM, ...).
"""
from __future__ import annotations

import contextvars

_calls = contextvars.ContextVar("calls", default=None)


def begin():
    """Start recording calls for one agent step. Returns (list, token)."""
    lst: list = []
    token = _calls.set(lst)
    return lst, token


def end(token) -> None:
    _calls.reset(token)


def record(kind: str, name: str, detail: str = "", **extra) -> None:
    """kind: 'tool' | 'nlp' | 'llm' | 'guardrail' | 'data'."""
    lst = _calls.get()
    if lst is not None:
        lst.append({"kind": kind, "name": name, "detail": detail, **extra})
