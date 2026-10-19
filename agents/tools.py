"""Deterministic tools the agents call.

Design rule: the LLM reads and writes text; all money, dates and policy look-ups
run in plain Python so they are exact and repeatable. Every call is recorded
for the execution-sequence diagram.
"""
from __future__ import annotations

import json
import re
from datetime import date
from functools import lru_cache
from pathlib import Path

from .trace import record

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


@lru_cache(maxsize=None)
def _load(name: str) -> str:
    return (DATA_DIR / name).read_text(encoding="utf-8")


def load_json(name: str):
    return json.loads(_load(name))


# ---------- data access tools ----------
def policy_lookup(policy_no: str) -> dict | None:
    p = load_json("policies.json").get(policy_no)
    record("tool", "policy_lookup", f"{policy_no} → {'found' if p else 'not found'}")
    return p


def tariff_lookup(procedure_key: str) -> dict | None:
    t = load_json("tariffs.json").get(procedure_key)
    record("tool", "tariff_lookup", f"{procedure_key} → {'₹{:,}–₹{:,}'.format(t['min'], t['max']) if t else 'none'}")
    return t


def clause_search(terms: list[str], k: int = 6) -> list[dict]:
    """Keyword retriever over the policy wording (tag match = 2 points, text match = 1)."""
    terms = [t.lower() for t in terms if t]
    scored = []
    for c in load_json("policy_clauses.json"):
        hay = " ".join(c["tags"] + [c["title"].lower(), c["text"].lower()])
        score = sum(2 if t in c["tags"] else 1 for t in terms if t in hay)
        if score:
            scored.append((score, c))
    scored.sort(key=lambda x: -x[0])
    out = [c for _, c in scored[:k]]
    record("tool", "clause_search", f"{len(terms)} terms → {', '.join(c['id'] for c in out)}")
    return out


# ---------- date tools ----------
def to_date(s: str) -> date:
    return date.fromisoformat(s)


def months_between(start: date, end: date) -> int:
    m = (end.year - start.year) * 12 + (end.month - start.month)
    if end.day < start.day:
        m -= 1
    return m


# ---------- money tools ----------
def cost_lines(costs: dict) -> dict:
    return {
        "room": costs["room_rent_per_day"] * costs["room_days"],
        "icu": costs.get("icu_per_day", 0) * costs.get("icu_days", 0),
        "surgery_ot": costs["surgery_ot"],
        "medicines_consumables": costs["medicines_consumables"],
        "investigations": costs["investigations"],
        "other": costs["other"],
    }


def total_cost(costs: dict) -> int:
    return sum(cost_lines(costs).values())


def inr(x: float) -> str:
    """Indian-style currency format: ₹1,50,000."""
    x = int(round(x))
    s = str(abs(x))
    if len(s) > 3:
        head, tail = s[:-3], s[-3:]
        head = re.sub(r"(\d)(?=(\d\d)+$)", r"\1,", head)
        s = f"{head},{tail}"
    return ("-" if x < 0 else "") + "₹" + s
