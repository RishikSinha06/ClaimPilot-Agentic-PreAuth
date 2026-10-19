"""Scoring one run against a hand-labelled case. Shared by the app's batch tab,
scripts/benchmark_llm.py and scripts/fault_injection.py."""
from __future__ import annotations

FIELDS = ["procedure_key", "chronic_conditions", "cosmetic_intent", "functional_need", "accident"]
HOLDS = ("ESCALATE", "QUERY")  # a human or the hospital is asked before anything is paid or refused


def field_ok(field: str, got: dict, gold: dict) -> bool:
    if field == "chronic_conditions":
        return sorted(c["condition"] for c in got[field]) == sorted(c["condition"] for c in gold[field])
    return got[field] == gold[field]


def outcome(claim: dict, decision: str, amount: int) -> str:
    """correct | held (asked a human or the hospital instead: safe, extra work) | wrong (a wrong decision went out)."""
    if decision == claim["expected_decision"] and (not claim.get("expected_amount") or amount == claim["expected_amount"]):
        return "correct"
    return "held" if decision in HOLDS else "wrong"


def score(claim: dict, ctx: dict) -> dict:
    d, facts, gold = ctx["decision"], ctx["extracted"], claim["gold_extraction"]
    fields = {f: field_ok(f, facts, gold) for f in FIELDS}
    return {"decision": d["decision"], "amount": d["amount"],
            "decision_ok": d["decision"] == claim["expected_decision"],
            "amount_ok": not claim.get("expected_amount") or d["amount"] == claim["expected_amount"],
            "fields": fields, "facts_correct": sum(fields.values()),
            "outcome": outcome(claim, d["decision"], d["amount"])}
