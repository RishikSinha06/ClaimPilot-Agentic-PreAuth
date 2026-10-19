"""End-to-end validation of the agent workflow on the 12 labelled synthetic cases."""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agents import Orchestrator, audit_record  # noqa: E402

CLAIMS = json.loads((ROOT / "data" / "claims.json").read_text(encoding="utf-8"))
BY_ID = {c["claim_id"]: c for c in CLAIMS}


@pytest.mark.parametrize("claim", CLAIMS, ids=[c["claim_id"] for c in CLAIMS])
def test_decision_matches_expected(claim):
    assert Orchestrator().run(claim)["decision"]["decision"] == claim["expected_decision"]


@pytest.mark.parametrize("claim", [c for c in CLAIMS if c["expected_amount"]], ids=lambda c: c["claim_id"])
def test_amount_matches_hand_calculation(claim):
    assert Orchestrator().run(claim)["decision"]["amount"] == claim["expected_amount"]


@pytest.mark.parametrize("claim", CLAIMS, ids=[c["claim_id"] for c in CLAIMS])
def test_extraction_matches_gold(claim):
    ex = Orchestrator().run(claim)["extracted"]
    gold = claim["gold_extraction"]
    assert ex["procedure_key"] == gold["procedure_key"]
    assert sorted(c["condition"] for c in ex["chronic_conditions"]) == sorted(c["condition"] for c in gold["chronic_conditions"])
    for k in ("cosmetic_intent", "functional_need", "accident"):
        assert ex[k] == gold[k], k


def test_critic_triggers_revision_loop():
    ctx = Orchestrator().run(BY_ID["PA-2026-0108"])
    assert ctx["revisions"] == 1
    assert [h["verdict"] for h in ctx["critic_history"]] == ["REVISE", "PASS"]
    assert any(m["type"] == "revise" for m in ctx["messages"])


def test_ineligible_claim_skips_calculation():
    ctx = Orchestrator().run(BY_ID["PA-2026-0109"])
    assert "calculate_payable" in ctx["skipped"]


def test_conditional_accident_task_runs_only_for_accidents():
    assert "verify_accident_docs" not in Orchestrator().run(BY_ID["PA-2026-0110"])["skipped"]
    # "No h/o trauma or RTA" must NOT trigger the accident branch (negation)
    assert "verify_accident_docs" in Orchestrator().run(BY_ID["PA-2026-0112"])["skipped"]


def test_query_then_resubmission_is_approved():
    o = Orchestrator()
    first = o.run(BY_ID["PA-2026-0106"])
    assert first["decision"]["decision"] == "QUERY"
    second = o.run(Orchestrator.resubmit(BY_ID["PA-2026-0106"]))
    assert second["decision"]["decision"] == BY_ID["PA-2026-0106"]["query_response"]["expected_decision_after"]


def test_human_review_overrides_escalation():
    o = Orchestrator()
    ctx = o.run(BY_ID["PA-2026-0107"])
    assert ctx["status"]["human_review"] == "awaiting human"
    o.human_review(ctx, "deny", "Hypertension since 2018 not disclosed", "Dr. Rao")
    assert ctx["decision"]["decision"] == "DENY"
    assert ctx["decision"]["decided_by"] == "human: Dr. Rao"
    assert "Hypertension since 2018 not disclosed" in ctx["decision"]["letter"]


def test_ped_escalation_review_reasons():
    o = Orchestrator()
    ctx = o.run(BY_ID["PA-2026-0107"])
    assert ctx["decision"]["escalation_causes"] == ["PED disclosure"]
    o.human_review(ctx, "approve", "", "Dr. Rao")
    assert ctx["decision"]["reasons"][0] == "Pre-existing condition reviewed by medical officer and accepted"


def _authority_limit_case():
    # PA-0101 is a clean ₹1,30,000 approval; a ₹50,000 limit makes the authority limit the only escalation cause.
    o = Orchestrator(auto_approve_limit=50000)
    ctx = o.run(BY_ID["PA-2026-0101"])
    assert ctx["decision"]["decision"] == "ESCALATE"
    assert ctx["decision"]["escalation_causes"] == ["Authority limit"]
    return o, ctx


def test_authority_limit_escalation_approved_by_human():
    o, ctx = _authority_limit_case()
    o.human_review(ctx, "approve", "", "Dr. Rao")
    d = ctx["decision"]
    assert (d["decision"], d["amount"]) == ("APPROVE", 130000)
    assert any("auto-approval limit" in r for r in d["reasons"])
    assert "Pre-existing" not in d["letter"] and "4.3" not in d["letter"]


def test_authority_limit_escalation_denied_or_queried_by_human():
    o, ctx = _authority_limit_case()
    o.human_review(ctx, "deny", "", "Dr. Rao")
    assert ctx["decision"]["decision"] == "DENY"
    assert "non-disclosure" not in ctx["decision"]["letter"].lower() and "4.3" not in ctx["decision"]["letter"]

    o, ctx = _authority_limit_case()
    o.human_review(ctx, "query", "", "Dr. Rao")
    assert ctx["decision"]["decision"] == "QUERY"
    assert "itemised cost estimate" in ctx["decision"]["letter"]
    assert "past treatment records" not in ctx["decision"]["letter"]


def test_amount_mismatch_approval_pays_lower_recomputed_amount(monkeypatch):
    import agents.critic as critic_mod
    monkeypatch.setattr(critic_mod, "independent_recompute", lambda *a, **k: 120000)
    o = Orchestrator()
    ctx = o.run(BY_ID["PA-2026-0101"])  # Executor computes ₹1,30,000
    assert ctx["decision"]["decision"] == "ESCALATE"
    assert ctx["decision"]["escalation_causes"] == ["Amount re-verification"]
    o.human_review(ctx, "approve", "", "Dr. Rao")
    d = ctx["decision"]
    assert (d["decision"], d["amount"]) == ("PARTIAL", 120000)
    assert "₹1,20,000" in d["letter"]
    assert "Pre-existing" not in d["letter"]


def test_audit_record_is_json_serialisable():
    rec = audit_record(Orchestrator().run(BY_ID["PA-2026-0103"]))
    assert json.loads(json.dumps(rec))["decision"]["amount"] == 120000


def test_every_step_records_its_calls():
    ctx = Orchestrator().run(BY_ID["PA-2026-0107"])
    kinds = {c["kind"] for t in ctx["trace"] for c in t["calls"]}
    assert {"nlp", "llm", "tool", "guardrail"} <= kinds
