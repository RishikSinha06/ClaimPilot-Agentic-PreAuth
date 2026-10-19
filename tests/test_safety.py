"""Safety properties: a wrong LLM answer must not become a wrong decision, a leaked
identifier or an unfaithful letter (fault injection, materiality, PII, audit)."""
import copy
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agents import FAULTS, FaultyLLM, MockLLM, Orchestrator, append_audit_log, verify_audit_log  # noqa: E402
from agents.evaluation import score  # noqa: E402
from agents.guardrails import letter_faithful, pii_gate  # noqa: E402
from agents.nlp.preprocess import p4_deidentify  # noqa: E402
from agents.nlp.prompts import clean_letter_output  # noqa: E402

CLAIMS = json.loads((ROOT / "data" / "claims.json").read_text(encoding="utf-8"))
BY_ID = {c["claim_id"]: c for c in CLAIMS}


def run_with_fault(claim, fault):
    llm = FaultyLLM(MockLLM(), fault)
    return Orchestrator(llm=llm).run(claim), llm.applied


def finding(ctx, check):
    return next(f for f in ctx["critic"]["findings"] if f["check"] == check)


# ------------------------------------------------------------ fault injection
@pytest.mark.parametrize("fault", [f for f, (target, _) in FAULTS.items() if target == "extraction"])
def test_no_extraction_fault_sends_a_wrong_decision(fault):
    for c in CLAIMS:
        ctx, applied = run_with_fault(c, fault)
        if applied:
            assert score(c, ctx)["outcome"] != "wrong", f"{fault} on {c['claim_id']} → {ctx['decision']['decision']}"


@pytest.mark.parametrize("fault", [f for f, (target, _) in FAULTS.items() if target == "letter"])
def test_every_corrupted_letter_is_blocked(fault):
    for c in CLAIMS:
        ctx, applied = run_with_fault(c, fault)
        assert applied
        assert ctx["decision"]["letter_method"].startswith("template"), f"{fault} letter sent for {c['claim_id']}"


def test_wrong_duration_is_corrected_from_the_text():
    ctx, _ = run_with_fault(BY_ID["PA-2026-0107"], "wrong_duration")  # LLM: 0.5 y; notes: since 2018, x 4 yrs
    durations = {c["condition"]: c["duration_years"] for c in ctx["extracted"]["chronic_conditions"]}
    assert durations == {"hypertension": 8, "diabetes": 4}
    assert finding(ctx, "Evidence grounding")["status"] == "WARN"
    assert ctx["decision"]["decision"] == "ESCALATE"  # undisclosed PED still caught


def test_functional_need_flip_cannot_lift_the_cosmetic_exclusion():
    ctx, _ = run_with_fault(BY_ID["PA-2026-0105"], "flipped_functional")  # rhinoplasty, "no nasal obstruction"
    assert ctx["eligibility"]["eligible"]  # the LLM facts alone would pay for cosmetic surgery
    assert finding(ctx, "Extractor agreement")["status"] == "FAIL"
    assert ctx["decision"]["decision"] == "ESCALATE"


def test_immaterial_disagreement_is_only_a_warning():
    ctx, _ = run_with_fault(BY_ID["PA-2026-0101"], "flipped_functional")  # appendectomy: flag is irrelevant
    assert finding(ctx, "Extractor agreement")["status"] == "WARN"
    assert (ctx["decision"]["decision"], ctx["decision"]["amount"]) == ("APPROVE", 130000)


def test_disputed_denial_is_escalated_and_priced_on_rule_facts_if_approved():
    o = Orchestrator(llm=FaultyLLM(MockLLM(), "flipped_cosmetic"))
    ctx = o.run(BY_ID["PA-2026-0101"])  # LLM says cosmetic → cosmetic exclusion; rules say eligible
    assert not ctx["eligibility"]["eligible"] and ctx.get("calculation") is None
    assert ctx["decision"]["decision"] == "ESCALATE"
    assert "Rejection grounds" in ctx["decision"]["escalation_causes"]
    o.human_review(ctx, "approve", "", "Dr. Rao")
    assert ctx["calculation"]["facts"] == "rule-based"
    assert (ctx["decision"]["decision"], ctx["decision"]["amount"]) == ("APPROVE", 130000)


def test_denial_on_disputed_facts_keeps_only_grounds_both_extractors_support():
    ctx, _ = run_with_fault(BY_ID["PA-2026-0109"], "wrong_procedure")  # expired policy; LLM says dental surgery
    assert ctx["decision"]["decision"] == "DENY"
    assert ctx["decision"]["reasons"] == [r for r in ctx["decision"]["reasons"] if r.startswith("Policy in force")]
    assert len(ctx["decision"]["reasons"]) == 1  # the dental exclusion exists only under the LLM's facts


def test_same_check_with_a_different_reason_is_not_a_robust_denial():
    ctx, _ = run_with_fault(BY_ID["PA-2026-0105"], "wrong_procedure")  # rules: cosmetic exclusion; LLM: dental exclusion
    assert ctx["decision"]["decision"] == "ESCALATE"  # never deny citing the wrong exclusion


def test_dropped_unrelated_undisclosed_condition_still_escalates():
    claim = copy.deepcopy(BY_ID["PA-2026-0108"])
    claim["doctor_notes"] += " K/c/o HTN since 2015 on Tab. Telmisartan."  # not related to gallstones, not declared
    assert Orchestrator().run(claim)["decision"]["decision"] == "ESCALATE"
    ctx, applied = run_with_fault(claim, "dropped_conditions")
    assert applied and ctx["decision"]["decision"] == "ESCALATE"


# ------------------------------------------------------------ policy rules
def test_dental_treatment_after_an_accident_is_covered():  # clause 5.2
    claim = copy.deepcopy(BY_ID["PA-2026-0112"])
    claim.update(doctor_notes="Pt sustained RTA 2 hours ago with fracture of lower left third molar. MLC registered. "
                              "Planned surgical extraction under GA as day care.",
                 documents=claim["documents"] + ["mlc_fir"])
    ctx = Orchestrator().run(claim)
    assert ctx["extracted"]["accident"]
    assert next(c for c in ctx["eligibility"]["checks"] if c["check"] == "Permanent exclusions")["result"] == "PASS"
    assert ctx["decision"]["decision"] in ("APPROVE", "PARTIAL")


def test_icu_days_without_icu_rate_does_not_crash():
    claim = copy.deepcopy(BY_ID["PA-2026-0101"])
    claim["costs"].pop("icu_per_day")
    claim["costs"]["icu_days"] = 1
    assert Orchestrator().run(claim)["decision"]["decision"] == "APPROVE"


def test_human_review_only_for_escalated_cases():
    o = Orchestrator()
    ctx = o.run(BY_ID["PA-2026-0101"])
    with pytest.raises(ValueError):
        o.human_review(ctx, "approve")


def test_simulated_agreement_is_labelled_as_not_independent():
    ctx = Orchestrator().run(BY_ID["PA-2026-0101"])
    assert "not independent evidence" in finding(ctx, "Extractor agreement")["detail"]


# ------------------------------------------------------------ PII
CLAIM = {"claim_id": "PA-X", "patient": "Lata Kulkarni", "hospital": "Jehangir Heart Institute, Pune", "policy_no": "P1002"}


def test_titled_names_landlines_and_emails_are_masked():
    notes = "Seen by Dr. Mehta. Son Mr Ramesh Patil accompanies, call 020-26123456 or ramesh.p@example.com."
    masked, pii = p4_deidentify({"notes": notes}, CLAIM)
    t = masked["notes"]
    assert "Mehta" not in t and "Ramesh" not in t and "26123456" not in t and "example.com" not in t
    assert "Dr. [PERSON_1]" in t and "Mr [PERSON_2]" in t
    assert pii["[PERSON_2]"] == "Ramesh Patil"


def test_pii_gate_scans_independently_of_the_masking():
    ok, leaks = pii_gate("Discussed with Dr. Rao; call 98765 43210. Kulkarni family informed.", {}, CLAIM)
    assert not ok
    assert {"unmasked number with 10+ digits (phone / ID)", "unmasked titled personal name", "unmasked patient name"} <= set(leaks)
    assert pii_gate("S1: Known case of hypertension since 2018. Request date: 2026-10-10", {}, CLAIM)[0]


def test_no_pii_false_positives_on_the_sample_cases():
    for c in CLAIMS:
        ctx = Orchestrator().run(c)
        assert ctx["llm_extraction"]["pii_gate"] == "pass", c["claim_id"]
        assert "PII gate blocked" not in ctx["decision"]["letter_method"], c["claim_id"]


class RecordingLLM(MockLLM):
    def __init__(self):
        self.prompts = []

    def generate(self, prompt, system=None, schema=None, purpose=""):
        self.prompts.append(prompt)
        return super().generate(prompt, system, schema, purpose)


def test_reviewer_remarks_are_masked_for_the_llm_and_restored_in_the_letter():
    llm = RecordingLLM()
    o = Orchestrator(llm=llm)
    ctx = o.run(BY_ID["PA-2026-0107"])
    remarks = "Spoke to Dr. Iyer and Mrs Kulkarni's son on 9876543210."
    o.human_review(ctx, "deny", remarks, "Dr. Rao")
    letter_prompt = llm.prompts[-1]
    assert "<<<LETTER" in letter_prompt
    assert not any(s in letter_prompt for s in ("Iyer", "Kulkarni", "9876543210"))
    assert remarks in ctx["decision"]["letter"]


# ------------------------------------------------------------ letter guardrail
TEMPLATE = ("We authorise cashless treatment up to ₹40,000, payable on the final bill.\n\nReasons:\n"
            "  • Sub-limit for cataract: ₹25,000 — Capped at ₹40,000 (clause 3.6)\n\nRegards")
REASON = "Sub-limit for cataract: ₹25,000 — Capped at ₹40,000 (clause 3.6)"


def test_faithful_polish_is_accepted():
    polished = ("We are happy to confirm cashless authorisation up to ₹40,000 payable on the final bill.\n\nReasons:\n"
                "- Sub-limit for cataract: ₹25,000 - Capped at ₹40,000 (clause 3.6).\n\nKind regards")
    ok, problems = letter_faithful(TEMPLATE, polished, must_keep=[REASON], decision="PARTIAL")
    assert ok, problems  # comma after an amount, bullet style, dash style and final full stop may change


def test_paraphrased_reason_and_flipped_decision_are_rejected():
    paraphrased = TEMPLATE.replace("Capped at", "Limited to")
    assert not letter_faithful(TEMPLATE, paraphrased, must_keep=[REASON], decision="PARTIAL")[0]
    flipped = TEMPLATE.replace("We authorise cashless treatment", "We regret that cashless treatment cannot be authorised")
    ok, problems = letter_faithful(TEMPLATE, flipped, must_keep=[REASON], decision="PARTIAL")
    assert not ok and any("contradicts" in p for p in problems)


def test_echoed_markers_and_fences_are_stripped():
    assert clean_letter_output("```\n<<<LETTER\nDear team\nLETTER>>>\n```") == "Dear team"


# ------------------------------------------------------------ audit log
def test_audit_log_hash_chain_detects_tampering(tmp_path):
    log = tmp_path / "audit.jsonl"
    log.write_text(json.dumps({"claim_id": "legacy line without a hash"}) + "\n", encoding="utf-8")
    o = Orchestrator()
    for cid in ("PA-2026-0101", "PA-2026-0102", "PA-2026-0104"):
        append_audit_log(o.run(BY_ID[cid]), log)
    assert verify_audit_log(log) == {"ok": True, "verified": 3, "bad_line": None}
    lines = log.read_text(encoding="utf-8").splitlines()
    entry = json.loads(lines[2])
    entry["amount"] = 999999
    lines[2] = json.dumps(entry, ensure_ascii=False)
    log.write_text("\n".join(lines) + "\n", encoding="utf-8")
    assert verify_audit_log(log) == {"ok": False, "verified": 1, "bad_line": 3}
