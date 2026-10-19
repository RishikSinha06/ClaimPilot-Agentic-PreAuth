"""Planner Agent: profiles the request and writes the task plan the
Orchestrator executes. Tasks can carry a run-time condition (`when`), so the
plan adapts to what later agents discover (e.g. accident -> verify MLC/FIR)."""
from __future__ import annotations

BASE_DOCS = ["preauth_form", "id_card", "doctor_notes", "investigation_reports"]


class PlannerAgent:
    name = "Planner Agent"
    short = "Planner"
    icon = "🧭"

    def run(self, ctx: dict) -> dict:
        claim = ctx["claim"]
        emergency = claim["admission_type"] == "emergency"
        surgical = claim["treatment_type"] == "surgical"

        special = []
        special.append("Surgical package: benchmark estimate against network tariff" if surgical
                       else "Medical management: benchmark per-day cost")
        if claim["age"] >= 60:
            special.append("Senior citizen: check co-pay and pre-existing disease history carefully")
        if emergency:
            special.append("Emergency: decide within the 1-hour cashless target")

        T = lambda tid, agent, goal, when="always", **kw: {"id": tid, "agent": agent, "goal": goal, "when": when, **kw}  # noqa: E731
        tasks = [
            T("check_documents", "Document Intake", f"Check the {len(BASE_DOCS)} mandatory documents"),
            T("preprocess_text", "Document Intake", "Clean notes: normalise, expand abbreviations, mask PII, split, tag, negation, durations"),
            T("extract_facts", "Document Intake", "LLM extraction to JSON schema, validate, ground in text, cross-check with rules"),
            T("verify_accident_docs", "Document Intake", "Accident case: require MLC / FIR copy", when="accident"),
            T("research_policy", "Policy Research", "Fetch policy schedule, retrieve clauses, tariff benchmark"),
            T("check_eligibility", "Coverage Executor", "Member, policy period, waiting periods, exclusions, PED"),
            T("calculate_payable", "Coverage Executor", "Room-rent ratio, ICU cap, sub-limit, co-pay, balance sum insured", when="eligible"),
            T("validate", "Critic", "Independent re-checks; PASS / REVISE (back to Executor) / HOLD", loop="REVISE → calculate_payable (max 2)"),
            T("decide", "Decision & Communication", "Apply decision ladder: Deny > Escalate > Query > Partial > Approve"),
            T("draft_letter", "Decision & Communication", "Template → de-identify → LLM polish → re-identify → faithfulness check"),
            T("human_review", "Medical officer (human)", "Review escalated case and approve / deny / query", when="escalated"),
        ]
        plan = {
            "priority": "HIGH (emergency, 1-hour target)" if emergency else "NORMAL (planned)",
            "claim_profile": f"{claim['admission_type']} · {claim['treatment_type']} · age {claim['age']}",
            "required_docs": list(BASE_DOCS),
            "special_checks": special,
            "tasks": tasks,
        }
        ctx["plan"] = plan
        n_cond = sum(t["when"] != "always" for t in tasks)
        summary = (f"Priority {plan['priority']}. {len(tasks)} tasks planned ({n_cond} conditional); "
                   f"{len(special)} claim-specific check{'s' if len(special) != 1 else ''}.")
        return {"summary": summary, "details": plan}
