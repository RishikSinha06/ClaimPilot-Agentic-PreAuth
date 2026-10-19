"""Decision & Communication Agent.

decide        apply the decision ladder to everything the other agents produced:
              Deny (only on grounds that hold under both extractors' facts)
              > Escalate > Query > Partial > Approve
draft_letter  template letter -> de-identify -> LLM polish -> re-identify ->
              faithfulness guardrail (falls back to the template if it fails)
apply_review  record a human medical officer's decision on an escalated case
"""
from __future__ import annotations

from . import guardrails
from .critic import deny_grounds
from .intake import DOC_LABELS
from .nlp.prompts import build_letter_prompt, clean_letter_output
from .tools import inr, total_cost

DECISION_META = {
    "APPROVE": ("Approved", "Cashless authorisation granted"),
    "PARTIAL": ("Partially approved", "Cashless authorisation granted for a reduced amount"),
    "QUERY": ("Query raised", "Additional information needed before a decision"),
    "DENY": ("Denied", "Cashless request not admissible under the policy"),
    "ESCALATE": ("Escalated to medical officer", "Held for human review"),
}

# Critic check that caused the escalation -> letter reason for each review action.
REVIEW_REASONS = {
    "PED disclosure": {
        "approve": "Pre-existing condition reviewed by medical officer and accepted",
        "deny": "Non-disclosure of a material pre-existing disease at proposal (clause 4.3), confirmed by medical officer",
        "query": "Please submit past treatment records and prescriptions for the conditions noted",
    },
    "Extractor agreement": {
        "approve": "Clinical facts in the treating doctor's notes verified by medical officer",
        "deny": "Treatment not admissible on the clinical facts, as assessed by medical officer",
        "query": "Please submit clarified treating doctor's notes stating the diagnosis, procedure and medical history",
    },
    "Rejection grounds": {
        "approve": "Clinical facts in the treating doctor's notes verified by medical officer",
        "deny": "Treatment not admissible on the clinical facts, as assessed by medical officer",
        "query": "Please submit clarified treating doctor's notes stating the diagnosis, procedure and medical history",
    },
    "Amount re-verification": {
        "approve": "Payable amount reviewed by medical officer; the lower of the two independently computed amounts is authorised",
        "deny": "Cost estimate not admissible as submitted, as assessed by medical officer",
        "query": "Please submit an itemised cost estimate",
    },
    "Amount sanity": {
        "approve": "Payable amount reviewed by medical officer",
        "deny": "Cost estimate not admissible as submitted, as assessed by medical officer",
        "query": "Please submit an itemised cost estimate",
    },
    "Authority limit": {
        "approve": "Amount above the auto-approval limit reviewed and approved by medical officer",
        "deny": "Amount above the auto-approval limit not approved by medical officer",
        "query": "Please submit an itemised cost estimate and the treating doctor's justification for the treatment",
    },
}


def escalation_causes(critic: dict) -> list[str]:
    """Names of the Critic checks that asked for escalation, in check order."""
    return list(dict.fromkeys(f["check"] for f in critic["findings"] if f["action"] == "escalate"))


def review_reasons(causes: list[str], action: str) -> list[str]:
    generic = {"approve": "{} reviewed and accepted by medical officer",
               "deny": "Not admissible after medical officer review of: {}",
               "query": "Please submit supporting documents for: {}"}
    out = [REVIEW_REASONS[c][action] if c in REVIEW_REASONS else generic[action].format(c) for c in causes]
    return list(dict.fromkeys(out)) or [generic[action].format("the escalated case")]


def template_letter(claim, decision, amount, reasons, reviewer_note=None):
    head = (f"To: Cashless Desk, {claim['hospital']}\n"
            f"Ref: Pre-authorisation {claim['claim_id']} · Policy {claim['policy_no']} · Patient {claim['patient']}\n\n")
    body = {
        "APPROVE": f"We are pleased to authorise cashless treatment for {claim['proposed_treatment']} "
                   f"up to {inr(amount)}. Final settlement will be on the actual itemised bill.",
        "PARTIAL": f"We authorise cashless treatment for {claim['proposed_treatment']} up to {inr(amount)}. "
                   "The balance of the estimate is not payable for the reasons below and is to be collected from the patient.",
        "QUERY": "We need the following before we can decide on this request. Please upload them on the portal "
                 "and the request will be re-processed immediately.",
        "DENY": f"We regret that cashless treatment for {claim['proposed_treatment']} cannot be authorised. "
                "The patient may still undergo treatment, but the expenses are not payable under the policy for the reasons below.",
        "ESCALATE": "This request has been referred to our medical officer for review. You will receive a decision "
                    "shortly; no action is needed from the hospital at this stage.",
    }[decision]
    rs = "\n".join(f"  • {r}" for r in reasons)
    note = f"\n\nMedical officer remarks: {reviewer_note}" if reviewer_note else ""
    return f"{head}{body}\n\nReasons:\n{rs}{note}\n\nRegards,\nPre-Authorisation Team (AI-assisted)"


class DecisionAgent:
    name = "Decision & Communication Agent"
    short = "Decision & Letter"
    icon = "✅"

    def __init__(self, llm):
        self.llm = llm

    # ------------------------------------------------------------ decide
    def decide(self, ctx: dict) -> dict:
        critic, elig, calc = ctx["critic"], ctx["eligibility"], ctx.get("calculation")
        actions = set(critic["actions"])
        reasons = []
        grounds = deny_grounds(elig, critic.get("alt_eligibility"))
        if grounds:
            decision, amount = "DENY", 0
            reasons += [f"{c['check']}: {c['detail']}" + (f" (clause {c['clause']})" if c["clause"] else "") for c in grounds]
        elif "escalate" in actions or not elig["eligible"]:
            # an ineligible claim without robust grounds means the denial rests on disputed facts
            decision, amount = "ESCALATE", calc["payable"] if calc else 0
            reasons += [f["detail"] for f in critic["findings"] if f["action"] == "escalate"]
        elif "query" in actions:
            decision, amount = "QUERY", 0
            if ctx["missing_docs"]:
                reasons.append("Please submit: " + ", ".join(DOC_LABELS.get(d, d) for d in ctx["missing_docs"]))
            reasons += [f["detail"] for f in critic["findings"] if f["action"] == "query" and f["check"] != "Document completeness"]
        elif calc["payable"] < calc["requested"]:
            decision, amount = "PARTIAL", calc["payable"]
            reasons += [f"{s['step']}: {inr(-s['amount'])} — {s['note']}" for s in calc["steps"] if s["amount"] < 0]
        else:
            decision, amount = "APPROVE", calc["payable"]
            reasons.append("All eligibility checks passed and the estimate is within network tariff")
        if decision in ("APPROVE", "PARTIAL"):
            reasons += [f["detail"] for f in critic["findings"] if f["status"] == "WARN" and f["action"] is None
                        and f["check"] in ("Tariff benchmark",)]

        label, sub = DECISION_META[decision]
        ctx["decision"] = {"decision": decision, "label": label, "subtitle": sub, "amount": amount,
                           "requested": total_cost(ctx["claim"]["costs"]), "reasons": reasons,
                           "human_review": decision == "ESCALATE", "confidence": critic["confidence"],
                           "decided_by": "agents"}
        if decision == "ESCALATE":
            ctx["decision"]["escalation_causes"] = escalation_causes(critic)
        return {"summary": f"{label}: {inr(amount)}. {len(reasons)} reason(s) recorded.",
                "details": ctx["decision"]}

    # ------------------------------------------------------------ letter
    def draft_letter(self, ctx: dict) -> dict:
        claim, d = ctx["claim"], ctx["decision"]
        template = template_letter(claim, d["decision"], d["amount"], d["reasons"], d.get("reviewer_note"))
        deid, mapping = guardrails.deidentify_text(template, claim)
        system, prompt = build_letter_prompt(deid)
        info = {"template": template, "deidentified": deid, "method": "template", "problems": []}
        letter = template
        ok, leaks = guardrails.pii_gate(system + prompt, mapping, claim)
        if not ok:
            info["method"] = f"template (PII gate blocked the LLM call: {', '.join(leaks)})"
        else:
            try:
                res = self.llm.generate(prompt, system=system, purpose="polish letter")
                candidate = guardrails.reidentify(clean_letter_output(res.text), mapping)
                must_keep = d["reasons"] + ([d["reviewer_note"]] if d.get("reviewer_note") else [])
                faithful, problems = guardrails.letter_faithful(template, candidate, must_keep=must_keep,
                                                                decision=d["decision"])
                info["problems"] = problems
                info["llm_raw"] = res.text
                if faithful and candidate:
                    letter = candidate
                    info["method"] = "Mock LLM (simulated: returns the template unchanged)" if self.llm.is_mock else self.llm.name
                else:
                    info["method"] = "template (LLM letter rejected by faithfulness check)"
            except Exception as e:  # noqa: BLE001
                info["method"] = f"template (LLM unavailable: {type(e).__name__})"
        d["letter"] = letter
        d["letter_method"] = info["method"]
        ctx["letter_info"] = info
        return {"summary": f"Letter drafted via {info['method']}." + (f" Issues: {'; '.join(info['problems'])}" if info["problems"] else ""),
                "details": {k: v for k, v in info.items() if k != "template"}}

    # ------------------------------------------------------------ human review
    def apply_review(self, ctx: dict, action: str, remarks: str, reviewer: str) -> dict:
        d, calc = ctx["decision"], ctx.get("calculation")
        before = d["decision"]
        causes = d.get("escalation_causes") or escalation_causes(ctx["critic"])
        reasons = review_reasons(causes, action)
        if action == "approve":
            amount = calc["payable"] if calc else 0
            recomputed = ctx["critic"].get("recomputed")
            if "Amount re-verification" in causes and recomputed is not None and recomputed < amount:
                reasons.append(f"Authorised the Critic's independent recomputation {inr(recomputed)} "
                               f"instead of the Executor's {inr(amount)}")
                amount = recomputed
            if calc:
                amount = max(0, min(amount, calc["requested"]))
            new = "PARTIAL" if calc and amount < calc["requested"] else "APPROVE"
            reasons += [f"{s['step']}: {inr(-s['amount'])} — {s['note']}" for s in (calc["steps"] if calc else []) if s["amount"] < 0]
        elif action == "deny":
            new, amount = "DENY", 0
        else:
            new, amount = "QUERY", 0
        label, sub = DECISION_META[new]
        d.update({"decision": new, "label": label, "subtitle": sub + " · after human review", "amount": amount,
                  "reasons": reasons, "human_review": False, "decided_by": f"human: {reviewer}",
                  "reviewer_note": remarks.strip() or None, "overridden_from": before})
        return {"summary": f"Medical officer {reviewer}: {before} → {new} ({inr(amount)}).",
                "details": {"action": action, "remarks": remarks, "from": before, "to": new}}
