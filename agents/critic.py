"""Critic Agent: independently validates everything the other agents produced
and returns a verdict:

    PASS    -> forward to the Decision agent
    REVISE  -> send back to the Coverage Executor with an instruction (loop)
    HOLD    -> needs a query to the hospital or a human medical officer

Checks: documents · ICD-10 vs diagnosis · LLM output validity · grounding ·
LLM-vs-rules agreement · PII protection · PED disclosure · tariff benchmark ·
independent amount recomputation · amount sanity · authority limit.
"""
from __future__ import annotations

from .executor import assess_eligibility
from .intake import DOC_LABELS
from .tools import inr
from .trace import record

# A disagreement on these fields always needs a human: the procedure drives sub-limits, tariffs and
# exclusions, and the accident flag drives waiting periods and the MLC / FIR requirement. Other fields
# (cosmetic intent, functional need, chronic conditions) only matter if they change the eligibility outcome.
ALWAYS_MATERIAL = ("procedure_key", "accident")


def outcome_changes(llm_elig: dict, rules_elig: dict) -> list[str]:
    """How the eligibility outcome differs between the LLM facts and the rule-based facts."""
    def flagged(e):
        return {c["check"]: c for c in e["checks"] if c["result"] in ("FAIL", "WARN")}
    a, b = flagged(llm_elig), flagged(rules_elig)
    out = []
    if llm_elig["eligible"] != rules_elig["eligible"]:
        out.append("eligible only under the " + ("LLM" if llm_elig["eligible"] else "rule-based") + " facts")
    for name in sorted(set(a) | set(b)):
        ca, cb = a.get(name), b.get(name)
        if ca == cb:
            continue
        same_result = ca and cb and ca["result"] == cb["result"]
        show = (lambda c: c["detail"]) if same_result else (lambda c: c["result"] if c else "PASS")
        out.append(f"{name}: {show(ca)} (LLM) vs {show(cb)} (rules)")
    return out


def deny_grounds(elig: dict, alt: dict | None = None) -> list[dict]:
    """Failed eligibility checks that can justify a denial. When the two extractors
    disagree (`alt` = eligibility on the rule-based facts), only checks that fail
    identically under both sets of facts count, so a misread note cannot deny a claim."""
    fails = [c for c in elig.get("checks", []) if c["result"] == "FAIL"]
    return [c for c in fails if c in alt["checks"]] if alt is not None else fails

# ICD-10 category -> words expected in the diagnosis text
ICD_MAP = {
    "K35": ["appendic"], "H25": ["cataract"], "M17": ["osteoarthritis", "knee"],
    "K40": ["hernia"], "K42": ["hernia"], "Z41": ["cosmetic", "aesthetic", "hump", "nasal"], "A97": ["dengue"],
    "I25": ["coronary", "ischaemic", "ischemic"], "K80": ["cholelith", "gallstone"],
    "J18": ["pneumonia"], "S72": ["femur", "hip"], "K01": ["molar", "tooth", "dental"],
}


def independent_recompute(claim, policy, procedure_key, benchmark, capped):
    """Second, separately written calculation used only for verification."""
    c = claim["costs"]
    cap = policy["room_rent_cap_per_day"]
    r = 1.0 if c["room_rent_per_day"] <= cap else cap / c["room_rent_per_day"]
    eligible_room = min(c["room_rent_per_day"], cap) * c["room_days"]
    eligible_icu = min(c.get("icu_per_day", 0), policy["icu_cap_per_day"]) * c.get("icu_days", 0)
    amt = (eligible_room + eligible_icu + c["surgery_ot"] * r + c["other"] * r
           + c["medicines_consumables"] + c["investigations"])
    amt = min(amt, policy["sub_limits"].get(procedure_key, float("inf")))
    if capped and benchmark:
        days = c["room_days"] + c.get("icu_days", 0)
        amt = min(amt, benchmark["max"] * (days if benchmark["basis"] == "per_day" else 1))
    amt = amt * (1 - policy["copay_pct"] / 100)
    return round(min(amt, policy["sum_insured"] - policy["utilised"]))


class CriticAgent:
    name = "Critic Agent"
    short = "Critic"
    icon = "🧐"

    def __init__(self, tariff_tolerance_pct=25, auto_approve_limit=300000):
        self.tol = tariff_tolerance_pct
        self.auto_limit = auto_approve_limit

    def run(self, ctx: dict) -> dict:
        claim, policy = ctx["claim"], ctx["policy"]
        ex, elig = ctx["extracted"], ctx.get("eligibility", {})
        calc, bench = ctx.get("calculation"), ctx.get("benchmark")
        llm = ctx.get("llm_extraction", {})
        findings = []
        recomputed = None

        def add(check, status, detail, severity="info", action=None):
            findings.append({"check": check, "status": status, "severity": severity, "detail": detail, "action": action})

        # 1. Documents
        if ctx["missing_docs"]:
            add("Document completeness", "FAIL", "Missing: " + ", ".join(DOC_LABELS.get(d, d) for d in ctx["missing_docs"]), "high", "query")
        else:
            add("Document completeness", "PASS", "All mandatory documents present")

        # 2. ICD-10 code vs diagnosis text
        words = ICD_MAP.get(claim["icd10"][:3])
        if words is None:
            add("ICD-10 consistency", "WARN", f"Code {claim['icd10']} not in reference table", "low")
        elif any(w in claim["diagnosis"].lower() for w in words):
            add("ICD-10 consistency", "PASS", f"{claim['icd10']} matches '{claim['diagnosis']}'")
        else:
            add("ICD-10 consistency", "FAIL", f"{claim['icd10']} does not match '{claim['diagnosis']}'", "medium", "query")
        record("tool", "icd10_lookup", claim["icd10"])

        # 3. LLM output validity
        attempts = llm.get("attempts", [])
        if llm.get("fallback"):
            add("LLM output validity", "WARN", llm["fallback"], "medium")
        elif len(attempts) > 1:
            add("LLM output validity", "WARN", f"Valid JSON after {len(attempts)} attempts (repair prompt used)", "low")
        else:
            add("LLM output validity", "PASS", "Schema-valid JSON on first attempt")

        # 4. Grounding (hallucination guard)
        grounding = llm.get("grounding", [])
        rejected = [g for g in grounding if g["status"].startswith("rejected")]
        corrected = [g for g in grounding if "duration corrected" in g["status"]]
        if rejected or corrected:
            add("Evidence grounding", "WARN", "; ".join(
                (["Dropped unsupported LLM claims: " + ", ".join(f"{g['condition']} ({g['status']})" for g in rejected)] if rejected else [])
                + (["Replaced LLM durations with the ones stated in the notes: " + ", ".join(g["condition"] for g in corrected)] if corrected else [])),
                "low")
        elif grounding:
            add("Evidence grounding", "PASS", "Every extracted condition is supported by a cited sentence")

        # 5. LLM vs rules agreement, and whether a disagreement changes the outcome
        diffs = [r for r in llm.get("compare", []) if not r["agree"]]
        alt = None
        if diffs and ctx.get("extracted_rules"):
            alt = assess_eligibility(claim, policy, ctx["extracted_rules"])  # what the rule-based facts would give
        always = [r["field"] for r in diffs if r["field"] in ALWAYS_MATERIAL]
        changes = outcome_changes(elig, alt) if alt and elig.get("checks") else []
        fields = ", ".join(r["field"] for r in diffs)
        if always or changes:
            add("Extractor agreement", "FAIL", f"LLM and rule-based extractors disagree on {fields}. "
                + ("Under the rule-based facts: " + "; ".join(changes) if changes else
                   "The procedure and accident flag drive limits, exclusions and documents") + ". Needs a medical officer",
                "medium", "escalate")
        elif diffs:
            add("Extractor agreement", "WARN", f"Minor disagreement on {fields}: the eligibility outcome is the same under both", "low")
        elif ctx.get("llm_simulated"):
            add("Extractor agreement", "PASS", "Agree on every field. Expected: the simulated LLM reuses the rule extractor, "
                "so this is not independent evidence")
        else:
            add("Extractor agreement", "PASS", "LLM and rule-based extractors agree on every field")

        # 6. PII protection
        if llm.get("pii_gate", "pass") == "pass":
            add("PII protection", "PASS", "No patient identifiers were sent to the LLM")
        else:
            add("PII protection", "FAIL", f"Prompt blocked: {llm.get('pii_gate')}", "medium")

        # 7. Disclosure / PED
        if elig.get("undisclosed_ped"):
            add("PED disclosure", "FAIL", f"Undisclosed pre-existing condition(s): {', '.join(elig['undisclosed_ped'])}. "
                "Needs medical officer review before any approval", "high", "escalate")

        if calc and policy:
            req = calc["requested"]
            # 8. Tariff benchmark
            days = claim["costs"]["room_days"] + claim["costs"].get("icu_days", 0)
            ceiling = bench["max"] * (days if bench["basis"] == "per_day" else 1) if bench else 0
            if ceiling > 0:
                over = (req - ceiling) / ceiling * 100
                if over > self.tol and not calc["revision_applied"]:
                    add("Tariff benchmark", "FAIL", f"Estimate {inr(req)} is {over:.0f}% above network ceiling {inr(ceiling)} "
                        f"(tolerance {self.tol}%). Sending back to Executor to cap at reasonable & customary", "high", "revise")
                elif over > self.tol:
                    add("Tariff benchmark", "WARN", f"Estimate {over:.0f}% above ceiling; payable already capped. Ask hospital for itemised bill", "medium")
                else:
                    add("Tariff benchmark", "PASS", f"Estimate within {self.tol}% of network ceiling {inr(ceiling)}")
            # 9. Maths re-verification
            check = recomputed = independent_recompute(claim, policy, ex["procedure_key"], bench, calc["revision_applied"])
            record("tool", "independent_recompute", inr(check))
            if abs(check - calc["payable"]) <= 1:
                add("Amount re-verification", "PASS", f"Independent recomputation agrees: {inr(check)}")
            else:
                add("Amount re-verification", "FAIL", f"Executor {inr(calc['payable'])} vs Critic {inr(check)}", "high", "escalate")
            # 10. Sanity
            ok = 0 <= calc["payable"] <= req
            add("Amount sanity", "PASS" if ok else "FAIL", "0 ≤ payable ≤ requested", "info" if ok else "high", None if ok else "escalate")
            # 11. Authority limit
            if calc["payable"] > self.auto_limit:
                add("Authority limit", "WARN", f"Payable above auto-approval limit {inr(self.auto_limit)}", "medium", "escalate")

        if not elig.get("eligible", True):
            grounds = deny_grounds(elig, alt)
            if grounds:
                add("Rejection grounds", "PASS", "Denial grounded in policy" + (" under both extractors' facts" if alt else "") + ": "
                    + ", ".join(f"{c['check']} ({c['clause'] or 'schedule'})" for c in grounds))
            else:
                fails = [c for c in elig["checks"] if c["result"] == "FAIL"]
                add("Rejection grounds", "FAIL", "The failed checks hold only under the LLM-extracted facts, not the rule-based ones: "
                    + ", ".join(c["check"] for c in fails) + ". A denial needs a medical officer", "high", "escalate")

        penalty = {"info": 0, "low": 0.05, "medium": 0.15, "high": 0.3}
        confidence = max(0.0, 1 - sum(penalty[f["severity"]] for f in findings if f["status"] != "PASS"))
        actions = {f["action"] for f in findings if f["action"]}
        verdict = "REVISE" if "revise" in actions else ("HOLD" if actions else "PASS")
        result = {"verdict": verdict, "confidence": round(confidence, 2), "findings": findings, "actions": sorted(actions),
                  "recomputed": recomputed, "alt_eligibility": alt}
        if verdict == "REVISE":
            result["instruction"] = {"cap_at_benchmark": True}
        ctx["critic"] = result
        ctx.setdefault("critic_history", []).append(result)
        n_fail = sum(f["status"] == "FAIL" for f in findings)
        summary = (f"Verdict {verdict} · confidence {confidence:.0%} · {len(findings)} checks, {n_fail} failed"
                   + (f" · actions: {', '.join(sorted(actions))}" if actions else "") + ".")
        return {"summary": summary, "details": result}
