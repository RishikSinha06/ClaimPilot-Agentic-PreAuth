"""Coverage Executor Agent: runs the eligibility rules and computes the
payable amount by calling deterministic tools."""
from __future__ import annotations

from .tools import cost_lines, inr, months_between, to_date, total_cost
from .trace import record

# Which chronic conditions are clinically related to which procedures (for PED).
PED_RELATED = {
    "hypertension": ["angioplasty"],
    "diabetes": ["angioplasty", "cataract"],
    "osteoarthritis": ["joint replacement"],
    "chronic kidney disease": ["kidney stone"],
}


def chk(name, result, detail, clause=""):
    return {"check": name, "result": result, "detail": detail, "clause": clause}


def assess_eligibility(claim: dict, policy: dict | None, ex: dict) -> dict:
    """The eligibility rules as a pure function of (claim, policy, facts).

    The Executor runs it on the facts handed over by Intake. The Critic re-runs it
    on the rule-based facts to see whether an LLM-vs-rules disagreement changes
    the outcome."""
    checks = []
    if policy is None:
        checks.append(chk("Policy exists", "FAIL", f"{claim['policy_no']} not found"))
        return {"eligible": False, "checks": checks, "undisclosed_ped": []}

    req = to_date(claim["request_date"])
    start, end = to_date(policy["start_date"]), to_date(policy["end_date"])
    tenure_days = (req - start).days
    tenure_m = months_between(start, req)
    record("tool", "date_tools", f"policy age {tenure_days} days / {tenure_m} months")
    proc = ex["procedure_key"]
    accident = ex["accident"]
    waived = policy["waiting_waived"]

    checks.append(chk("Member covered",
                      "PASS" if claim["patient"] in policy["insured"] else "FAIL",
                      f"{claim['patient']} {'is' if claim['patient'] in policy['insured'] else 'is NOT'} on policy"))

    active = start <= req <= end
    checks.append(chk("Policy in force", "PASS" if active else "FAIL",
                      f"Period {policy['start_date']} to {policy['end_date']}; request on {claim['request_date']}", "2.1"))

    if waived:
        checks.append(chk("Initial waiting period", "PASS", "Waived under group policy", "4.1"))
    elif accident:
        checks.append(chk("Initial waiting period", "PASS", "Accident claim: waiting period not applicable", "4.1"))
    else:
        ok = tenure_days >= policy["initial_waiting_days"]
        checks.append(chk("Initial waiting period", "PASS" if ok else "FAIL",
                          f"Policy age {tenure_days} days vs {policy['initial_waiting_days']} required", "4.1"))

    if proc in policy["specific_waiting_conditions"]:
        if waived:
            checks.append(chk("Specific waiting period", "PASS", "Waived under group policy", "4.2"))
        else:
            ok = tenure_m >= policy["specific_waiting_months"]
            checks.append(chk("Specific waiting period", "PASS" if ok else "FAIL",
                              f"'{proc}' needs {policy['specific_waiting_months']} months; policy age {tenure_m} months", "4.2"))
    else:
        checks.append(chk("Specific waiting period", "PASS", f"'{proc}' not on specific waiting list", "4.2"))

    excluded = None
    if (proc == "rhinoplasty" or ex["cosmetic_intent"]) and not (ex["functional_need"] or accident):
        excluded = "cosmetic"
    for e in policy["exclusions"]:
        if e in proc and not (e == "dental" and accident):  # clause 5.2: dental after an accident is covered
            excluded = e
    checks.append(chk("Permanent exclusions", "FAIL" if excluded else "PASS",
                      f"Falls under '{excluded}' exclusion" if excluded else "No exclusion triggered",
                      {"cosmetic": "5.1", "dental": "5.2"}.get(excluded, "")))

    undisclosed = []
    for c in ex["chronic_conditions"]:
        cond = c["condition"]
        related = proc in PED_RELATED.get(cond, [])
        declared = cond in policy["declared_ped"]
        yrs = c.get("duration_years")
        pre_existing = yrs is not None and yrs * 12 > tenure_m
        if declared and related:
            ok = waived or tenure_m >= policy["ped_waiting_months"]
            checks.append(chk(f"PED waiting: {cond}", "PASS" if ok else "FAIL",
                              f"Declared PED; needs {policy['ped_waiting_months']} months, policy age {tenure_m}", "4.3"))
        elif pre_existing and not declared:
            undisclosed.append(cond)
            checks.append(chk(f"PED disclosure: {cond}", "WARN",
                              f"Present ~{yrs:g} yrs (before inception) but not declared at proposal"
                              + ("; clinically related to this treatment" if related else ""), "4.3"))
    if not ex["chronic_conditions"]:
        checks.append(chk("Pre-existing diseases", "PASS", "No chronic conditions in notes", "4.3"))

    eligible = all(c["result"] != "FAIL" for c in checks)
    return {"eligible": eligible, "checks": checks, "undisclosed_ped": undisclosed}


class CoverageExecutorAgent:
    name = "Coverage Executor Agent"
    short = "Coverage Executor"
    icon = "⚙️"

    # ---------- task 1: eligibility ----------
    def eligibility(self, ctx: dict) -> dict:
        ctx["eligibility"] = elig = assess_eligibility(ctx["claim"], ctx["policy"], ctx["extracted"])
        if ctx["policy"] is None:
            return {"summary": "Policy not found. Not eligible.", "details": elig}
        checks = elig["checks"]
        fails = [c["check"] for c in checks if c["result"] == "FAIL"]
        warns = [c["check"] for c in checks if c["result"] == "WARN"]
        summary = (f"{len(checks)} checks run. " +
                   ("ELIGIBLE." if elig["eligible"] else f"NOT ELIGIBLE: {', '.join(fails)}.") +
                   (f" Warnings: {', '.join(warns)}." if warns else ""))
        return {"summary": summary, "details": elig}

    # ---------- task 2: payable amount ----------
    def calculate(self, ctx: dict, revision: dict | None = None, facts: dict | None = None) -> dict:
        """`facts` overrides the Intake facts (used when a medical officer approves a case
        whose LLM-extracted facts were disputed)."""
        claim, policy = ctx["claim"], ctx["policy"]
        ex = facts or ctx["extracted"]
        costs = claim["costs"]
        lines = cost_lines(costs)
        requested = total_cost(costs)
        steps = [{"step": "Requested estimate", "amount": requested, "note": "Sum of hospital cost lines"}]

        # 1. Room-rent proportionate deduction (IRDAI-style: not on pharmacy/diagnostics)
        cap = policy["room_rent_cap_per_day"]
        ratio = min(1.0, cap / costs["room_rent_per_day"]) if costs["room_rent_per_day"] else 1.0
        affected = lines["room"] + lines["surgery_ot"] + lines["other"]
        prop_cut = round(affected * (1 - ratio))
        icu_per_day, icu_days = costs.get("icu_per_day", 0), costs.get("icu_days", 0)
        icu_cut = max(0, icu_per_day - policy["icu_cap_per_day"]) * icu_days
        payable = requested - prop_cut - icu_cut
        if prop_cut:
            steps.append({"step": "Room-rent proportionate deduction", "amount": -prop_cut,
                          "note": f"Room {inr(costs['room_rent_per_day'])}/day vs cap {inr(cap)}/day → "
                                  f"{ratio:.0%} of room, surgery/OT and other charges (clause 3.4)"})
        if icu_cut:
            steps.append({"step": "ICU rent above cap", "amount": -icu_cut, "note": "Clause 3.4"})

        # 2. Sub-limit
        proc = ex["procedure_key"]
        if proc in policy["sub_limits"] and payable > policy["sub_limits"][proc]:
            cut = payable - policy["sub_limits"][proc]
            payable -= cut
            steps.append({"step": f"Sub-limit for {proc}", "amount": -cut,
                          "note": f"Capped at {inr(policy['sub_limits'][proc])} (clause 3.6)"})

        # 3. Reasonable & customary cap requested by the Critic
        if revision and revision.get("cap_at_benchmark") and ctx.get("benchmark"):
            b = ctx["benchmark"]
            days = costs["room_days"] + icu_days
            limit = b["max"] * (days if b["basis"] == "per_day" else 1)
            if payable > limit:
                cut = payable - limit
                payable -= cut
                steps.append({"step": "Reasonable & customary cap", "amount": -cut,
                              "note": f"Capped at network tariff ceiling {inr(limit)} (clause 6.3), on Critic's request"})

        # 4. Co-pay
        if policy["copay_pct"]:
            cut = round(payable * policy["copay_pct"] / 100)
            payable -= cut
            steps.append({"step": f"Co-pay {policy['copay_pct']}%", "amount": -cut, "note": "Borne by insured (clause 3.5)"})

        # 5. Balance sum insured
        balance = policy["sum_insured"] - policy["utilised"]
        if payable > balance:
            cut = payable - balance
            payable -= cut
            steps.append({"step": "Balance sum insured", "amount": -cut, "note": f"Only {inr(balance)} left"})

        steps.append({"step": "Payable (provisional)", "amount": payable, "note": ""})
        record("tool", "money_calculator", f"{len(steps) - 2} adjustment(s) → {inr(payable)}")
        ctx["calculation"] = {"requested": requested, "payable": payable, "steps": steps,
                              "room_ratio": ratio, "revision_applied": bool(revision),
                              "facts": "rule-based" if facts else "intake"}
        summary = (f"Requested {inr(requested)} → payable {inr(payable)} after {len(steps) - 2} adjustment(s)"
                   + (" [revised]" if revision else "") + (" [on rule-based facts]" if facts else "") + ".")
        return {"summary": summary, "details": ctx["calculation"]}
