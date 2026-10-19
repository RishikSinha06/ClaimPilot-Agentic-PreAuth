"""Policy Research Agent: gathers the policy schedule, the relevant policy
wording clauses (keyword retrieval) and the network tariff benchmark."""
from __future__ import annotations

from .tools import clause_search, inr, policy_lookup, tariff_lookup


class PolicyResearchAgent:
    name = "Policy Research Agent"
    short = "Policy Research"
    icon = "🔎"

    def run(self, ctx: dict) -> dict:
        claim, ex = ctx["claim"], ctx["extracted"]
        policy = policy_lookup(claim["policy_no"])
        ctx["policy"] = policy
        if policy is None:
            ctx["clauses"], ctx["benchmark"] = [], None
            return {"summary": f"Policy {claim['policy_no']} not found in policy master.", "details": {}}

        proc = ex["procedure_key"]
        terms = ["active", "waiting", proc, "room", "documents", "tariff"]
        if policy["copay_pct"]:
            terms.append("copay")
        if proc in policy["sub_limits"]:
            terms.append("sub-limit")
        if ex.get("cosmetic_intent") or proc == "rhinoplasty":
            terms.append("cosmetic")
        if "dental" in proc:
            terms.append("dental")
        if ex.get("chronic_conditions"):
            terms.append("ped")
        clauses = clause_search(terms, k=6)
        ctx["clauses"] = clauses

        bench = tariff_lookup(proc)
        ctx["benchmark"] = bench
        bench_txt = (f"{bench['label']}: {inr(bench['min'])}–{inr(bench['max'])}"
                     + (" per day" if bench["basis"] == "per_day" else "")) if bench else "no benchmark"
        balance = policy["sum_insured"] - policy["utilised"]
        summary = (f"{policy['product']} · SI {inr(policy['sum_insured'])} · balance {inr(balance)}. "
                   f"Clauses {', '.join(c['id'] for c in clauses)}. Tariff: {bench_txt}.")
        return {"summary": summary,
                "details": {"policy": policy, "clauses": [f"{c['id']} {c['title']}" for c in clauses], "benchmark": bench}}
