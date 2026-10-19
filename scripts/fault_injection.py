"""Fault injection: corrupt the LLM's answers on purpose and check that no wrong
decision and no unfaithful letter gets out.

Usage:
    python scripts/fault_injection.py                 # simulated LLM as the base model (offline, ~10 s)
    python scripts/fault_injection.py --strict        # exit code 1 if anything wrong gets out
    python scripts/fault_injection.py --ollama qwen2.5:7b   # corrupt a real model's answers (slow on CPU)

Every fault in agents/faults.py is run against every labelled case. Extraction
faults are scored on the decision that goes out:
    correct  same decision (and amount) as the label
    held     a human or the hospital is asked instead (ESCALATE / QUERY): safe, extra work
    wrong    a different decision went out
Letter faults are scored on whether the corrupted letter was blocked (the
template letter is sent instead). Cases where a fault changes nothing (e.g.
"drop conditions" when there are none) are not counted.
Writes results/fault_injection.md and .json.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agents import Orchestrator, make_llm  # noqa: E402
from agents.evaluation import score  # noqa: E402
from agents.faults import FAULTS, FaultyLLM  # noqa: E402


def caught_by(ctx: dict) -> str:
    """Which safeguard dealt with an extraction fault."""
    info = ctx["llm_extraction"]
    if info["fallback"]:
        return "schema validation → rule fallback"
    if any(g["status"].startswith("rejected") or "corrected" in g["status"] for g in info["grounding"]):
        return "grounding"
    findings = {f["check"]: f for f in ctx["critic"]["findings"]}
    if findings.get("Rejection grounds", {}).get("status") == "FAIL" or findings.get("Extractor agreement", {}).get("status") == "FAIL":
        if ctx["decision"]["decision"] == "DENY":
            return "Critic: denial kept only on grounds both extractors support"
        return "Critic: extractors disagree → human"
    if findings.get("Extractor agreement", {}).get("status") == "WARN":
        return "Critic: disagreement does not change the outcome"
    return "-"


def run(base, claims) -> list[dict]:
    rows = []
    for fault, (target, _) in FAULTS.items():
        for c in claims:
            llm = FaultyLLM(base, fault)
            ctx = Orchestrator(llm=llm).run(c)
            if not llm.applied:
                continue
            s = score(c, ctx)
            row = {"fault": fault, "target": target, "claim_id": c["claim_id"], "expected": c["expected_decision"],
                   "decision": s["decision"], "outcome": s["outcome"]}
            if target == "letter":
                method = ctx["decision"]["letter_method"]
                row["letter"] = "blocked" if method.startswith("template") else "SENT"
                row["caught_by"] = "faithfulness check" if row["letter"] == "blocked" else "-"
                row["problems"] = ctx["letter_info"]["problems"]
            else:
                row["caught_by"] = caught_by(ctx)
            rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strict", action="store_true", help="exit 1 if a wrong decision or a corrupted letter gets out")
    ap.add_argument("--ollama", metavar="MODEL", help="corrupt a real Ollama model's answers instead of the simulated LLM")
    ap.add_argument("--host", default="http://localhost:11434")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    base, msg = make_llm("ollama" if args.ollama else "mock", args.ollama or "llama3.2", args.host, timeout=900)
    if args.ollama and base.is_mock:
        sys.exit(f"Ollama model not available: {msg}")
    claims = json.loads((ROOT / "data" / "claims.json").read_text(encoding="utf-8"))
    rows = run(base, claims)

    md = [f"Base model: {base.name}. {len(claims)} labelled cases × {len(FAULTS)} faults; "
          "only runs where the fault changed the LLM's answer are counted.", "",
          "| Fault | What it does | Runs | Correct | Held for review | Wrong decision out | Letter blocked | Caught by |",
          "|---|---|---|---|---|---|---|---|"]
    summary = {}
    for fault, (target, desc) in FAULTS.items():
        fr = [r for r in rows if r["fault"] == fault]
        oc = Counter(r["outcome"] for r in fr)
        blocked = sum(r.get("letter") == "blocked" for r in fr)
        by = ", ".join(f"{k} ({v})" for k, v in Counter(r["caught_by"] for r in fr).most_common())
        summary[fault] = {"runs": len(fr), **oc, "letters_blocked": blocked if target == "letter" else None}
        md.append(f"| {fault} | {desc} | {len(fr)} | {oc['correct']} | {oc['held']} | {oc['wrong']} | "
                  f"{f'{blocked}/{len(fr)}' if target == 'letter' else '—'} | {by} |")
    wrong = [r for r in rows if r["outcome"] == "wrong"]
    sent = [r for r in rows if r.get("letter") == "SENT"]
    md += ["", f"**{len(rows)} faulty runs: {len(wrong)} wrong decision(s) and {len(sent)} corrupted letter(s) got out.**"]
    if wrong or sent:
        md += ["", "| Fault | Case | Expected | Got | Letter |", "|---|---|---|---|---|"]
        md += [f"| {r['fault']} | {r['claim_id']} | {r['expected']} | {r['decision']} | {r.get('letter', '')} |" for r in wrong + sent]

    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    (out / "fault_injection.json").write_text(json.dumps({"base": base.name, "summary": summary, "rows": rows}, indent=2,
                                                         ensure_ascii=False), encoding="utf-8")
    (out / "fault_injection.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    print("\nSaved results/fault_injection.md and .json")
    if args.strict and (wrong or sent):
        sys.exit(1)


if __name__ == "__main__":
    main()
