"""Benchmark a local open-source LLM (via Ollama) on the 12 labelled cases.

Usage (Ollama must be running and the model pulled):
    ollama pull llama3.2
    python scripts/benchmark_llm.py                       # default: llama3.2
    python scripts/benchmark_llm.py --model qwen2.5:7b    # any other Ollama model
    python scripts/benchmark_llm.py --mock                # no Ollama: simulated LLM

Reports, for each model: extraction accuracy per field vs the hand-labelled
gold facts, decision accuracy, safety outcomes (correct / held for review /
wrong decision sent), JSON validity on the first attempt, repair retries, rule
fallbacks, grounding rejections and corrections, LLM letters accepted by the
faithfulness check, token counts and latency.
Writes results/benchmark_<model>.json (with the raw LLM output per case) and .md.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agents import Orchestrator, make_llm  # noqa: E402
from agents.evaluation import FIELDS, score  # noqa: E402
from agents.llm import _CACHE  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama3.2")
    ap.add_argument("--host", default="http://localhost:11434")
    ap.add_argument("--timeout", type=int, default=600, help="seconds per LLM call (CPU-only machines need a lot)")
    ap.add_argument("--mock", action="store_true", help="use the simulated LLM instead of Ollama")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    llm, msg = make_llm("mock" if args.mock else "ollama", args.model, args.host, timeout=args.timeout)
    print(f"LLM backend: {llm.name}  ({msg})")
    if not args.mock and llm.is_mock:
        print("Ollama/model not available - aborting. Start Ollama and pull the model first.")
        sys.exit(1)
    _CACHE.clear()

    claims = json.loads((ROOT / "data" / "claims.json").read_text(encoding="utf-8"))
    orch = Orchestrator(llm=llm)
    rows, t0 = [], time.perf_counter()
    for c in claims:
        ctx = orch.run(c)
        info = ctx["llm_extraction"]
        att = info["attempts"]
        s = score(c, ctx)
        rows.append({
            "claim_id": c["claim_id"],
            "expected": c["expected_decision"], "decision": s["decision"], "decision_ok": s["decision_ok"],
            "amount_ok": s["amount_ok"], "outcome": s["outcome"],
            **{f"{f}_ok": ok for f, ok in s["fields"].items()},
            "valid_first_try": bool(att) and not att[0].get("errors") and "error" not in att[0],
            "attempts": len(att), "fallback": bool(info["fallback"]), "llm_used": info["method"] != "rules",
            "grounding_rejections": sum(g["status"].startswith("rejected") for g in info["grounding"]),
            "grounding_corrections": sum("corrected" in g["status"] for g in info["grounding"]),
            "extractor_disagreements": [r["field"] for r in info["compare"] if not r["agree"]],
            "prompt_tokens": sum(a.get("prompt_tokens") or 0 for a in att),
            "output_tokens": sum(a.get("output_tokens") or 0 for a in att),
            "llm_ms": sum(a.get("latency_ms") or 0 for a in att),
            "letter_method": ctx["decision"]["letter_method"],
            "letter_problems": ctx["letter_info"]["problems"],
            "llm_raw_extraction": att[-1].get("raw") if att else None,
            "critic_verdict": ctx["critic"]["verdict"],
        })
        r = rows[-1]
        print(f"{c['claim_id']}  {r['decision']:9} {r['outcome']:8} fields {sum(r[f + '_ok'] for f in FIELDS)}/5  "
              f"attempts {r['attempts']}  {r['llm_ms']:.0f} ms  letter: {'LLM' if not r['letter_method'].startswith('template') else 'template'}")

    n = len(rows)
    summary = {
        "model": llm.name, "cases": n, "wall_time_s": round(time.perf_counter() - t0, 1),
        "decision_accuracy": sum(r["decision_ok"] for r in rows) / n,
        "wrong_decisions_sent": sum(r["outcome"] == "wrong" for r in rows),
        "held_for_review": sum(r["outcome"] == "held" for r in rows),
        **{f"{f}_accuracy": sum(r[f + "_ok"] for r in rows) / n for f in FIELDS},
        "json_valid_first_try": sum(r["valid_first_try"] for r in rows) / n,
        "repair_retries": sum(max(0, r["attempts"] - 1) for r in rows),
        "rule_fallbacks": sum(r["fallback"] for r in rows),
        "grounding_rejections": sum(r["grounding_rejections"] for r in rows),
        "grounding_corrections": sum(r["grounding_corrections"] for r in rows),
        "cases_with_extractor_disagreement": sum(bool(r["extractor_disagreements"]) for r in rows),
        "avg_prompt_tokens": round(statistics.mean(r["prompt_tokens"] for r in rows)),
        "avg_output_tokens": round(statistics.mean(r["output_tokens"] for r in rows)),
        "avg_llm_latency_ms": round(statistics.mean(r["llm_ms"] for r in rows)),
        "letters_accepted_from_llm": sum(not r["letter_method"].startswith("template") for r in rows),
    }
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    tag = "mock" if llm.is_mock else args.model.replace(":", "_")
    (out / f"benchmark_{tag}.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=2, ensure_ascii=False),
                                               encoding="utf-8")
    md = ["| Metric | Value |", "|---|---|"] + [f"| {k} | {v:.0%} |" if isinstance(v, float) and v <= 1 and k.endswith(("accuracy", "try"))
                                              else f"| {k} | {v} |" for k, v in summary.items()]
    (out / f"benchmark_{tag}.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n" + "\n".join(md))
    print(f"\nSaved results/benchmark_{tag}.json and .md")


if __name__ == "__main__":
    main()
