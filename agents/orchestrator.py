"""Orchestrator.

* Executes the Planner's task list in order, evaluating each task's run-time
  condition (`when`): always | accident | eligible | escalated.
* Agents never call each other. They read and write one shared state object
  (the "blackboard" `ctx`); the Orchestrator routes between them.
* Runs the Critic -> Executor revision loop (max N revisions).
* Records an audit trace: every agent step, every tool / NLP / LLM / guardrail
  call made inside it, and every message between participants.
* Human-in-the-loop: escalated cases wait for a medical officer; queried
  cases can be re-submitted by the hospital with the missing documents.
"""
from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import time
from datetime import datetime
from pathlib import Path

from . import trace
from .critic import CriticAgent
from .decision import DecisionAgent
from .executor import CoverageExecutorAgent
from .intake import IntakeAgent
from .llm import MockLLM
from .planner import PlannerAgent
from .research import PolicyResearchAgent


class HumanReviewer:
    name = "Medical Officer (human)"
    short = "Medical officer"
    icon = "🩺"


class Orchestrator:
    name = "Orchestrator"

    def __init__(self, llm=None, tariff_tolerance_pct=25, auto_approve_limit=300000, max_revisions=2):
        self.llm = llm or MockLLM()
        self.planner = PlannerAgent()
        self.intake = IntakeAgent(self.llm)
        self.research = PolicyResearchAgent()
        self.executor = CoverageExecutorAgent()
        self.critic = CriticAgent(tariff_tolerance_pct, auto_approve_limit)
        self.decider = DecisionAgent(self.llm)
        self.max_revisions = max_revisions
        self.registry = {
            "check_documents": (self.intake, self.intake.check_documents),
            "preprocess_text": (self.intake, self.intake.preprocess),
            "extract_facts": (self.intake, self.intake.extract),
            "verify_accident_docs": (self.intake, self.intake.verify_accident_docs),
            "research_policy": (self.research, self.research.run),
            "check_eligibility": (self.executor, self.executor.eligibility),
            "calculate_payable": (self.executor, self.executor.calculate),
            "validate": (self.critic, self.critic.run),
            "decide": (self.decider, self.decider.decide),
            "draft_letter": (self.decider, self.decider.draft_letter),
        }

    # ------------------------------------------------------------ helpers
    @staticmethod
    def _condition(ctx: dict, when: str) -> bool:
        return {
            "always": lambda: True,
            "accident": lambda: bool(ctx.get("extracted", {}).get("accident")),
            "eligible": lambda: bool(ctx.get("eligibility", {}).get("eligible")),
            "escalated": lambda: ctx.get("decision", {}).get("decision") == "ESCALATE",
        }[when]()

    @staticmethod
    def _msg(ctx, frm, to, mtype, content):
        ctx["messages"].append({"from": frm, "to": to, "type": mtype, "content": content})

    def _step(self, ctx, agent, task, fn, on_event):
        calls, token = trace.begin()
        t0 = time.perf_counter()
        try:
            out = fn()
        finally:
            trace.end(token)
        ms = (time.perf_counter() - t0) * 1000
        entry = {"n": len(ctx["trace"]) + 1, "agent": agent.name, "short": agent.short, "icon": agent.icon,
                 "task": task, "summary": out["summary"], "details": out["details"], "ms": round(ms, 2), "calls": calls}
        ctx["trace"].append(entry)
        self._msg(ctx, "Orchestrator", agent.short, "task", task)
        self._msg(ctx, agent.short, "Orchestrator", "result", out["summary"])
        if on_event:
            on_event(entry)
        return out

    # ------------------------------------------------------------ main run
    def run(self, claim: dict, on_event=None) -> dict:
        ctx = {"claim": copy.deepcopy(claim), "trace": [], "messages": [], "revisions": 0, "skipped": [],
               "status": {}, "llm": self.llm.name, "llm_simulated": self.llm.is_mock,
               "started_at": datetime.now().isoformat(timespec="seconds")}
        self._msg(ctx, "Streamlit UI", "Orchestrator", "request", f"process {claim['claim_id']}")
        self._step(ctx, self.planner, "plan", lambda: self.planner.run(ctx), on_event)

        for task in ctx["plan"]["tasks"]:
            tid = task["id"]
            if tid == "human_review":
                if self._condition(ctx, "escalated"):
                    ctx["status"][tid] = "awaiting human"
                    self._msg(ctx, "Orchestrator", "Medical officer", "escalate", "case held for review")
                else:
                    ctx["status"][tid] = "not needed"
                continue
            if not self._condition(ctx, task["when"]):
                ctx["skipped"].append(tid)
                ctx["status"][tid] = "skipped"
                continue
            agent, fn = self.registry[tid]
            if tid == "validate":
                while True:
                    self._step(ctx, agent, "validate", lambda: fn(ctx), on_event)
                    if ctx["critic"]["verdict"] != "REVISE" or ctx["revisions"] >= self.max_revisions:
                        break
                    ctx["revisions"] += 1
                    self._msg(ctx, "Critic", "Coverage Executor", "revise", "cap payable at network tariff ceiling")
                    self._step(ctx, self.executor, f"calculate_payable (revision {ctx['revisions']})",
                               lambda: self.executor.calculate(ctx, revision=ctx["critic"]["instruction"]), on_event)
            else:
                self._step(ctx, agent, tid, lambda fn=fn: fn(ctx), on_event)
            ctx["status"][tid] = "done"

        self._msg(ctx, "Orchestrator", "Streamlit UI", "response", ctx["decision"]["decision"])
        ctx["elapsed_ms"] = round(sum(t["ms"] for t in ctx["trace"]), 2)  # agent compute only
        return ctx

    # ------------------------------------------------------------ human in the loop
    def human_review(self, ctx: dict, action: str, remarks: str = "", reviewer: str = "Dr. Medical Officer", on_event=None) -> dict:
        if ctx["decision"]["decision"] != "ESCALATE":
            raise ValueError(f"Only escalated cases go to the medical officer (this one is {ctx['decision']['decision']})")
        if action not in ("approve", "deny", "query"):
            raise ValueError(f"Unknown review action: {action}")
        if action == "approve" and ctx.get("calculation") is None and ctx.get("policy"):
            # Escalated before any amount was computed: the denial rested on disputed LLM facts.
            # Approving means the officer accepts the claim, so price it on the rule-based facts.
            self._step(ctx, self.executor, "calculate_payable (for medical officer)",
                       lambda: self.executor.calculate(ctx, facts=ctx.get("extracted_rules")), on_event)
        self._step(ctx, HumanReviewer(), f"human_review ({action})",
                   lambda: self.decider.apply_review(ctx, action, remarks, reviewer), on_event)
        self._step(ctx, self.decider, "draft_letter", lambda: self.decider.draft_letter(ctx), on_event)
        ctx["status"]["human_review"] = "done"
        self._msg(ctx, "Orchestrator", "Streamlit UI", "response", ctx["decision"]["decision"])
        return ctx

    @staticmethod
    def resubmit(claim: dict) -> dict:
        """Hospital answers a query: add the missing documents and any new notes."""
        new = copy.deepcopy(claim)
        qr = claim.get("query_response", {})
        for d in qr.get("documents_added", ["investigation_reports"]):
            if d not in new["documents"]:
                new["documents"].append(d)
        if qr.get("notes_append"):
            new["doctor_notes"] = new["doctor_notes"].rstrip() + " " + qr["notes_append"]
        new["resubmission_of"] = claim["claim_id"]
        new.pop("query_response", None)
        return new


def audit_record(ctx: dict) -> dict:
    """JSON-serialisable record of one run (for download / audit log)."""
    def clean(o):
        if dataclasses.is_dataclass(o):
            return dataclasses.asdict(o)
        raise TypeError(type(o).__name__)
    keep = ["claim", "started_at", "llm", "llm_simulated", "plan", "status", "missing_docs", "prep", "llm_extraction",
            "extracted", "extracted_rules", "policy", "benchmark", "eligibility", "calculation", "critic_history",
            "decision", "letter_info", "messages", "trace", "revisions", "elapsed_ms"]
    rec = {k: ctx.get(k) for k in keep}
    rec["clauses"] = [c["id"] for c in ctx.get("clauses", [])]
    return json.loads(json.dumps(rec, default=clean, ensure_ascii=False))


GENESIS = "0" * 64


def _entry_hash(body: dict) -> str:
    return hashlib.sha256(json.dumps(body, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def append_audit_log(ctx: dict, path: str | Path) -> None:
    """Append one line to a hash-chained JSONL log: each entry stores the previous entry's
    hash, so editing or deleting an earlier line breaks the chain (see verify_audit_log)."""
    rec = audit_record(ctx)
    line = {"time": datetime.now().isoformat(timespec="seconds"), "claim_id": rec["claim"]["claim_id"],
            "decision": rec["decision"]["decision"], "amount": rec["decision"]["amount"],
            "decided_by": rec["decision"].get("decided_by"), "llm": rec["llm"]}
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    prev = GENESIS
    lines = p.read_text(encoding="utf-8").splitlines() if p.exists() else []
    if lines:
        try:
            prev = json.loads(lines[-1]).get("hash", GENESIS)
        except ValueError:
            prev = GENESIS
    line["prev_hash"] = prev
    line["hash"] = _entry_hash(line)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(line, ensure_ascii=False) + "\n")


def verify_audit_log(path: str | Path) -> dict:
    """Re-walk the hash chain. Lines written before hashing existed are skipped (they restart the chain)."""
    prev, verified = GENESIS, 0
    p = Path(path)
    for i, raw in enumerate(p.read_text(encoding="utf-8").splitlines() if p.exists() else [], start=1):
        if not raw.strip():
            continue
        try:
            entry = json.loads(raw)
        except ValueError:
            return {"ok": False, "verified": verified, "bad_line": i}
        if "hash" not in entry:
            prev = GENESIS
            continue
        body = {k: v for k, v in entry.items() if k != "hash"}
        if entry.get("prev_hash") != prev or _entry_hash(body) != entry["hash"]:
            return {"ok": False, "verified": verified, "bad_line": i}
        prev, verified = entry["hash"], verified + 1
    return {"ok": True, "verified": verified, "bad_line": None}
