"""Document Intake Agent.

check_documents       mandatory documents present?
preprocess_text       stages P1-P9 (see nlp/preprocess.py)
extract_facts         stages P10-P13: prompt -> LLM -> parse -> validate (+repair retries)
                      -> ground in text -> cross-check against the rule-based extractor
verify_accident_docs  conditional task: accident cases need an MLC / FIR copy
"""
from __future__ import annotations

from . import guardrails
from .nlp.parsing import compare, ground, parse_and_validate
from .nlp.preprocess import facts_from_entities, preprocess_claim
from .nlp.prompts import EXTRACTION_SCHEMA, build_extraction_prompt, build_repair_prompt

DOC_LABELS = {
    "preauth_form": "Pre-authorisation form",
    "id_card": "Insured ID card",
    "doctor_notes": "Treating doctor's notes",
    "investigation_reports": "Investigation reports",
    "mlc_fir": "MLC / FIR copy",
}


class IntakeAgent:
    name = "Document Intake Agent"
    short = "Doc Intake"
    icon = "📄"

    def __init__(self, llm, max_retries: int = 2):
        self.llm = llm
        self.max_retries = max_retries

    # ------------------------------------------------------------ T1
    def check_documents(self, ctx: dict) -> dict:
        claim = ctx["claim"]
        required = ctx["plan"]["required_docs"]
        missing = [d for d in required if d not in claim["documents"]]
        ctx["missing_docs"] = missing
        summary = ("All mandatory documents present." if not missing else
                   f"Missing: {', '.join(DOC_LABELS[d] for d in missing)}.")
        return {"summary": summary, "details": {"required": [DOC_LABELS[d] for d in required],
                                                "received": [DOC_LABELS.get(d, d) for d in claim["documents"]],
                                                "missing": [DOC_LABELS[d] for d in missing]}}

    # ------------------------------------------------------------ T2
    def preprocess(self, ctx: dict) -> dict:
        prep = preprocess_claim(ctx["claim"])
        ctx["prep"] = prep
        neg = sum(e["negated"] for e in prep.entities)
        summary = (f"{sum(c for *_, c in prep.expansions)} abbreviations expanded, {len(prep.pii_map)} identifier(s) masked, "
                   f"{len(prep.sentences)} sentences, {len(prep.entities)} entities ({neg} negated), ~{prep.tokens_est} tokens.")
        return {"summary": summary, "details": {"stages": prep.stage_log, "pii_masked": list(prep.pii_map)}}

    # ------------------------------------------------------------ T3
    def extract(self, ctx: dict) -> dict:
        claim, prep = ctx["claim"], ctx["prep"]
        rules = facts_from_entities(prep.sentences, prep.entities)
        ctx["extracted_rules"] = rules

        system, prompt = build_extraction_prompt(prep, claim)
        info = {"system": system, "prompt": prompt, "attempts": [], "grounding": [], "compare": [],
                "backend": self.llm.name, "simulated": self.llm.is_mock, "fallback": None}
        pii_ok, leaks = guardrails.pii_gate(system + prompt, prep.pii_map, claim)
        info["pii_gate"] = "pass" if pii_ok else f"blocked ({', '.join(leaks)})"

        facts, method = None, "rules"
        if not pii_ok:
            info["fallback"] = "PII gate blocked the prompt; LLM not called"
        else:
            current = prompt
            for attempt in range(1, self.max_retries + 2):
                try:
                    res = self.llm.generate(current, system=system, schema=EXTRACTION_SCHEMA, purpose="extract facts")
                except Exception as e:  # noqa: BLE001 - network / timeout / server errors
                    info["attempts"].append({"attempt": attempt, "error": f"{type(e).__name__}: {e}"[:200]})
                    info["fallback"] = "LLM call failed; rule-based facts used"
                    break
                obj, errors = parse_and_validate(res.text, EXTRACTION_SCHEMA)
                info["attempts"].append({"attempt": attempt, "raw": res.text, "errors": errors,
                                         "prompt_tokens": res.prompt_tokens, "output_tokens": res.output_tokens,
                                         "latency_ms": res.latency_ms, "cached": res.cached, "format": res.format_used})
                if not errors:
                    facts, grounding = ground(obj, prep)
                    info["grounding"] = grounding
                    method = self.llm.name
                    break
                current = build_repair_prompt(prompt, res.text, errors)
            else:
                info["fallback"] = f"Output still invalid after {self.max_retries} repair attempt(s); rule-based facts used"

        if facts is None:
            facts = rules
        info["compare"] = compare(facts, rules)
        info["method"] = method
        ctx["extracted"] = facts
        ctx["llm_extraction"] = info

        conds = ", ".join(f"{c['condition']}" + (f" ({c['duration_years']:g} y)" if c.get("duration_years") is not None else "")
                          for c in facts["chronic_conditions"]) or "none"
        disagree = [r["field"] for r in info["compare"] if not r["agree"]]
        summary = (f"Facts via {method}: procedure = {facts['procedure_key']}; chronic = {conds}; accident = {facts['accident']}. "
                   f"{len(info['attempts'])} LLM attempt(s). "
                   + ("LLM and rules agree." if not disagree else f"Disagree on: {', '.join(disagree)}.")
                   + (f" {info['fallback']}." if info["fallback"] else ""))
        return {"summary": summary, "details": {"method": method, "facts": facts, "pii_gate": info["pii_gate"],
                                                "attempts": len(info["attempts"]), "grounding": info["grounding"],
                                                "compare": info["compare"]}}

    # ------------------------------------------------------------ T4 (conditional)
    def verify_accident_docs(self, ctx: dict) -> dict:
        claim = ctx["claim"]
        if "mlc_fir" not in claim["documents"] and "mlc_fir" not in ctx["missing_docs"]:
            ctx["missing_docs"].append("mlc_fir")
        ok = "mlc_fir" in claim["documents"]
        return {"summary": "Accident case: MLC / FIR copy " + ("present." if ok else "MISSING."),
                "details": {"mlc_fir_present": ok}}
