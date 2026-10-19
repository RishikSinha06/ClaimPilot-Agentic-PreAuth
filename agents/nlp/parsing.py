"""Turning raw LLM text into trusted facts (stages P11-P13).

P11 Parse     pull the JSON object out of the raw model text
P12 Validate  check it against the schema (types, enums, required keys); on
              failure the Intake agent sends a repair prompt (max 2 retries)
P13 Ground    every chronic condition must be supported by the sentence the
              model cited, and must not be negated there. Unsupported claims
              are dropped (hallucination guard). Durations are numbers, so the
              one stated in the text (P8) replaces the model's. Then compare with
              the rule-based extractor so the Critic can see any disagreement.
"""
from __future__ import annotations

import json
import re

from .preprocess import CONDITIONS


def extract_json(text: str):
    """Return the first JSON object found in text (handles ``` fences and chatter)."""
    if not text:
        raise ValueError("empty response")
    t = re.sub(r"```(?:json)?", "", text).strip()
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        pass
    start = t.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(t)):
            if t[i] == "{":
                depth += 1
            elif t[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(t[start:i + 1])
                    except json.JSONDecodeError:
                        break
        start = t.find("{", start + 1)
    raise ValueError("no JSON object found")


_TYPES = {"object": dict, "array": list, "string": str, "integer": int, "number": (int, float), "boolean": bool, "null": type(None)}


def validate(obj, schema: dict, path: str = "$") -> list[str]:
    """Minimal JSON-schema validator (type, enum, required, properties, items)."""
    errors = []
    types = schema.get("type")
    if types:
        types = types if isinstance(types, list) else [types]
        ok = False
        for t in types:
            py = _TYPES[t]
            if isinstance(obj, py) and not (t in ("integer", "number") and isinstance(obj, bool)):
                ok = True
        if not ok:
            return [f"{path}: expected {' or '.join(types)}, got {type(obj).__name__}"]
    if "enum" in schema and obj not in schema["enum"]:
        errors.append(f"{path}: '{obj}' is not an allowed value")
    if isinstance(obj, dict):
        for key in schema.get("required", []):
            if key not in obj:
                errors.append(f"{path}: missing required key '{key}'")
        for key, sub in schema.get("properties", {}).items():
            if key in obj:
                errors += validate(obj[key], sub, f"{path}.{key}")
    if isinstance(obj, list) and "items" in schema:
        for i, item in enumerate(obj):
            errors += validate(item, schema["items"], f"{path}[{i}]")
    return errors


def canonicalise(obj: dict) -> dict:
    """Light clean-up before validation: lower-case enums, map condition synonyms."""
    if not isinstance(obj, dict):
        return obj
    out = dict(obj)
    if isinstance(out.get("procedure_key"), str):
        out["procedure_key"] = out["procedure_key"].strip().lower()
    conds = []
    for c in out.get("chronic_conditions") or []:
        if not isinstance(c, dict):
            conds.append(c)
            continue
        c = dict(c)
        name = str(c.get("condition", "")).strip().lower()
        for canon, syns in CONDITIONS.items():
            if name == canon or name in syns:
                name = canon
        c["condition"] = name
        if isinstance(c.get("evidence_sentence"), str) and c["evidence_sentence"].strip().upper().lstrip("S").isdigit():
            c["evidence_sentence"] = int(c["evidence_sentence"].strip().upper().lstrip("S"))
        conds.append(c)
    out["chronic_conditions"] = conds
    return out


def parse_and_validate(text: str, schema: dict) -> tuple[dict | None, list[str]]:
    try:
        obj = canonicalise(extract_json(text))
    except ValueError as e:
        return None, [f"parse error: {e}"]
    errs = validate(obj, schema)
    return (obj if not errs else None), errs


def ground(facts: dict, prep) -> tuple[dict, list[dict]]:
    """Check each LLM-claimed condition against the preprocessed, negation-tagged entities."""
    report = []
    kept = []
    live = [e for e in prep.entities if e["type"] == "CONDITION" and not e["negated"]]
    negated = [e for e in prep.entities if e["type"] == "CONDITION" and e["negated"]]
    declared_sents = {e["sentence"] for e in prep.entities if e["type"] == "FLAG" and e["canonical"] == "declared" and not e["negated"]}
    for c in facts.get("chronic_conditions", []):
        cited = c.get("evidence_sentence")
        at_cited = [e for e in live if e["canonical"] == c["condition"] and e["sentence"] == cited]
        anywhere = [e for e in live if e["canonical"] == c["condition"]]
        if at_cited:
            status, ent = "grounded", at_cited[0]
        elif anywhere:
            status, ent = "grounded (evidence corrected)", anywhere[0]
        elif any(e["canonical"] == c["condition"] for e in negated):
            status, ent = "rejected: negated in text", None
        else:
            status, ent = "rejected: not found in text", None
        if ent:
            item = dict(c)
            item["evidence_sentence"] = ent["sentence"]
            # Numbers come from deterministic tools: a duration stated in the text (P8) wins over the model's.
            text_years = next((e["duration_years"] for e in [ent] + anywhere if e["duration_years"] is not None), None)
            llm_years = c.get("duration_years")
            if text_years is not None:
                if llm_years is None or abs(llm_years - text_years) > 0.5:
                    status += f"; duration corrected to {text_years:g} y (LLM said {'none' if llm_years is None else f'{llm_years:g} y'})"
                item["duration_years"] = text_years
            item["stated_as_declared"] = ent["sentence"] in declared_sents
            kept.append(item)
        report.append({"condition": c["condition"], "cited": cited, "status": status})
    # procedure grounding
    procs = {e["canonical"] for e in prep.entities if e["type"] == "PROCEDURE" and not e["negated"]}
    pk = facts["procedure_key"]
    proc_status = "grounded" if (pk in procs or (pk == "medical management" and not procs)) else (
        "plausible (no surgery terms found)" if pk == "medical management" else "not found in text")
    report.append({"condition": f"procedure: {pk}", "cited": None, "status": proc_status})
    out = dict(facts)
    out["chronic_conditions"] = kept
    return out, report


def compare(llm: dict, rules: dict) -> list[dict]:
    rows = []
    for key in ["procedure_key", "cosmetic_intent", "functional_need", "accident"]:
        rows.append({"field": key, "llm": llm.get(key), "rules": rules.get(key), "agree": llm.get(key) == rules.get(key)})
    def conds(facts):
        return sorted(c["condition"] + (f" ({c['duration_years']:g} y)" if c.get("duration_years") is not None else "")
                      for c in facts.get("chronic_conditions", []))
    lc, rc = conds(llm), conds(rules)
    rows.append({"field": "chronic_conditions", "llm": ", ".join(lc) or "none", "rules": ", ".join(rc) or "none", "agree": lc == rc})
    return rows
