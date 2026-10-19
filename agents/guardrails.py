"""Guardrails around every LLM call.

1. PII gate        - before a prompt leaves the process, check it two ways:
                     (a) no identifier that P4 masked is still in it, and
                     (b) an independent scan that does not trust the masking: the
                         claim's own identifiers (patient name and each part of it,
                         hospital, policy and claim numbers), any number with 10 or
                         more digits, e-mail addresses and titled names (Dr / Mrs ...).
2. De-identify     - letters are masked like the notes; real values are put back
   / re-identify     only AFTER the LLM has answered.
3. Letter check    - the LLM-polished letter must keep exactly the amounts, clause
                     numbers and placeholders of the template, every reason word for
                     word, and must not swap approval wording for denial wording (or
                     the reverse); otherwise the template letter is used.
"""
from __future__ import annotations

import re

from .nlp.preprocess import TITLED_NAME, mask_identifiers
from .trace import record

_EMAIL = re.compile(r"\S+@\S+\.\w+")
_DIGIT_RUN = re.compile(r"\d[\d\s-]*\d")


def claim_identifiers(claim: dict) -> list[tuple[str, str, bool]]:
    """(label, value, case_sensitive) for every identifier the claim itself carries."""
    out = []
    name = claim.get("patient", "")
    if name:
        out.append(("patient name", name, False))
        out += [("patient name", p, True) for p in name.split() if len(p) >= 3]
    hosp = claim.get("hospital", "")
    if hosp:
        out.append(("hospital name", hosp, False))
        short = hosp.split(",")[0].strip()
        if short and short != hosp:
            out.append(("hospital name", short, False))
    for key, label in (("policy_no", "policy number"), ("claim_id", "claim ID")):
        if claim.get(key):
            out.append((label, claim[key], False))
    return out


def residual_pii(text: str, claim: dict | None = None) -> list[str]:
    """Independent scan of outgoing text. It does not rely on what the masking step found."""
    found = []
    for label, value, case_sensitive in claim_identifiers(claim or {}):
        if re.search(rf"(?<!\w){re.escape(value)}(?!\w)", text, 0 if case_sensitive else re.IGNORECASE):
            found.append(label)
    if any(len(re.sub(r"\D", "", m.group(0))) >= 10 for m in _DIGIT_RUN.finditer(text)):
        found.append("number with 10+ digits (phone / ID)")
    if _EMAIL.search(text):
        found.append("e-mail address")
    if TITLED_NAME.search(text):
        found.append("titled personal name")
    return list(dict.fromkeys(found))


def pii_gate(text: str, pii_map: dict, claim: dict | None = None) -> tuple[bool, list[str]]:
    low = text.lower()
    leaks = [ph for ph, original in pii_map.items() if original and original.lower() in low]
    leaks += [f"unmasked {x}" for x in residual_pii(text, claim)]
    ok = not leaks
    record("guardrail", "PII gate", "pass: no identifiers in prompt" if ok else f"BLOCKED: {', '.join(leaks)} found")
    return ok, leaks


def deidentify_text(text: str, claim: dict) -> tuple[str, dict]:
    mapping = {}
    out = text
    hosp = claim.get("hospital", "")
    short = hosp.split(",")[0].strip()
    for value, ph in [(hosp, "[HOSPITAL]"), (short if short != hosp else "", "[HOSPITAL_SHORT]"),
                      (claim.get("patient", ""), "[PATIENT]"), (claim.get("policy_no", ""), "[POLICY_NO]"),
                      (claim.get("claim_id", ""), "[CLAIM_ID]")]:
        if value and value in out:
            out = out.replace(value, ph)
            mapping[ph] = value
    # A part of the patient's name on its own (e.g. a surname in the officer's remarks) gets its
    # own placeholder, so re-identification restores exactly what was written.
    for i, part in enumerate((p for p in claim.get("patient", "").split() if len(p) >= 3), start=1):
        pat = rf"(?<!\w){re.escape(part)}(?!\w)"
        if re.search(pat, out):
            out = re.sub(pat, f"[PATIENT_NAME_{i}]", out)
            mapping[f"[PATIENT_NAME_{i}]"] = part
    out = mask_identifiers(out, mapping, {})  # phones, e-mails, IDs, "Dr. A. Rao" in remarks
    return out, mapping


def reidentify(text: str, mapping: dict) -> str:
    for ph, value in mapping.items():
        text = text.replace(ph, value)
    return text


_AMOUNT = re.compile(r"₹\s?\d+(?:,\d+)*")
_CLAUSE = re.compile(r"clause\s+\d+\.\d+", re.I)
_PLACEHOLDER = re.compile(r"\[[A-Z_0-9]+\]")
# Wording that commits to an outcome. A polished letter may not introduce the opposite outcome's wording.
_APPROVAL = ["pleased to authorise", "pleased to approve", "we authorise", "we approve", "has been approved",
             "is approved", "are approved", "has been authorised", "is authorised", "authorisation is granted",
             "approval is granted"]
_REFUSAL = ["cannot be authorised", "cannot authorise", "not be authorised", "unable to authorise", "is denied",
            "been denied", "is rejected", "been rejected", "not admissible", "not payable"]


def _facts(text: str) -> dict[str, set]:
    return {"amounts": {a.replace(" ", "") for a in _AMOUNT.findall(text)},
            "clauses": {c.lower() for c in _CLAUSE.findall(text)},
            "placeholders": set(_PLACEHOLDER.findall(text))}


def _norm(s: str) -> str:
    s = s.lower().replace("authoriz", "authoris")
    s = re.sub(r"[‐-―−]", "-", s)
    s = s.replace("‘", "'").replace("’", "'").replace("“", '"').replace("”", '"')
    s = re.sub(r"₹\s+", "₹", s)
    return re.sub(r"\s+", " ", s).strip()


def letter_faithful(template: str, generated: str, must_keep: list[str] | tuple = (),
                    decision: str | None = None) -> tuple[bool, list[str]]:
    a, b = _facts(template), _facts(generated)
    problems = []
    for k in a:
        if b[k] - a[k]:
            problems.append(f"new {k}: {', '.join(sorted(b[k] - a[k]))}")
        if a[k] - b[k]:
            problems.append(f"missing {k}: {', '.join(sorted(a[k] - b[k]))}")
    if len(generated.strip()) < 0.4 * len(template.strip()):
        problems.append("letter too short")
    t, g = _norm(template), _norm(generated)
    for reason in must_keep:
        if _norm(reason).rstrip(".") not in g:
            problems.append(f"reason changed or missing: '{reason[:60]}{'…' if len(reason) > 60 else ''}'")
    if decision:
        opposite = _REFUSAL if decision in ("APPROVE", "PARTIAL") else _APPROVAL
        flipped = [p for p in opposite if p in g and p not in t]
        if flipped:
            problems.append(f"wording contradicts the {decision} decision: {', '.join(flipped)}")
    ok = not problems
    record("guardrail", "Letter faithfulness", "pass" if ok else "; ".join(problems))
    return ok, problems
