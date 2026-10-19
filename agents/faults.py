"""Fault injection: wrap any LLM backend and corrupt its answers on purpose.

Shows that schema validation, grounding, the rule cross-check, the Critic and the
letter faithfulness check stop a wrong LLM answer from becoming a wrong decision
or an unfaithful letter. Used by scripts/fault_injection.py, the tests and the
"Inject an LLM error" option in the app.
"""
from __future__ import annotations

import copy
import dataclasses
import json
import re

from .nlp.preprocess import CONDITIONS, analyze_numbered_text
from .nlp.prompts import TASK_MARKER

FAULTS = {  # name -> (what it corrupts, description)
    "malformed_json": ("extraction", "Returns broken JSON on every attempt"),
    "hallucinated_condition": ("extraction", "Adds a chronic condition the notes never mention"),
    "negated_condition": ("extraction", "Reports a condition the notes explicitly negate"),
    "wrong_duration": ("extraction", "Says every chronic condition started 6 months ago"),
    "dropped_conditions": ("extraction", "Leaves out every chronic condition"),
    "wrong_procedure": ("extraction", "Swaps the procedure (dental surgery <-> appendectomy)"),
    "flipped_accident": ("extraction", "Flips the accident flag"),
    "flipped_cosmetic": ("extraction", "Flips the cosmetic-intent flag"),
    "flipped_functional": ("extraction", "Flips the functional-need flag"),
    "letter_changed_amount": ("letter", "Changes (or adds) a ₹ amount in the letter"),
    "letter_dropped_reason": ("letter", "Drops one reason from the letter"),
    "letter_flipped_decision": ("letter", "Rewrites the letter with the opposite decision"),
}

_AMOUNT = re.compile(r"₹\s?\d+(?:,\d+)*")


def corrupt_extraction(fault: str, text: str, prompt: str) -> str:
    if fault == "malformed_json":
        return 'Sure! Here are the facts: {"procedure_key": "'
    try:
        orig = json.loads(text)
    except ValueError:
        return text  # the base answer is already broken; nothing to corrupt
    obj = copy.deepcopy(orig)
    task = prompt.split(TASK_MARKER, 1)[1]
    conds = obj.setdefault("chronic_conditions", [])
    if fault == "hallucinated_condition":
        low = task.lower()
        absent = next((k for k, syns in CONDITIONS.items() if k not in low and not any(s in low for s in syns)), None)
        if absent:
            conds.append({"condition": absent, "duration_years": 10, "evidence_sentence": 1})
    elif fault == "negated_condition":
        m = re.search(r"Request date:\s*(\d{4})", task)
        _, ents = analyze_numbered_text(task, int(m.group(1)) if m else 2026)
        present = {c["condition"] for c in conds}
        neg = next((e for e in ents if e["type"] == "CONDITION" and e["negated"] and e["canonical"] not in present), None)
        if neg:
            conds.append({"condition": neg["canonical"], "duration_years": 5, "evidence_sentence": neg["sentence"]})
    elif fault == "wrong_duration":
        for c in conds:
            c["duration_years"] = 0.5
    elif fault == "dropped_conditions":
        obj["chronic_conditions"] = []
    elif fault == "wrong_procedure":
        obj["procedure_key"] = "appendectomy" if obj.get("procedure_key") == "dental surgery" else "dental surgery"
    elif fault.startswith("flipped_"):
        key = {"flipped_accident": "accident", "flipped_cosmetic": "cosmetic_intent",
               "flipped_functional": "functional_need"}[fault]
        obj[key] = not obj.get(key)
    return json.dumps(obj) if obj != orig else text


def corrupt_letter(fault: str, text: str) -> str:
    if fault == "letter_changed_amount":
        if _AMOUNT.search(text):
            return _AMOUNT.sub("₹9,99,999", text, count=1)
        return text + "\nAs a goodwill gesture we will also pay ₹50,000."
    if fault == "letter_dropped_reason":
        lines = text.splitlines()
        i = next((i for i, line in enumerate(lines) if line.strip().startswith("•")), None)
        return "\n".join(lines[:i] + lines[i + 1:]) if i is not None else text
    if fault == "letter_flipped_decision":
        swaps = [(r"We regret that cashless treatment for (.+?) cannot be authorised\.",
                  r"We are pleased to authorise cashless treatment for \1."),
                 (r"We (?:are pleased to )?authorise cashless treatment",
                  "We regret that cashless treatment cannot be authorised. Treatment")]
        for pat, rep in swaps:
            new = re.sub(pat, rep, text, count=1)
            if new != text:
                return new
        return text.replace("\n\nReasons:", "\n\nThis request has been approved.\n\nReasons:", 1)
    return text


class FaultyLLM:
    """Same interface as MockLLM / OllamaLLM; corrupts the base model's answers."""

    def __init__(self, base, fault: str):
        if fault not in FAULTS:
            raise ValueError(f"Unknown fault: {fault}")
        self.base, self.fault = base, fault
        self.target = FAULTS[fault][0]
        self.applied = False  # set when the fault actually changed an answer

    @property
    def name(self) -> str:
        return f"{self.base.name} + injected fault: {self.fault}"

    @property
    def is_mock(self) -> bool:
        return self.base.is_mock

    @property
    def model(self) -> str:
        return self.base.model

    def available(self):
        return self.base.available()

    def generate(self, prompt: str, system: str | None = None, schema: dict | None = None, purpose: str = ""):
        res = self.base.generate(prompt, system=system, schema=schema, purpose=purpose)
        if self.target == "extraction" and TASK_MARKER in prompt:
            text = corrupt_extraction(self.fault, res.text, prompt)
        elif self.target == "letter" and "<<<LETTER" in prompt:
            text = corrupt_letter(self.fault, res.text)
        else:
            return res
        self.applied = self.applied or text != res.text
        return dataclasses.replace(res, text=text)
