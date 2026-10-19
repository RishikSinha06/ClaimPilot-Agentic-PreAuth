"""Prompt templates and the JSON schema the LLM must follow (stage P10).

The schema is sent to Ollama in the `format` field. Ollama then uses
grammar-constrained decoding, so the model can only produce tokens that keep
the output valid against this schema. We still validate locally (P12) because
older Ollama versions only support plain `format: "json"`.
"""
from __future__ import annotations

import json
import re

from .preprocess import ALL_CONDITION_KEYS, ALL_PROCEDURE_KEYS

EXTRACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "procedure_key": {"type": "string", "enum": ALL_PROCEDURE_KEYS},
        "chronic_conditions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "condition": {"type": "string", "enum": ALL_CONDITION_KEYS},
                    "duration_years": {"type": ["number", "null"]},
                    "evidence_sentence": {"type": "integer"},
                },
                "required": ["condition", "duration_years", "evidence_sentence"],
            },
        },
        "cosmetic_intent": {"type": "boolean"},
        "functional_need": {"type": "boolean"},
        "accident": {"type": "boolean"},
    },
    "required": ["procedure_key", "chronic_conditions", "cosmetic_intent", "functional_need", "accident"],
}

EXTRACTION_SYSTEM = """You are a clinical information-extraction assistant at a health-insurance pre-authorisation desk in India.
You read DE-IDENTIFIED hospital notes split into numbered sentences (S1, S2, ...) and return ONLY one JSON object that matches the schema.
Rules:
1. Extract only facts that are explicitly written. Never guess.
2. Ignore negated findings: "no history of diabetes" means diabetes is NOT present.
3. chronic_conditions = long-standing diseases the patient already has (not the current acute illness). Give duration_years if stated; convert "since <year>" using the request date. evidence_sentence = the number of the sentence that states it.
4. procedure_key must be one of the allowed values. Use "medical management" when no surgery or procedure is planned.
5. accident = true only if the current illness was caused by an accident or injury.
6. cosmetic_intent = true only if the treatment is wanted to change appearance. functional_need = true only when such a cosmetic-type treatment is needed to restore a bodily function (e.g. a blocked nose) or for reconstruction after an accident, burn or cancer. Treating an ordinary illness or injury is NOT functional_need: use false.
7. Never output names, phone numbers or identifiers. Output JSON only, no explanations."""

FEW_SHOT_INPUT = """Request date: 2025-03-01
Patient: [PATIENT], age 46, M
S1: Diagnosis: Right inguinal hernia.
S2: Proposed treatment: Laparoscopic hernia repair.
S3: Known case of asthma for 10 years on inhaler.
S4: No history of diabetes mellitus or hypertension."""

FEW_SHOT_OUTPUT = {
    "procedure_key": "hernia",
    "chronic_conditions": [{"condition": "asthma", "duration_years": 10, "evidence_sentence": 3}],
    "cosmetic_intent": False, "functional_need": False, "accident": False,
}

TASK_MARKER = "### Task"


def build_extraction_prompt(prep, claim: dict) -> tuple[str, str]:
    """Returns (system_prompt, user_prompt). Only de-identified text is used."""
    task = (f"Request date: {claim['request_date']}\n"
            f"Patient: [PATIENT], age {claim.get('age', '?')}, {claim.get('gender', '?')}\n"
            f"{prep.numbered_text()}")
    user = (
        "### Allowed values\n"
        f"procedure_key: {', '.join(ALL_PROCEDURE_KEYS)}\n"
        f"condition: {', '.join(ALL_CONDITION_KEYS)}\n\n"
        "### Example\nInput:\n" + FEW_SHOT_INPUT + "\nOutput:\n" + json.dumps(FEW_SHOT_OUTPUT) + "\n\n"
        f"{TASK_MARKER}\nInput:\n{task}\nOutput:\n"
    )
    return EXTRACTION_SYSTEM, user


def build_repair_prompt(original_prompt: str, bad_output: str, errors: list[str]) -> str:
    return (original_prompt.rstrip() + "\n" + bad_output.strip()[:600] + "\n\n"
            "### Your previous output was rejected by the validator:\n- " + "\n- ".join(errors[:6]) +
            "\nReturn the corrected JSON object only.\nOutput:\n")


LETTER_SYSTEM = """You polish insurance letters. Rewrite the opening paragraph of the letter between <<<LETTER and LETTER>>> so it is clear, polite and professional.
Copy every other line exactly as written, character for character: the To and Ref lines, every line under "Reasons:", any medical officer remarks and the sign-off.
Keep every amount, clause number and placeholder in square brackets. Do not add facts, amounts or promises.
Return only the letter text, without the <<<LETTER and LETTER>>> markers."""


def build_letter_prompt(deidentified_letter: str) -> tuple[str, str]:
    return LETTER_SYSTEM, f"<<<LETTER\n{deidentified_letter}\nLETTER>>>"


def clean_letter_output(text: str) -> str:
    """Strip code fences and echoed <<<LETTER / LETTER>>> markers from the model's answer."""
    t = re.sub(r"^```\w*\s*|\s*```$", "", text.strip())
    t = re.sub(r"^<<<LETTER\s*|\s*LETTER>>>$", "", t.strip())
    return t.strip()
