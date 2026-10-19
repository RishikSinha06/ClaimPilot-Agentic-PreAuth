"""Unit tests for the text preprocessing pipeline and LLM output handling."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agents.nlp.parsing import extract_json, parse_and_validate  # noqa: E402
from agents.nlp.preprocess import (p2_normalize, p3_expand, p4_deidentify, preprocess_claim)  # noqa: E402
from agents.nlp.prompts import EXTRACTION_SCHEMA  # noqa: E402

CLAIM = {"claim_id": "PA-X", "patient": "Lata Kulkarni", "hospital": "Jehangir Heart Institute, Pune",
         "policy_no": "P1002", "request_date": "2026-10-10", "age": 64, "gender": "F",
         "diagnosis": "Coronary artery disease", "proposed_treatment": "PTCA with DES",
         "doctor_notes": "Mrs Lata Kulkarni c/o chest pain x 3 mths. K/c/o HTN since 2018. "
                         "No h/o DM/asthma. Contact 9000000001."}


def test_normalise_units_and_punctuation():
    assert p2_normalize("Tab  Amlodipine 5mg ,  once “daily”") == 'Tab Amlodipine 5 mg, once "daily"'


def test_abbreviation_expansion():
    out, used = p3_expand("K/c/o HTN, T2DM x 4 yrs. No h/o DM/HTN.")
    assert out == "Known case of hypertension, type 2 diabetes mellitus for 4 years. No history of diabetes mellitus/hypertension."
    assert ("HTN", "hypertension", 2) in used


def test_pii_is_masked():
    masked, pii = p4_deidentify({"notes": CLAIM["doctor_notes"]}, CLAIM)
    assert "Kulkarni" not in masked["notes"] and "9000000001" not in masked["notes"]
    assert pii["[PATIENT]"] == "Lata Kulkarni" and pii["[PHONE_1]"] == "9000000001"


def test_negation_duration_and_sentences():
    prep = preprocess_claim(CLAIM)
    conds = {e["canonical"]: e for e in prep.entities if e["type"] == "CONDITION"}
    assert conds["hypertension"]["negated"] is False and conds["hypertension"]["duration_years"] == 8
    assert conds["diabetes"]["negated"] is True and conds["asthma"]["negated"] is True
    assert prep.sentences[0]["text"].startswith("Diagnosis:")
    assert all("Kulkarni" not in s["text"] for s in prep.sentences)


def test_json_extraction_handles_chatter_and_fences():
    assert extract_json('Sure!\n```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json('text before {"a": {"b": 2}} after') == {"a": {"b": 2}}


def test_schema_validation_catches_bad_values():
    obj, errs = parse_and_validate('{"procedure_key": "brain surgery", "chronic_conditions": [], '
                                   '"cosmetic_intent": "no", "functional_need": false, "accident": false}', EXTRACTION_SCHEMA)
    assert obj is None
    assert any("not an allowed value" in e for e in errs) and any("expected boolean" in e for e in errs)


def test_canonicalisation_maps_synonyms():
    obj, errs = parse_and_validate('{"procedure_key": "Hernia", "chronic_conditions": [{"condition": "High blood pressure", '
                                   '"duration_years": 3, "evidence_sentence": "S4"}], "cosmetic_intent": false, '
                                   '"functional_need": false, "accident": false}', EXTRACTION_SCHEMA)
    assert not errs and obj["chronic_conditions"][0] == {"condition": "hypertension", "duration_years": 3, "evidence_sentence": 4}
