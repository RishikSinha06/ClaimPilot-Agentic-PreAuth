"""Integration tests: the real OllamaLLM client talking HTTP to a fake Ollama server."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from fake_ollama import FakeOllama  # noqa: E402

from agents import OllamaLLM, Orchestrator  # noqa: E402
from agents.llm import _CACHE  # noqa: E402

CLAIMS = json.loads((ROOT / "data" / "claims.json").read_text(encoding="utf-8"))
BY_ID = {c["claim_id"]: c for c in CLAIMS}


def run(server, claim):
    _CACHE.clear()
    return Orchestrator(llm=OllamaLLM(model="llama3.2", host=server.host)).run(claim)


def test_all_cases_over_http_with_schema_and_no_pii():
    with FakeOllama() as srv:
        assert OllamaLLM(host=srv.host).available()[0]
        for c in CLAIMS:
            ctx = run(srv, c)
            assert ctx["decision"]["decision"] == c["expected_decision"], c["claim_id"]
            assert len(ctx["llm_extraction"]["attempts"]) == 1
        extraction = [r for r in srv.requests if "### Task" in r["prompt"]]
        assert all(isinstance(r["format"], dict) for r in extraction)            # JSON schema sent
        assert all(r["options"]["temperature"] == 0 for r in srv.requests)       # deterministic
        for c in CLAIMS:                                                         # PII never sent
            assert all(c["patient"] not in r["prompt"] + r.get("system", "") for r in srv.requests)
        assert all("9000000001" not in r["prompt"] for r in srv.requests)


def test_invalid_json_triggers_repair_prompt():
    with FakeOllama(mode="bad_then_good") as srv:
        ctx = run(srv, BY_ID["PA-2026-0104"])
    attempts = ctx["llm_extraction"]["attempts"]
    assert len(attempts) == 2 and attempts[0]["errors"] and not attempts[1]["errors"]
    assert "rejected by the validator" in srv.requests[1]["prompt"]
    assert ctx["decision"]["decision"] == "DENY"


def test_persistent_bad_output_falls_back_to_rules():
    with FakeOllama(mode="always_bad") as srv:
        ctx = run(srv, BY_ID["PA-2026-0101"])
    assert ctx["llm_extraction"]["fallback"]
    assert len(ctx["llm_extraction"]["attempts"]) == 3
    assert ctx["decision"]["decision"] == "APPROVE"


def test_hallucinated_condition_is_rejected_by_grounding():
    with FakeOllama(mode="hallucinate") as srv:
        ctx = run(srv, BY_ID["PA-2026-0111"])  # notes say "No h/o ... asthma"
    statuses = {g["condition"]: g["status"] for g in ctx["llm_extraction"]["grounding"]}
    assert statuses["asthma"] == "rejected: negated in text"
    assert ctx["extracted"]["chronic_conditions"] == []
    assert ctx["decision"]["decision"] == "APPROVE"


def test_unfaithful_letter_is_replaced_by_template():
    with FakeOllama(mode="unfaithful_letter") as srv:
        ctx = run(srv, BY_ID["PA-2026-0101"])
    assert "9,99,999" not in ctx["decision"]["letter"]
    assert ctx["decision"]["letter_method"].startswith("template")


def test_echoed_letter_markers_never_reach_the_hospital():
    with FakeOllama(mode="echo_markers") as srv:
        ctx = run(srv, BY_ID["PA-2026-0102"])
    letter = ctx["decision"]["letter"]
    assert ctx["decision"]["letter_method"] == "Ollama · llama3.2"
    assert "<<<LETTER" not in letter and "LETTER>>>" not in letter
    assert letter.startswith("To: Cashless Desk")


def test_old_ollama_without_structured_outputs():
    with FakeOllama(mode="old_server") as srv:
        ctx = run(srv, BY_ID["PA-2026-0102"])
    assert ctx["llm_extraction"]["attempts"][0]["format"] == "json"
    assert ctx["decision"]["decision"] == "PARTIAL"


def test_server_down_falls_back_to_mock():
    from agents import make_llm
    llm, msg = make_llm("ollama", host="http://127.0.0.1:9")
    assert llm.is_mock and "not running" in msg
