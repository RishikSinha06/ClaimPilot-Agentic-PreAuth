"""ClaimPilot — Agentic AI PoC for health-insurance cashless pre-authorisation.

Run:  streamlit run app.py
Optional local LLM:  ollama pull llama3.2   (then choose "Ollama" in the sidebar)
"""
from __future__ import annotations

import json
import re
import time
from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st

from agents import (FAULTS, RECOMMENDED_MODELS, FaultyLLM, OllamaLLM, Orchestrator, append_audit_log, audit_record,
                    make_llm, verify_audit_log)
from agents.evaluation import score
from agents.intake import DOC_LABELS
from agents.nlp.preprocess import ABBREVIATIONS, CONDITIONS, PROCEDURES
from agents.nlp.prompts import EXTRACTION_SCHEMA
from agents.planner import BASE_DOCS
from agents.tools import inr, load_json, total_cost
from viz import charts
from viz import theme as ui
from viz.sequence import message_count, render_svg

ROOT = Path(__file__).resolve().parent
AUDIT_LOG = ROOT / "audit" / "audit_log.jsonl"
st.set_page_config(page_title="ClaimPilot", page_icon="🏥", layout="wide")
st.markdown(ui.CSS, unsafe_allow_html=True)

DECISION_PASTEL = {"APPROVE": ("#0f7b4d", "#dcf5e7"), "PARTIAL": ("#8a5a00", "#fff1cc"), "QUERY": ("#1f5fae", "#dbeafe"),
                   "DENY": ("#b42318", "#fde4e1"), "ESCALATE": ("#5b3fb0", "#ece6fd")}
ENTITY_STYLE = {"CONDITION": "#fde2e1", "PROCEDURE": "#dbeafe", "MEDICATION": "#d9f2ef", "INVESTIGATION": "#ece6fd", "FLAG": "#fff1d6"}

CLAIMS = load_json("claims.json")
POLICIES = load_json("policies.json")
h = ui.h


def md(text) -> str:
    """Escape request text for st.markdown: shown literally, never as Markdown, HTML or LaTeX."""
    return re.sub(r"([\\`*_{}\[\]<>#|~$])", r"\\\1", str(text))


def html(block: str):
    st.markdown(block, unsafe_allow_html=True)


@st.cache_data(ttl=30, show_spinner=False)
def ollama_models(host: str) -> list[str]:
    """Installed Ollama models (cached: an unreachable server costs ~2 s per check on Windows)."""
    try:
        return OllamaLLM(host=host).list_models()
    except Exception:  # noqa: BLE001
        return []


@st.cache_resource(ttl=30, show_spinner=False)
def engine(backend: str, model: str, host: str):
    return make_llm(backend, model, host)


def mode() -> str:
    try:
        return st.context.theme.type or "light"
    except Exception:  # noqa: BLE001
        return "light"


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    html(ui.sidebar_brand())
    html(ui.side_label("Reasoning engine"))
    backend = st.radio("LLM backend", ["Ollama · local open-source LLM", "Simulated LLM (offline)"], label_visibility="collapsed")
    model, host = "llama3.2", "http://localhost:11434"
    if backend.startswith("Ollama"):
        host = st.text_input("Ollama URL", host)
        installed = ollama_models(host)
        options = list(dict.fromkeys([m.replace(":latest", "") for m in installed] + list(RECOMMENDED_MODELS)))
        model = st.selectbox("Model", options, format_func=lambda m: f"{m} — {RECOMMENDED_MODELS.get(m, 'installed')}")
    llm, llm_msg = engine("ollama" if backend.startswith("Ollama") else "mock", model, host)
    (st.success if not llm.is_mock else (st.warning if backend.startswith("Ollama") else st.info))(llm_msg, icon="🧠")
    st.caption("The LLM only reads de-identified notes and polishes letters. Money, dates and policy rules run in "
               "deterministic Python (temperature 0, JSON-schema output).")

    html(ui.side_label("Guardrail demo"))
    fault = st.selectbox("Inject an LLM error", ["none"] + list(FAULTS),
                         format_func=lambda f: "None · normal operation" if f == "none" else f"{f}: {FAULTS[f][1]}")
    if fault != "none":
        llm = FaultyLLM(llm, fault)
        st.warning("Every LLM answer is corrupted on purpose. Watch the guardrails, the Critic and the medical "
                   "officer stop it.", icon="🧪")

    html(ui.side_label("Decision policy"))
    tol = st.slider("Tariff tolerance above network ceiling (%)", 0, 60, 25, 5)
    limit = st.number_input("Auto-approval authority limit (₹)", 50000, 1000000, 300000, 25000)
    max_rev = st.slider("Max Critic → Executor revisions", 0, 3, 2)
    animate = st.toggle("Animate agent trace (for demo)", value=True)
    st.caption("All data in this proof of concept is synthetic.")

orch = Orchestrator(llm=llm, tariff_tolerance_pct=tol, auto_approve_limit=limit, max_revisions=max_rev)


# ---------------------------------------------------------------- helpers
def workflow_graph(ctx: dict) -> str:
    status = ctx["status"]
    nodes = [("plan", "Planner"), ("check_documents", "Intake:\ncheck docs"), ("preprocess_text", "Intake:\npreprocess P1–P9"),
             ("extract_facts", "Intake:\nLLM extract P10–P13"), ("verify_accident_docs", "Intake:\nMLC / FIR\n(if accident)"),
             ("research_policy", "Policy\nResearch"), ("check_eligibility", "Executor:\neligibility"),
             ("calculate_payable", "Executor:\ncalculate\n(if eligible)"), ("validate", "Critic"),
             ("decide", "Decision"), ("draft_letter", "Letter\n(LLM + guardrail)"), ("human_review", "Medical officer\n(if escalated)")]
    lines = ['digraph G { rankdir=LR; bgcolor="transparent"; nodesep=0.28; ranksep=0.36; pad=0.15; '
             'node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=10, penwidth=1.2, margin="0.12,0.07"]; '
             'edge [color="#8a8f98", fontcolor="#8a8f98", fontsize=9, fontname="Helvetica", arrowsize=0.7];']
    for key, label in nodes:
        s = "done" if key == "plan" else status.get(key, "")
        fill, font, line = {"done": ("#dbeafe", "#123a73", "#6da7ec"), "skipped": ("#eef0f3", "#8a8f98", "#c9ced6"),
                            "not needed": ("#eef0f3", "#8a8f98", "#c9ced6"),
                            "awaiting human": ("#fff1cc", "#7a4a00", "#e0a100")}.get(s, ("#ffffff", "#475467", "#c9ced6"))
        label += {"skipped": "\n(skipped)", "not needed": "\n(skipped)", "awaiting human": "\n(waiting)"}.get(s, "")
        lines.append(f'{key} [label="{label}", fillcolor="{fill}", fontcolor="{font}", color="{line}"];')
    lines += ["plan -> check_documents -> preprocess_text -> extract_facts -> verify_accident_docs -> research_policy -> check_eligibility;",
              'check_eligibility -> calculate_payable [label="eligible"];',
              'check_eligibility -> validate [label="not eligible", style=dashed];',
              "calculate_payable -> validate; validate -> decide -> draft_letter;",
              'decide -> human_review [label="ESCALATE", style=dashed];']
    if ctx["revisions"]:
        lines.append(f'validate -> calculate_payable [label="REVISE ×{ctx["revisions"]}", color="#d92d20", fontcolor="#d92d20", penwidth=2, constraint=false];')
    dec = ctx["decision"]["decision"]
    fg, bg = DECISION_PASTEL[dec]
    lines.append(f'out [label="{dec}", shape=ellipse, fillcolor="{bg}", fontcolor="{fg}", color="{fg}", penwidth=2, fontsize=11]; draft_letter -> out; }}')
    return "\n".join(lines)


def show_request(claim: dict):
    pol = POLICIES.get(claim["policy_no"], {})
    c1, c2 = st.columns([1.35, 1], gap="medium")
    with c1:
        html(ui.patient_card(claim, pol.get("product", "unknown policy"), DOC_LABELS, BASE_DOCS))
    with c2:
        c = claim["costs"]
        rows = [("Room rent", f"{inr(c['room_rent_per_day'])} × {c['room_days']} days", c["room_rent_per_day"] * c["room_days"])]
        if c.get("icu_days"):
            icu = c.get("icu_per_day", 0)
            rows.append(("ICU", f"{inr(icu)} × {c['icu_days']} days", icu * c["icu_days"]))
        rows += [("Surgery / OT / anaesthesia", "", c["surgery_ot"]), ("Medicines, consumables, implants", "", c["medicines_consumables"]),
                 ("Investigations", "", c["investigations"]), ("Other (nursing etc.)", "", c["other"])]
        html(ui.cost_card(rows, total_cost(c), inr))


def highlight_sentence(s: dict, ents: list[dict]) -> str:
    text = s["text"]
    out, pos = [], 0
    for e in sorted([e for e in ents if e["sentence"] == s["n"]], key=lambda e: e["start"]):
        out.append(h(text[pos:e["start"]]))
        tag = {"CONDITION": "COND", "PROCEDURE": "PROC", "MEDICATION": "DRUG", "INVESTIGATION": "TEST", "FLAG": "FLAG"}[e["type"]] + (" · NEG" if e["negated"] else "") + (" · ?" if e["uncertain"] else "") + \
            (f" · {e['duration_years']:g} y" if e["duration_years"] is not None else "")
        cls = "ent neg" if e["negated"] else "ent"
        out.append(f"<span class='{cls}' style='background:{ENTITY_STYLE[e['type']]}'>{h(text[e['start']:e['end']])}<small>{tag}</small></span>")
        pos = e["end"]
    out.append(h(text[pos:]))
    return f"<div class='sent'><span class='sn'>S{s['n']}</span>{''.join(out)}</div>"


def mask_value(v: str) -> str:
    return " ".join(w[:2] + "•" * max(1, len(w) - 2) for w in v.split())


def show_llm_pipeline(ctx: dict):
    prep, info = ctx["prep"], ctx["llm_extraction"]
    html(ui.stage_stepper(prep.stage_log))
    st.caption("Every step the doctor's free text goes through before and after the LLM. "
               + ("**LLM backend: simulated** (offline stand-in that reads the same prompt)." if info["simulated"] else f"**LLM backend: {md(info['backend'])}**"))
    log = {s["stage"]: s["summary"] for s in prep.stage_log}
    with st.expander(f"P1 · Assemble — {log['P1']}"):
        st.json(prep.sections)
    with st.expander(f"P2 · Normalise — {log['P2']}"):
        st.code(prep.normalized["notes"], language=None, wrap_lines=True)
    with st.expander(f"P3 · Expand medical abbreviations — {log['P3']}", expanded=True):
        if prep.expansions:
            st.dataframe(pd.DataFrame(prep.expansions, columns=["Abbreviation", "Expanded to", "Count"]), hide_index=True, width="stretch")
        st.code(prep.expanded["notes"], language=None, wrap_lines=True)
    with st.expander(f"P4 · De-identify (PII masking) — {log['P4']}", expanded=bool(prep.pii_map)):
        if prep.pii_map:
            st.dataframe(pd.DataFrame([(k, mask_value(v)) for k, v in prep.pii_map.items()], columns=["Placeholder sent to LLM", "Original (kept locally)"]),
                         hide_index=True, width="stretch")
        st.code(prep.masked["notes"], language=None, wrap_lines=True)
        st.caption("Identifiers never leave the machine. A PII gate re-checks the final prompt before every LLM call.")
    with st.expander(f"P5–P8 · Sentences, entities, negation, durations — {log['P5']}; {log['P6']}; {log['P7']}", expanded=True):
        html("".join(highlight_sentence(s, prep.entities) for s in prep.sentences))
        st.caption("COND = condition (red) · PROC = procedure (blue) · DRUG = medication (teal) · TEST = investigation (purple) · FLAG = cosmetic / accident / declared (yellow). "
                   "Struck-through = negated (NegEx-style scope), '· 8 y' = normalised duration.")
    with st.expander(f"P9 · Token budget — {log['P9']}"):
        st.progress(min(1.0, prep.tokens_est / prep.budget), text=f"~{prep.tokens_est} of {prep.budget} tokens for clinical text "
                    "(≈4 characters per token for Llama-family tokenizers; the real count is returned by Ollama)")
    with st.expander("P10 · Prompt sent to the LLM (system + user, with JSON schema and one worked example)"):
        st.markdown("**System prompt**")
        st.code(info["system"], language=None, wrap_lines=True)
        st.markdown("**User prompt**")
        st.code(info["prompt"], language=None, wrap_lines=True)
        st.markdown(f"**PII gate:** {md(info['pii_gate'])}")
        st.markdown("**JSON schema** (sent in Ollama's `format` field → grammar-constrained decoding)")
        st.json(EXTRACTION_SCHEMA, expanded=False)
    with st.expander(f"P11–P12 · LLM response, parsing and schema validation — {len(info['attempts'])} attempt(s)", expanded=True):
        for a in info["attempts"]:
            meta = f"attempt {a['attempt']}"
            if "error" in a:
                st.error(f"{meta}: {a['error']}")
                continue
            meta += f" · {a.get('prompt_tokens')}→{a.get('output_tokens')} tokens · {a.get('latency_ms')} ms · format {a.get('format')}" + (" · cache hit" if a.get("cached") else "")
            st.markdown(f"**{meta}**")
            st.code(a["raw"], language="json", wrap_lines=True)
            if a["errors"]:
                st.warning("Validator errors → repair prompt sent:\n- " + "\n- ".join(a["errors"]))
            else:
                st.success("Valid against the schema", icon="✅")
        if info["fallback"]:
            st.warning(info["fallback"])
    with st.expander("P13 · Grounding (hallucination guard) and cross-check with the rule-based extractor", expanded=True):
        if info["grounding"]:
            gdf = pd.DataFrame(info["grounding"])
            gdf["cited"] = gdf["cited"].map(lambda v: f"S{v}" if v is not None else "—")
            st.dataframe(gdf.rename(columns={"condition": "Claim", "cited": "Cited sentence", "status": "Status"}), hide_index=True, width="stretch")
        cmp = pd.DataFrame(info["compare"])
        cmp["llm"], cmp["rules"] = cmp["llm"].astype(str), cmp["rules"].astype(str)
        cmp["agree"] = cmp["agree"].map({True: "✅", False: "❌"})
        st.dataframe(cmp.rename(columns={"field": "Field", "llm": "LLM", "rules": "Rules", "agree": "Agree"}), hide_index=True, width="stretch")
        st.markdown("**Final facts handed to the other agents**")
        st.json(ctx["extracted"])


def _resubmit(claim: dict):
    """Callback: runs before the next rerun, so it may change widget state."""
    st.session_state["custom_claim"] = Orchestrator.resubmit(claim)
    st.session_state["src"] = "Enter a new request"
    st.session_state["autorun"] = True


def _clear_custom():
    st.session_state.pop("custom_claim", None)
    st.session_state.pop("result", None)


def show_result(ctx: dict):
    dec, calc = ctx["decision"], ctx.get("calculation")
    html(ui.decision_card(dec, inr))
    m = st.columns(4)
    m[0].metric("Agent steps", len(ctx["trace"]), border=True)
    m[1].metric("Critic revisions", ctx["revisions"], border=True)
    m[2].metric("Agent compute", f"{ctx['elapsed_ms']:.0f} ms", border=True)
    llm_ms = sum(a.get("latency_ms") or 0 for a in ctx["llm_extraction"]["attempts"])
    m[3].metric("LLM time", f"{llm_ms / 1000:.1f} s" if llm_ms >= 1000 else f"{llm_ms:.0f} ms", border=True,
                help="Extraction call(s) only; the letter polish adds one more call.")

    # ---------- human in the loop
    if dec["decision"] == "ESCALATE":
        with st.container(border=True):
            causes = dec.get("escalation_causes", [])
            html(ui.section("🩺 Medical officer review", "The agents stopped here on purpose: a human decides."))
            html("".join(ui.chip(c, "violet") for c in causes) + f'<div class="pa-tl-txt" style="margin:6px 0 10px">{h("; ".join(dec["reasons"]))}</div>')
            c1, c2 = st.columns([1, 2])
            action = c1.radio("Decision", ["approve", "deny", "query"], format_func=lambda a: {
                "approve": "✓ Approve", "deny": "✕ Deny", "query": "? Query the hospital"}[a])
            c1.caption("The letter reason follows the escalation cause and the action chosen.")
            reviewer = c2.text_input("Reviewer", "Dr. A. Kulkarni (Medical Officer)")
            remarks = c2.text_input("Remarks", "", placeholder="e.g. Hypertension on treatment since 2018 was not declared at proposal.")
            if st.button("Submit review", type="primary"):
                orch.human_review(ctx, action, remarks, reviewer)
                append_audit_log(ctx, AUDIT_LOG)
                st.rerun()
    if dec["decision"] == "QUERY":
        with st.container(border=True):
            html(ui.section("🏥 Hospital response to the query", "Simulates the hospital uploading the requested documents."))
            html("".join(ui.chip(r, "info") for r in dec["reasons"]))
            st.button("Upload requested documents and re-run", type="primary", on_click=_resubmit, args=(ctx["claim"],))

    tabs = st.tabs(["🔀 Workflow", "🧬 Execution sequence", "🧠 LLM text pipeline", "🧾 Eligibility", "🧮 Amount",
                    "🧐 Critic", "✉️ Letter", "🧭 Plan", "💬 Messages", "🔍 Audit"])
    with tabs[0]:
        st.graphviz_chart(workflow_graph(ctx), width="stretch")
        html(ui.chip("executed", "info") + ui.chip("skipped by a run-time condition", "neutral")
             + ui.chip("waiting for a human", "warn") + ui.chip("red edge: Critic sent work back", "bad"))
    with tabs[1]:
        collapse = st.toggle("Collapse the 9 NLP stages into one arrow", value=False)
        svg = render_svg(ctx, collapse_nlp=collapse, title=f"Execution sequence · {ctx['claim']['claim_id']}")
        st.caption(f"{message_count(ctx, collapse)} messages, drawn from the audit trace of this exact run.")
        html(f"<div class='seqwrap'>{svg}</div>")
        st.download_button("Download diagram (.svg)", svg, file_name=f"{ctx['claim']['claim_id']}_sequence.svg", mime="image/svg+xml")
    with tabs[2]:
        show_llm_pipeline(ctx)
    with tabs[3]:
        checks = ctx["eligibility"]["checks"]
        n = {r: sum(c["result"] == r for c in checks) for r in ("PASS", "WARN", "FAIL")}
        html(ui.section("Eligibility rules", f"{n['PASS']} passed · {n['WARN']} warnings · {n['FAIL']} failed")
             + ui.check_list(checks))
    with tabs[4]:
        if calc:
            html(ui.section("Payable amount waterfall", "From the hospital's estimate to what the policy pays, one deduction at a time"))
            st.altair_chart(charts.waterfall(calc["steps"], inr, mode()), width="stretch")
            df = pd.DataFrame(calc["steps"])
            df["amount"] = df["amount"].map(inr)
            st.dataframe(df.rename(columns=str.title), hide_index=True, width="stretch")
        else:
            html(ui.empty_state("🧮", "No amount computed", "The claim failed eligibility, so the payable calculation was skipped."))
    with tabs[5]:
        for i, rnd in enumerate(ctx["critic_history"], start=1):
            last = i == len(ctx["critic_history"])
            n_fail = sum(f["status"] == "FAIL" for f in rnd["findings"])
            with st.expander(f"Critic round {i} · verdict {rnd['verdict']} · confidence {rnd['confidence']:.0%} · {n_fail} failed", expanded=last):
                html(ui.findings_list(rnd["findings"]))
        st.caption("PASS = forward to decision · REVISE = back to Executor · HOLD = query the hospital or escalate to a human.")
    with tabs[6]:
        li = ctx["letter_info"]
        html(ui.letter(dec["letter"], f"via {dec['letter_method']}"))
        if li["problems"]:
            st.warning("Faithfulness check rejected the LLM letter: " + "; ".join(li["problems"]))
        c1, c2 = st.columns([1, 3])
        c1.download_button("Download letter (.txt)", dec["letter"], file_name=f"{ctx['claim']['claim_id']}_letter.txt")
        with c2.expander("What was sent to the LLM (de-identified)"):
            st.code(li["deidentified"], language=None, wrap_lines=True)
    with tabs[7]:
        plan = ctx["plan"]
        html(ui.chip(plan["priority"], "bad" if plan["priority"].startswith("HIGH") else "info")
             + ui.chip(plan["claim_profile"], "neutral") + "".join(ui.chip(s, "violet") for s in plan["special_checks"]))
        html(ui.task_list(plan["tasks"], ctx["status"], ctx["revisions"]))
    with tabs[8]:
        st.dataframe(pd.DataFrame(ctx["messages"]).rename(columns=str.title), hide_index=True, width="stretch", height=420)
        st.caption("Agents never call each other directly; every message goes through the Orchestrator and the shared state.")
    with tabs[9]:
        rec = audit_record(ctx)
        st.download_button("Download full audit record (.json)", json.dumps(rec, indent=2, ensure_ascii=False),
                           file_name=f"{ctx['claim']['claim_id']}_audit.json", mime="application/json")
        for t in ctx["trace"]:
            with st.expander(f"{t['n']}. {t['icon']} {t['agent']} · {t['task']} ({t['ms']} ms, {len(t['calls'])} calls)"):
                st.write(t["summary"])
                if t["calls"]:
                    st.dataframe(pd.DataFrame(t["calls"]).astype(str), hide_index=True, width="stretch")
                st.json(t["details"], expanded=False)
        if AUDIT_LOG.exists():
            html(ui.section("Audit log", "latest entries, newest first"))
            chain = verify_audit_log(AUDIT_LOG)
            if chain["ok"]:
                html(ui.chip(f"🔗 hash chain intact · {chain['verified']} entries verified", "good"))
            else:
                st.error(f"Audit log hash chain broken at line {chain['bad_line']}: an earlier entry was edited or removed.")
            lines = [json.loads(x) for x in AUDIT_LOG.read_text(encoding="utf-8").splitlines()[-15:] if x.strip()]
            log = pd.DataFrame(lines[::-1])
            for col in ("hash", "prev_hash"):
                if col in log:
                    log[col] = log[col].map(lambda v: v[:12] + "…" if isinstance(v, str) else "")
            st.dataframe(log, hide_index=True, width="stretch")


def run_with_trace(claim: dict) -> dict:
    events: list[dict] = []
    with st.status("Agents working on the request…", expanded=True) as status:
        board = st.empty()

        def on_event(e):
            events.append(e)
            board.markdown(ui.timeline(events), unsafe_allow_html=True)
            if animate:
                time.sleep(0.3)
        ctx = orch.run(claim, on_event=on_event)
        status.update(label=f"Done · {len(ctx['trace'])} agent steps · decision: {ctx['decision']['decision']}", state="complete", expanded=False)
    append_audit_log(ctx, AUDIT_LOG)
    return ctx


def new_request_form() -> dict | None:
    with st.form("new_req", border=True):
        html(ui.section("New pre-authorisation request", "Abbreviations are welcome in the notes: the NLP pipeline expands them"))
        a, b, c = st.columns(3)
        pol_no = a.selectbox("Policy", list(POLICIES), format_func=lambda p: f"{p} · {POLICIES[p]['product']}")
        patient = b.selectbox("Patient", POLICIES[pol_no]["insured"] + ["(not on policy)"])
        age = c.number_input("Age", 0, 100, 45)
        a, b, c = st.columns(3)
        adm = a.selectbox("Admission", ["planned", "emergency"])
        trt = b.selectbox("Treatment", ["surgical", "medical"])
        req_date = c.date_input("Request date", value=date(2026, 10, 13))
        a, b = st.columns([2, 1])
        diagnosis = a.text_input("Diagnosis", "Symptomatic cholelithiasis")
        icd = b.text_input("ICD-10", "K80.20")
        treatment = st.text_input("Proposed treatment", "Laparoscopic cholecystectomy")
        notes = st.text_area("Doctor's notes", "C/o RUQ pain after fatty meals x 3 mths. USG: multiple gallstones. "
                             "K/c/o HTN since 2015 on Tab. Telmisartan. No h/o DM. Planned lap cholecystectomy.")
        st.markdown("**Cost estimate (₹)**")
        cols = st.columns(4)
        rr = cols[0].number_input("Room rent / day", 0, 50000, 4000, 500)
        rd = cols[1].number_input("Room days", 0, 60, 3)
        iu = cols[2].number_input("ICU / day", 0, 100000, 0, 1000)
        idays = cols[3].number_input("ICU days", 0, 30, 0)
        cols = st.columns(4)
        so = cols[0].number_input("Surgery / OT", 0, 2000000, 80000, 5000)
        med = cols[1].number_input("Medicines & implants", 0, 2000000, 30000, 5000)
        inv = cols[2].number_input("Investigations", 0, 500000, 10000, 1000)
        oth = cols[3].number_input("Other", 0, 500000, 8000, 1000)
        docs = st.multiselect("Documents attached", list(DOC_LABELS), default=["preauth_form", "id_card", "doctor_notes", "investigation_reports"],
                              format_func=lambda d: DOC_LABELS[d])
        ok = st.form_submit_button("Use this request", type="primary")
    if ok:
        return {"claim_id": "PA-2026-NEW", "hospital": "Demo Hospital, Pune", "policy_no": pol_no, "patient": patient,
                "age": int(age), "gender": "-", "request_date": str(req_date), "admission_type": adm, "treatment_type": trt,
                "diagnosis": diagnosis, "icd10": icd, "proposed_treatment": treatment, "doctor_notes": notes, "documents": docs,
                "costs": {"room_rent_per_day": rr, "room_days": rd, "icu_per_day": iu, "icu_days": idays,
                          "surgery_ot": so, "medicines_consumables": med, "investigations": inv, "other": oth}}
    return None


# ---------------------------------------------------------------- main
html(ui.hero(llm.name, llm.is_mock, fault if fault != "none" else None))

tab_run, tab_batch, tab_data, tab_arch = st.tabs(["⚡ Process a request", "📊 Batch evaluation", "🗂️ Data & dictionaries", "🏗️ Architecture"])

with tab_run:
    st.session_state.setdefault("src", "Sample request")
    top_l, top_r = st.columns([1, 2.2], vertical_alignment="bottom")
    src = top_l.segmented_control("Request source", ["Sample request", "Enter a new request"], required=True, key="src")
    claim = None
    if src == "Sample request":
        claim = top_r.selectbox("Choose a synthetic request", CLAIMS,
                                format_func=lambda c: f"{c['claim_id']} · {c['patient']} · {c['proposed_treatment']} (expected {c['expected_decision']})")
    else:
        if (st.session_state.get("custom_claim") or {}).get("resubmission_of"):
            top_r.info(f"Re-submission of {st.session_state['custom_claim']['resubmission_of']} with the requested documents.", icon="🔁")
            st.button("Start a different new request", on_click=_clear_custom)
        else:
            new = new_request_form()
            if new:
                st.session_state["custom_claim"] = new
        claim = st.session_state.get("custom_claim")

    res = st.session_state.get("result")
    showing = bool(claim and res and st.session_state.get("result_for") == (claim["claim_id"], claim.get("resubmission_of")))
    flow_slot = st.empty()
    flow_slot.markdown(ui.agent_flow(res if showing else None), unsafe_allow_html=True)
    if claim:
        show_request(claim)
        run_clicked = st.button("▶  Run the agents", type="primary", width="stretch")
        if run_clicked or st.session_state.pop("autorun", False):
            st.session_state["result"] = res = run_with_trace(claim)
            st.session_state["result_for"] = (claim["claim_id"], claim.get("resubmission_of"))
            showing = True
            flow_slot.markdown(ui.agent_flow(res), unsafe_allow_html=True)
        if showing:
            show_result(res)

with tab_batch:
    html(ui.section("Batch evaluation", "Every labelled synthetic request, compared with hand-labelled decisions, amounts and facts"))
    c1, c2 = st.columns([1, 2.2], vertical_alignment="bottom")
    manual_min = c1.number_input("Manual handling time per request (min, assumed)", 5, 240, 45, 5)
    run_all = c2.button("▶  Run all 12 requests", type="primary", width="stretch")
    if run_all:
        rows, prog = [], st.progress(0.0, text="Running…")
        for i, c in enumerate(CLAIMS):
            ctx = orch.run(c)
            d, s = ctx["decision"], score(c, ctx)
            info = ctx["llm_extraction"]
            rows.append({"Request": c["claim_id"], "Treatment": c["proposed_treatment"], "Expected": c["expected_decision"],
                         "Agents": d["decision"], "Match": "✅" if s["decision_ok"] else "❌",
                         "Outcome": {"correct": "✅ correct", "held": "🟡 held for review", "wrong": "❌ wrong decision sent"}[s["outcome"]],
                         "Facts correct": s["facts_correct"], "_outcome": s["outcome"],
                         "Amount ok": "✅" if s["amount_ok"] else "❌",
                         "Requested": d["requested"], "Authorised": d["amount"] if d["decision"] in ("APPROVE", "PARTIAL") else 0,
                         "Confidence": d["confidence"], "Revisions": ctx["revisions"], "LLM attempts": len(info["attempts"]),
                         "Fallback": "yes" if info["fallback"] else "", "LLM ms": round(sum(a.get("latency_ms") or 0 for a in info["attempts"])),
                         "Agent ms": ctx["elapsed_ms"]})
            prog.progress((i + 1) / len(CLAIMS), text=f"{c['claim_id']} done")
        prog.empty()
        st.session_state["batch"] = pd.DataFrame(rows)
        st.session_state["batch_llm"] = (llm.name, llm.is_mock)
    df = st.session_state.get("batch")
    if df is None:
        html(ui.empty_state("📊", "No batch run yet",
                            "Run all 12 labelled requests to see accuracy, safety outcomes, a confusion matrix and amounts."))
    else:
        engine, simulated = st.session_state.get("batch_llm")
        html(ui.chip(f"engine: {engine}", "violet" if not simulated else "warn"))
        note = {"delta_color": "off", "delta_arrow": "off", "border": True}
        wrong, held = (df["_outcome"] == "wrong").sum(), (df["_outcome"] == "held").sum()
        payable = df[df["Agents"].isin(["APPROVE", "PARTIAL"])]
        k = st.columns(3)
        k[0].metric("Decision accuracy", f"{(df['Match'] == '✅').mean():.0%}", f"{(df['Match'] == '✅').sum()} of {len(df)} cases", **note)
        k[1].metric("Wrong decisions sent", int(wrong), f"{held} held for review", **note,
                    help="A wrong decision that went out without a human. A case held for review (escalated or "
                         "queried instead of the labelled decision) is safe but costs extra work.")
        k[2].metric("Fact extraction", f"{df['Facts correct'].sum() / (5 * len(df)):.0%}", f"{df['Facts correct'].sum()} of {5 * len(df)} fields", **note)
        k = st.columns(3)
        k[0].metric("Amounts exact", f"{(df['Amount ok'] == '✅').mean():.0%}", "vs hand calculation", **note)
        k[1].metric("Manual effort replaced", f"{manual_min * len(df) / 60:.1f} h", f"assumes {manual_min} min per request", **note)
        k[2].metric("Non-payable identified", inr((payable["Requested"] - payable["Authorised"]).sum()), "limits, caps and co-pay", **note)
        if simulated:
            st.info("Simulated LLM: it answers with the same rule extractor the Critic cross-checks against, so the fact "
                    "accuracy here measures the rule extractor, not a language model. Real-model results are in "
                    "`results/` (`python scripts/benchmark_llm.py`).", icon="ℹ️")
        c1, c2 = st.columns([1.1, 1], gap="medium")
        with c1, st.container(border=True):
            html(ui.section("Confusion matrix", "rows = expected, columns = agents; the diagonal is correct"))
            labels = ["APPROVE", "PARTIAL", "QUERY", "ESCALATE", "DENY"]
            cm = pd.crosstab(df["Expected"], df["Agents"]).reindex(index=labels, columns=labels, fill_value=0)
            st.altair_chart(charts.confusion_heatmap(cm), width="stretch")
        with c2, st.container(border=True):
            html(ui.section("Authorised vs requested", "approved and partially approved requests"))
            st.altair_chart(charts.authorised_vs_requested(payable[["Request", "Requested", "Authorised"]], inr, mode()), width="stretch")
        show = df.drop(columns=["_outcome"]).copy()
        show["Requested"] = show["Requested"].map(inr)
        show["Authorised"] = show["Authorised"].map(inr)
        st.dataframe(show, hide_index=True, width="stretch", height=(len(show) + 1) * 35 + 3,
                     column_config={"Confidence": st.column_config.ProgressColumn("Confidence", min_value=0, max_value=1, format="percent"),
                                    "Facts correct": st.column_config.ProgressColumn("Facts correct", min_value=0, max_value=5, format="%d/5")})
        st.caption("Manual time is an assumption you can change above, not a measured figure. For a real-LLM benchmark run "
                   "`python scripts/benchmark_llm.py --model llama3.2`.")

with tab_data:
    # a switch rather than nested tabs: tables drawn inside hidden nested tabs mount with zero-width columns
    view = st.segmented_control("Data set", ["Policies", "Network tariffs", "Policy wording", "NLP dictionaries"],
                                required=True, default="Policies", label_visibility="collapsed", key="data_view")
    if view == "Policies":
        html(ui.section("Policy master", "five synthetic policies"))
        pdf = pd.DataFrame(POLICIES.values())
        pdf["sum_insured"] = pdf["sum_insured"].map(inr)
        for col in ("insured", "declared_ped", "sub_limits", "exclusions"):
            pdf[col] = pdf[col].map(lambda v: ", ".join(f"{k}: {inr(x)}" for k, x in v.items()) if isinstance(v, dict) else ", ".join(v))
        st.dataframe(pdf[["policy_no", "product", "insured", "start_date", "end_date", "sum_insured", "room_rent_cap_per_day",
                          "copay_pct", "ped_waiting_months", "declared_ped", "waiting_waived", "sub_limits", "exclusions"]],
                     hide_index=True, width="stretch")
    elif view == "Network tariffs":
        tariffs = load_json("tariffs.json")
        html(ui.section("Network tariff ranges", "package price per procedure; the Critic flags estimates far above the ceiling"))
        with st.container(border=True):
            st.altair_chart(charts.tariff_ranges(tariffs, inr, mode()), width="stretch")
        mm = tariffs.get("medical management")
        if mm:
            st.caption(f"Medical management is priced per day: {inr(mm['min'])} – {inr(mm['max'])}.")
    elif view == "Policy wording":
        html(ui.section("Policy wording", "11 clauses the agents cite in their reasons"))
        clauses = load_json("policy_clauses.json")
        cols = st.columns(2, gap="medium")
        for i, c in enumerate(clauses):
            cols[i % 2].markdown(f'<div class="pa-card"><div class="pa-kicker">Clause {h(c["id"])}</div>'
                                 f'<div class="pa-tl-head" style="font-size:1rem">{h(c["title"])}</div>'
                                 f'<div class="pa-tl-txt" style="margin:4px 0 8px">{h(c["text"])}</div>'
                                 + "".join(ui.chip(t, "neutral") for t in c["tags"]) + "</div>", unsafe_allow_html=True)
    else:
        c1, c2 = st.columns(2, gap="medium")
        with c1:
            html(ui.section("Medical abbreviations", f"{len(ABBREVIATIONS)} entries, expanded in stage P3"))
            st.dataframe(pd.DataFrame(ABBREVIATIONS, columns=["Abbreviation", "Expansion", "Case-sensitive"]), hide_index=True, width="stretch", height=420)
        with c2:
            html(ui.section("Clinical gazetteer", "entity dictionary used in stage P6"))
            gz = [("CONDITION", k, ", ".join(v)) for k, v in CONDITIONS.items()] + [("PROCEDURE", k, ", ".join(v)) for k, v in PROCEDURES.items()]
            st.dataframe(pd.DataFrame(gz, columns=["Type", "Canonical", "Synonyms"]), hide_index=True, width="stretch", height=420)

with tab_arch:
    figs = [("architecture_system.png", "System architecture", "Layers, agents, tools, LLM runtime, guardrails, data and the numbered flow"),
            ("architecture_sequence.png", "Execution sequence", "Every call of one real run (PA-2026-0108), in order"),
            ("architecture_llm_pipeline.png", "LLM text pipeline", "How text is prepared for, sent to, and checked after the LLM (P1–P13)"),
            ("architecture_decision_logic.png", "Decision logic", "Eligibility rules, payable waterfall, Critic checks, decision ladder")]
    for i, (f, title, sub) in enumerate(figs, start=1):
        p = ROOT / "assets" / f
        if p.exists():
            with st.container(border=True):
                html(ui.section(f"{i} · {title}", sub))
                st.image(str(p), width="stretch")
