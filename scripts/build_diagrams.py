"""Builds the architecture diagrams from the real code and a real run.

    python scripts/build_diagrams.py            # writes HTML / SVG / DOT sources
    python scripts/build_diagrams.py --png      # also renders PNGs (needs `pip install playwright`
                                                #   and, without Microsoft Edge, `playwright install chromium`)

Outputs in assets/diagrams/ and assets/:
  llm_pipeline.html      -> architecture_llm_pipeline.png   (stages P1-P13 with live example data)
  sequence_full.svg      -> architecture_sequence.png       (every call of run PA-2026-0108)
  sequence_part1/2.svg   (same, split in two for slides)
  decision_logic.html    -> architecture_decision_logic.png (rules, waterfall, Critic, decision ladder)
  system.html            -> architecture_system.png         (hand-laid-out system architecture)
"""
from __future__ import annotations

import html
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agents import Orchestrator  # noqa: E402
from agents.nlp.preprocess import preprocess_claim  # noqa: E402
from viz.sequence import message_count, render_svg  # noqa: E402

D = ROOT / "assets" / "diagrams"
CLAIMS = {c["claim_id"]: c for c in json.loads((ROOT / "data" / "claims.json").read_text(encoding="utf-8"))}
esc = html.escape


def short(t: str, n: int) -> str:
    return t if len(t) <= n else t[: n - 1] + "…"


# ---------------------------------------------------------------- LLM pipeline (P1-P13)
def llm_pipeline_html() -> str:
    claim = CLAIMS["PA-2026-0107"]
    ctx = Orchestrator().run(claim)
    prep, info = ctx["prep"], ctx["llm_extraction"]
    neg_prep = preprocess_claim(CLAIMS["PA-2026-0111"])
    neg_sent = next(s for s in neg_prep.sentences if s["text"].startswith("No history"))
    negated = [e["text"] for e in neg_prep.entities if e["negated"] and e["sentence"] == neg_sent["n"]]
    sent = {s["n"]: s["text"] for s in prep.sentences}
    conds = [e for e in prep.entities if e["type"] == "CONDITION"]
    exp = prep.expansions
    raw = json.loads(info["attempts"][0]["raw"])
    raw_txt = json.dumps(raw, separators=(", ", ": "))
    ents_by_type = {}
    for e in prep.entities:
        ents_by_type.setdefault(e["type"], []).append(e["text"])

    cards = [
        # lane A
        ("P1", "Assemble", "Collect diagnosis, proposed treatment and the doctor's free-text notes as three sections.",
         f"Notes: “{short(claim['doctor_notes'], 150)}”", "nlp"),
        ("P2", "Normalise", "Unicode NFKC, straight quotes, dashes, number–unit spacing, collapse whitespace.",
         "“Amlodipine 5mg once daily” → “Amlodipine 5 mg once daily”", "nlp"),
        ("P3", "Expand abbreviations", f"Dictionary of 45 clinical short forms, case-aware word boundaries. Here: {sum(c for *_, c in exp)} expansions.",
         "K/c/o HTN → Known case of hypertension<br>T2DM x 4 yrs → type 2 diabetes mellitus for 4 years<br>CAG · LAD · PTCA · DES · Tab. → full words", "nlp"),
        ("P4", "De-identify (PII)", "Patient name, hospital, policy / claim no., phone, e-mail, ID numbers → placeholders. Map kept locally.",
         "Mrs Lata Kulkarni → Mrs [PATIENT]<br>Attendant contact 9000000001 → [PHONE_1]", "guard"),
        ("P5", "Segment sentences", "Split on sentence punctuation (protecting Dr., Mrs., e.g.) and number them S1…Sn.",
         f"S4: {esc(short(sent[4], 70))}<br>S5: {esc(short(sent[5], 70))}", "nlp"),
        # lane B
        ("P6", "Tag clinical entities", "Longest-match gazetteer: conditions, procedures, drugs, tests, flags (cosmetic, accident, declared).",
         f"CONDITION: {', '.join(ents_by_type.get('CONDITION', []))}<br>PROCEDURE: {short(', '.join(dict.fromkeys(ents_by_type.get('PROCEDURE', []))), 60)}<br>MEDICATION: {', '.join(ents_by_type.get('MEDICATION', []))}", "nlp"),
        ("P7", "Detect negation", "NegEx-style: trigger words (no, not, denies, without, no history of) negate entities in the next 8 tokens until 'but' / ';'.",
         f"PA-0111: “{esc(neg_sent['text'])}”<br>→ NEGATED: {', '.join(negated)}", "nlp"),
        ("P8", "Normalise durations", "'since YYYY' (vs request date), 'for / x N years|months', 'N years ago' → years; attached to the nearest condition.",
         "; ".join(f"{e['text']} → {e['duration_years']} y" for e in conds), "nlp"),
        ("P9", "Token budget", "≈4 characters per token (Llama BPE). If the notes exceed 1,200 tokens, low-value sentences are dropped first.",
         f"~{prep.tokens_est} tokens of {prep.budget} → nothing dropped", "nlp"),
        ("P10", "Build prompt", "System rules + allowed values + 1 worked example + numbered, de-identified sentences + request date + JSON schema.",
         "SYSTEM: 7 extraction rules (no guessing, ignore negated…)<br>USER: ### Allowed values · ### Example · ### Task<br>Request date: 2026-10-10 · Patient: [PATIENT], 64, F · S1…S8", "llm"),
        # lane C — inside the LLM runtime
        ("GATE", "PII gate", "Before the request leaves the app, every original identifier is searched for in the final prompt. Any hit blocks the call.",
         f"{info['pii_gate']}: “Lata Kulkarni”, “9000000001” not in prompt", "guard"),
        ("L1", "HTTP request to Ollama", "POST /api/generate with system + prompt + JSON schema in `format`; options: temperature 0, seed 42, num_ctx 4096, num_predict 512. Ollama wraps it in the Llama 3 chat template.",
         "&lt;|start_header_id|&gt;system&lt;|end_header_id|&gt; rules… &lt;|eot_id|&gt;<br>&lt;|start_header_id|&gt;user&lt;|end_header_id|&gt; ### Task …", "llm"),
        ("L2", "Tokenize", "Byte-pair-encoding tokenizer with a 128,256-entry vocabulary turns the text into token IDs; long medical words split into several sub-word tokens.",
         f"prompt → ~{info['attempts'][0]['prompt_tokens']} token IDs* (window 4,096 of the model's 128K)", "llm"),
        ("L3", "Transformer forward pass", "Token embeddings (3,072-d) → 28 decoder layers: grouped-query self-attention (24 heads, 8 KV heads), RoPE positions, SwiGLU MLP. 4-bit Q4_K_M weights (2.0 GB) on llama.cpp, CPU or GPU, KV cache reused per token.",
         "Meta Llama 3.2 3B Instruct · open weights", "llm"),
        ("L4", "Constrained decoding", "For each step the model scores all 128,256 tokens; a grammar built from the JSON schema masks every token that would break the schema; temperature 0 picks the top token (greedy). Repeat until the JSON closes.",
         'only valid next pieces allowed, e.g. after "procedure_key": → one of the 10 enum values', "llm"),
        ("L5", "Detokenize and respond", "Token IDs → text. Ollama returns the response plus prompt_eval_count, eval_count and timings, which the app logs.",
         f"{info['attempts'][0]['output_tokens']} output tokens* → JSON text", "llm"),
        # lane D — after the LLM
        ("P11", "Parse", "Strip ``` fences and chatter, pull out the first balanced JSON object, map synonyms ('high blood pressure' → hypertension).",
         esc(short(raw_txt, 170)), "guard"),
        ("P12", "Validate", "Check types, enums and required keys against the schema. Errors → repair prompt listing the errors (max 2) → else rule-based facts.",
         f"errors: {len(info['attempts'][0]['errors'])} · attempts: {len(info['attempts'])} · fallback: {'yes' if info['fallback'] else 'no'}", "guard"),
        ("P13", "Ground and cross-check", "Each condition must be stated, not negated, in the sentence it cites (else dropped as a hallucination). Then compare field-by-field with the rule extractor.",
         "; ".join(f"{g['condition']} S{g['cited']} → {g['status']}" for g in info["grounding"] if g["cited"]) +
         f"<br>LLM vs rules: {sum(r['agree'] for r in info['compare'])}/{len(info['compare'])} fields agree", "guard"),
        ("OUT", "Facts → shared state", "Validated facts go to the blackboard; disagreements become Critic findings; the Executor uses them for PED and exclusion rules.",
         f"procedure: {ctx['extracted']['procedure_key']} · " + ", ".join(f"{c['condition']} {c['duration_years']} y" for c in ctx["extracted"]["chronic_conditions"]) +
         f" · accident: {str(ctx['extracted']['accident']).lower()}", "out"),
        ("LTR", "Same path for the letter", "Decision agent: template letter → de-identify → PII gate → LLM polish (plain text) → re-identify → faithfulness check: identical amounts, clause numbers and placeholders, else the template is sent.",
         "same ₹ amounts ✓ · same clause nos. ✓ · same [PLACEHOLDERS] ✓ → LLM letter accepted", "out"),
    ]
    lanes = [("Prepare the text — runs locally, before any LLM", cards[0:5]),
             ("Understand the text and build the prompt", cards[5:10]),
             ("Inside the LLM runtime — Ollama → llama.cpp → Meta Llama 3.2 3B Instruct", cards[10:16]),
             ("Verify what comes back before any agent uses it", cards[16:21])]
    color = {"nlp": ("#EEF8F8", "#0E7C7B"), "llm": ("#F5EFFC", "#6B3FA0"), "guard": ("#FFF5EE", "#B54708"), "out": ("#E8F6EE", "#12804F")}

    lane_html = []
    for li, (title, cs) in enumerate(lanes):
        items = []
        for i, (code, name, what, ex, kind) in enumerate(cs):
            bg, fg = color[kind]
            arrow = '<div class="arr">→</div>' if i < len(cs) - 1 else ""
            items.append(f'<div class="card" style="background:{bg};border-color:{fg}"><div class="code" style="background:{fg}">{code}</div>'
                         f'<div class="name">{esc(name)}</div><div class="what">{esc(what)}</div><div class="ex">{ex}</div></div>{arrow}')
        lane_html.append(f'<div class="lane"><div class="lt">{li + 1} · {esc(title)}</div><div class="row">{"".join(items)}</div></div>'
                         + ('<div class="down">↓</div>' if li < len(lanes) - 1 else ""))
    return f"""<!doctype html><html><head><meta charset="utf-8"><style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{width:2400px;background:#fff;font-family:Inter,"Segoe UI",Helvetica,Arial,sans-serif;color:#101828;padding:28px 40px 30px}}
h1{{font-size:34px;font-weight:800;letter-spacing:-.5px}} .st{{font-size:17px;color:#475467;margin:8px 0 18px}}
.lane{{border:1.5px solid #EAECF0;border-radius:14px;padding:12px 14px 14px;background:#FCFCFD}}
.lt{{font-size:16px;font-weight:800;color:#344054;margin-bottom:10px}}
.row{{display:flex;align-items:stretch;gap:0}}
.card{{flex:1;border:2px solid;border-radius:12px;padding:11px 12px;position:relative;min-height:232px}}
.code{{display:inline-block;color:#fff;font-weight:800;font-size:13px;border-radius:6px;padding:2px 8px;margin-bottom:6px}}
.name{{font-size:17px;font-weight:800;margin-bottom:5px}}
.what{{font-size:14px;line-height:1.38;color:#344054;margin-bottom:8px}}
.ex{{font-family:"DejaVu Sans Mono",Consolas,monospace;font-size:12.8px;line-height:1.45;background:#fff;border:1px solid #D0D5DD;border-radius:8px;padding:7px 8px;color:#101828;word-break:break-word}}
.arr{{flex:none;width:30px;display:flex;align-items:center;justify-content:center;font-size:26px;color:#667085;font-weight:700}}
.down{{text-align:center;font-size:20px;font-weight:800;color:#667085;margin:2px 0;line-height:1}}
.foot{{margin-top:12px;font-size:13.5px;color:#475467}}
.legend span{{display:inline-block;margin-right:22px}} .legend i{{display:inline-block;width:14px;height:14px;border-radius:4px;margin-right:6px;vertical-align:-2px}}
</style></head><body>
<h1>How text is prepared for, sent to, and checked after the LLM</h1>
<div class="st">Pipeline run inside the Document Intake agent (tasks preprocess_text and extract_facts) · live example: request PA-2026-0107, angioplasty, notes full of abbreviations plus a name and phone number</div>
{''.join(lane_html)}
<div class="foot legend"><span><i style="background:#0E7C7B"></i>local NLP step</span><span><i style="background:#6B3FA0"></i>LLM step</span><span><i style="background:#B54708"></i>guardrail</span><span><i style="background:#12804F"></i>hand-off to other agents</span>
<span>* Token counts shown are from the simulated LLM run used to build this figure; with Ollama the real counts come from the model's response.</span></div>
</body></html>"""


def main(png: bool):
    D.mkdir(parents=True, exist_ok=True)
    (D / "llm_pipeline.html").write_text(llm_pipeline_html(), encoding="utf-8")
    ctx = Orchestrator().run(CLAIMS["PA-2026-0108"])
    n = message_count(ctx)
    half = 25
    (D / "sequence_full.svg").write_text(render_svg(ctx, title="End-to-end execution sequence · PA-2026-0108 (every call, in order)"), encoding="utf-8")
    (D / "sequence_part1.svg").write_text(render_svg(ctx, 0, half, title=f"Execution sequence · PA-2026-0108 · steps 1–{half}: plan, intake, LLM extraction, research"), encoding="utf-8")
    (D / "sequence_part2.svg").write_text(render_svg(ctx, half, n, title=f"Execution sequence · PA-2026-0108 · steps {half + 1}–{n}: eligibility, payable, Critic loop, decision, letter"), encoding="utf-8")
    print(f"sources written to {D} ({n} sequence messages)")
    if not png:
        return
    from playwright.sync_api import sync_playwright  # optional dependency
    import re
    jobs = [("system.html", "architecture_system.png"), ("llm_pipeline.html", "architecture_llm_pipeline.png"),
            ("sequence_full.svg", "architecture_sequence.png"), ("sequence_part1.svg", "sequence_part1.png"),
            ("sequence_part2.svg", "sequence_part2.png"), ("decision_logic.html", "architecture_decision_logic.png")]
    with sync_playwright() as p:
        try:
            b = p.chromium.launch(channel="msedge")  # use an installed Edge if there is one
        except Exception:  # noqa: BLE001
            b = p.chromium.launch()
        for src, out in jobs:
            path = D / src
            if not path.exists():
                continue
            if path.suffix == ".svg":
                svg = path.read_text(encoding="utf-8")
                svg = re.sub(r"<\?xml[^>]*>|<!DOCTYPE[^>]*>", "", svg)
                w = float(re.search(r'width="([\d.]+)(pt)?"', svg).group(1)) * (4 / 3 if 'pt"' in svg[:600] else 1)
                h = float(re.search(r'height="([\d.]+)(pt)?"', svg).group(1)) * (4 / 3 if 'pt"' in svg[:600] else 1)
                tmp = path.with_suffix(".wrap.html")
                tmp.write_text(f"<!doctype html><html><body style='margin:0;background:#fff'><div style='width:{w}px;height:{h}px'>{svg}</div></body></html>", encoding="utf-8")
                url, size = tmp.as_uri(), (int(w) + 1, int(h) + 1)
            else:
                url, size = path.as_uri(), None
            pg = b.new_page(viewport={"width": size[0] if size else 2400, "height": size[1] if size else 600}, device_scale_factor=1.6)
            pg.goto(url)
            pg.wait_for_timeout(600)
            if size is None:
                dims = pg.evaluate("() => [document.documentElement.scrollWidth, document.documentElement.scrollHeight]")
                pg.set_viewport_size({"width": dims[0], "height": dims[1]})
                pg.wait_for_timeout(300)
            target = (ROOT / "assets" / out) if out.startswith("architecture_") else (D / out)
            pg.screenshot(path=str(target))
            pg.close()
            print("rendered", target.relative_to(ROOT))
        b.close()


if __name__ == "__main__":
    main("--png" in sys.argv)
