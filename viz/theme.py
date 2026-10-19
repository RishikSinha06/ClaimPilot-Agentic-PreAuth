"""Look and feel of the dashboard: one stylesheet plus small HTML building blocks.

Every block works in both light and dark mode. Surfaces are translucent tints of
the current text colour (`currentColor`) or of a tone colour, so nothing depends
on knowing which theme is active. All request text is escaped with `h()`.
"""
from __future__ import annotations

import html

# Status / decision tones (fixed, never reused as chart series colours). Each is
# always shown with an icon and a label, never colour alone.
TONE = {"good": "#12a150", "warn": "#c47f00", "bad": "#d03b3b", "info": "#2a78d6", "violet": "#7a5af8",
        "neutral": "#8a8f98"}
DECISION_TONE = {"APPROVE": "good", "PARTIAL": "warn", "QUERY": "info", "DENY": "bad", "ESCALATE": "violet"}
DECISION_GLYPH = {"APPROVE": "✓", "PARTIAL": "◐", "QUERY": "?", "DENY": "✕", "ESCALATE": "⚑"}
STATUS_TONE = {"PASS": "good", "WARN": "warn", "FAIL": "bad"}
STATUS_GLYPH = {"PASS": "✓", "WARN": "!", "FAIL": "✕"}

CSS = """
<style>
:root{--pa-good:#12a150;--pa-warn:#c47f00;--pa-bad:#d03b3b;--pa-info:#2a78d6;--pa-violet:#7a5af8;--pa-neutral:#8a8f98;
      --pa-line:rgba(127,127,127,.22);--pa-grad:linear-gradient(120deg,#0b2a55 0%,#1a4f96 50%,#0d7f7a 100%);}
.block-container{padding-top:1.4rem;padding-bottom:3rem;max-width:1440px}
[data-testid="stMetric"]{padding:14px 16px}
[data-testid="stMetricValue"]{font-weight:800;letter-spacing:-.02em}
[data-testid="stMetricLabel"] p{font-weight:600;opacity:.75}
button[data-baseweb="tab"] p{font-weight:600}

/* hero */
.pa-hero{position:relative;overflow:hidden;border-radius:22px;padding:28px 32px 24px;margin:0 0 18px;color:#fff;
  background:radial-gradient(900px 260px at 95% -10%,rgba(56,189,248,.35),transparent 60%),var(--pa-grad);
  box-shadow:0 14px 34px rgba(11,42,85,.28)}
.pa-hero::after{content:"";position:absolute;right:-70px;top:-90px;width:280px;height:280px;border-radius:50%;
  background:rgba(255,255,255,.07)}
.pa-hero::before{content:"";position:absolute;right:120px;bottom:-120px;width:220px;height:220px;border-radius:50%;
  background:rgba(255,255,255,.05)}
.pa-hero-top{display:flex;align-items:center;gap:16px;position:relative;z-index:1}
.pa-logo{width:56px;height:56px;border-radius:16px;display:grid;place-items:center;font-size:30px;
  background:rgba(255,255,255,.14);border:1px solid rgba(255,255,255,.25)}
.pa-hero-title{font-size:2.05rem;font-weight:800;letter-spacing:-.025em;line-height:1.1;color:#fff}
.pa-hero-sub{opacity:.88;margin-top:5px;font-size:1rem;max-width:880px;color:#fff}
.pa-hero-badges{display:flex;flex-wrap:wrap;gap:8px;margin-top:18px;position:relative;z-index:1}
.pa-hbadge{display:inline-flex;align-items:center;gap:7px;padding:5px 12px;border-radius:999px;font-size:.8rem;font-weight:600;
  background:rgba(255,255,255,.12);border:1px solid rgba(255,255,255,.22);color:#fff}
.pa-hbadge .dot{width:8px;height:8px;border-radius:50%;background:#4ade80;box-shadow:0 0 0 3px rgba(74,222,128,.25)}
.pa-hbadge .dot.sim{background:#facc15;box-shadow:0 0 0 3px rgba(250,204,21,.25)}
.pa-hbadge .dot.fault{background:#f87171;box-shadow:0 0 0 3px rgba(248,113,113,.3)}

/* section header */
.pa-h{display:flex;align-items:baseline;gap:10px;margin:20px 0 10px;flex-wrap:wrap}
.pa-h .t{font-size:1.15rem;font-weight:800;letter-spacing:-.01em}
.pa-h .s{opacity:.6;font-size:.88rem}

/* cards */
.pa-card{border:1px solid var(--pa-line);border-radius:18px;padding:18px 20px;margin-bottom:12px;
  background:color-mix(in srgb,currentColor 3%,transparent)}
.pa-kicker{font-size:.7rem;font-weight:800;letter-spacing:.09em;text-transform:uppercase;opacity:.55;margin-bottom:10px}
.pa-empty{border:1.5px dashed rgba(127,127,127,.35);border-radius:18px;padding:30px;text-align:center}
.pa-empty .ic{font-size:2.1rem}.pa-empty .t{font-weight:700;font-size:1.05rem;margin-top:6px}.pa-empty .s{opacity:.65;margin-top:4px}

/* chips */
.pa-chip{--tone:var(--pa-neutral);display:inline-flex;align-items:center;gap:6px;padding:3px 10px;border-radius:999px;
  font-size:.75rem;font-weight:650;line-height:1.45;margin:0 6px 6px 0;white-space:nowrap;
  background:color-mix(in srgb,var(--tone) 14%,transparent);border:1px solid color-mix(in srgb,var(--tone) 42%,transparent)}
.pa-chip i{width:7px;height:7px;border-radius:50%;background:var(--tone);display:inline-block}

/* patient card */
.pa-patient{display:flex;align-items:center;gap:14px;margin-bottom:12px}
.pa-avatar{width:50px;height:50px;border-radius:50%;display:grid;place-items:center;font-weight:800;color:#fff;
  background:var(--pa-grad);font-size:1.05rem;flex:0 0 50px}
.pa-name{font-size:1.3rem;font-weight:800;line-height:1.15;letter-spacing:-.01em}
.pa-sub{opacity:.62;font-size:.86rem;margin-top:2px}
.pa-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px 20px;margin:14px 0}
.pa-field .k{font-size:.68rem;text-transform:uppercase;letter-spacing:.07em;opacity:.55;font-weight:800}
.pa-field .v{font-weight:600;margin-top:2px}
.pa-field code{font-size:.8rem}
.pa-notes{border-left:3px solid var(--pa-info);padding:10px 14px;border-radius:0 12px 12px 0;font-size:.93rem;line-height:1.6;
  background:color-mix(in srgb,var(--pa-info) 8%,transparent);margin:6px 0 12px}

/* cost breakdown */
.pa-cost-row{display:grid;grid-template-columns:1fr auto;gap:5px 12px;margin:11px 0}
.pa-cost-row .lbl{font-weight:600;font-size:.92rem}.pa-cost-row .basis{opacity:.55;font-size:.78rem;font-weight:500;margin-left:6px}
.pa-cost-row .amt{font-variant-numeric:tabular-nums;font-weight:700;text-align:right;font-size:.92rem}
.pa-bar{grid-column:1/-1;height:6px;border-radius:99px;background:rgba(127,127,127,.16);overflow:hidden}
.pa-bar>span{display:block;height:100%;border-radius:99px;background:var(--pa-info)}
.pa-total{display:flex;justify-content:space-between;align-items:baseline;border-top:1px solid var(--pa-line);margin-top:16px;padding-top:12px}
.pa-total .k{font-weight:700;opacity:.7}.pa-total .v{font-size:2rem;font-weight:800;letter-spacing:-.03em}

/* agent flow */
.pa-flow{display:grid;grid-template-columns:repeat(7,minmax(0,1fr));gap:14px;margin:4px 0 18px}
.pa-node{--tone:var(--pa-neutral);position:relative;border-radius:14px;padding:10px 6px 9px;text-align:center;
  border:1px solid color-mix(in srgb,var(--tone) 32%,transparent);background:color-mix(in srgb,var(--tone) 8%,transparent)}
.pa-node:not(:last-child)::after{content:"›";position:absolute;right:-11px;top:50%;transform:translateY(-50%);opacity:.4;font-weight:800}
@media (max-width:1150px){.pa-flow{grid-template-columns:repeat(4,minmax(0,1fr))}.pa-node::after{display:none}}
@media (max-width:640px){.pa-flow{grid-template-columns:repeat(2,minmax(0,1fr))}}
.pa-node .ic{font-size:1.35rem;line-height:1.2}.pa-node .nm{font-weight:750;font-size:.84rem;margin-top:3px}
.pa-node .st{font-size:.7rem;opacity:.7;margin-top:2px}
.pa-node.done{--tone:var(--pa-info)}.pa-node.skip{opacity:.5}
.pa-node.wait{--tone:var(--pa-warn);animation:pa-pulse 1.8s ease-in-out infinite}
.pa-arrow{align-self:center;opacity:.35;font-weight:800}
@keyframes pa-pulse{0%,100%{box-shadow:0 0 0 0 color-mix(in srgb,var(--pa-warn) 50%,transparent)}
  60%{box-shadow:0 0 0 7px transparent}}

/* decision */
.pa-decision{--tone:var(--pa-info);display:flex;align-items:center;gap:20px;flex-wrap:wrap;border-radius:20px;padding:20px 24px;
  margin:4px 0 14px;border:1px solid color-mix(in srgb,var(--tone) 45%,transparent);
  background:linear-gradient(100deg,color-mix(in srgb,var(--tone) 18%,transparent),color-mix(in srgb,var(--tone) 4%,transparent))}
.pa-dec-icon{width:60px;height:60px;border-radius:18px;display:grid;place-items:center;font-size:30px;font-weight:800;color:#fff;
  background:var(--tone);box-shadow:0 8px 20px color-mix(in srgb,var(--tone) 40%,transparent);flex:0 0 60px}
.pa-dec-main{flex:1 1 300px}
.pa-dec-kicker{font-size:.7rem;font-weight:800;letter-spacing:.09em;text-transform:uppercase;opacity:.6}
.pa-dec-label{font-size:1.75rem;font-weight:800;letter-spacing:-.02em;line-height:1.15;margin-top:2px}
.pa-dec-sub{opacity:.75;margin-top:3px}
.pa-dec-amt{text-align:right;flex:0 1 auto}
.pa-dec-amt .v{font-size:2.5rem;font-weight:800;letter-spacing:-.03em;line-height:1}
.pa-dec-amt .s{opacity:.65;font-size:.85rem;margin-top:6px}
.pa-meter{height:7px;border-radius:99px;background:rgba(127,127,127,.18);overflow:hidden;margin:8px 0 0 auto;width:230px;max-width:100%}
.pa-meter>span{display:block;height:100%;background:var(--tone);border-radius:99px}

/* live timeline */
.pa-tl{display:flex;gap:12px;padding:7px 0;position:relative}
.pa-tl:not(:last-child)::before{content:"";position:absolute;left:17px;top:44px;bottom:-8px;width:2px;background:var(--pa-line)}
.pa-tl-dot{flex:0 0 36px;height:36px;border-radius:12px;display:grid;place-items:center;font-size:1.05rem;
  background:color-mix(in srgb,var(--pa-info) 13%,transparent);border:1px solid color-mix(in srgb,var(--pa-info) 35%,transparent)}
.pa-tl-head{font-weight:700;font-size:.9rem}.pa-tl-head code{font-size:.76rem;margin-left:4px}
.pa-tl-ms{opacity:.5;font-weight:500;font-size:.76rem;margin-left:6px}
.pa-tl-txt{opacity:.78;font-size:.86rem;margin-top:1px;line-height:1.45}

/* check / finding / task lists */
.pa-list{display:flex;flex-direction:column;gap:8px;margin:4px 0 8px}
.pa-item{--tone:var(--pa-neutral);display:grid;grid-template-columns:32px 1fr auto;gap:12px;align-items:center;padding:11px 14px;
  border-radius:14px;border:1px solid var(--pa-line);border-left:4px solid var(--tone);
  background:color-mix(in srgb,currentColor 2.5%,transparent)}
.pa-item .ic{width:30px;height:30px;border-radius:50%;display:grid;place-items:center;font-weight:800;font-size:.88rem;color:#fff;background:var(--tone)}
.pa-item .t{font-weight:700;font-size:.93rem}.pa-item .d{opacity:.75;font-size:.86rem;margin-top:2px;line-height:1.45}
.pa-item .r{display:flex;flex-wrap:wrap;justify-content:flex-end;max-width:260px}
.pa-item .r .pa-chip{margin:2px 0 2px 6px}

/* letter */
.pa-paper{background:#fff;color:#1d2433;border-radius:16px;overflow:hidden;max-width:880px;margin:6px 0 12px;
  border:1px solid rgba(16,24,40,.1);box-shadow:0 1px 2px rgba(16,24,40,.06),0 14px 34px rgba(16,24,40,.14)}
.pa-paper-head{display:flex;justify-content:space-between;align-items:center;gap:10px;padding:14px 30px;background:var(--pa-grad);color:#fff;font-weight:700}
.pa-paper-head span{opacity:.8;font-weight:500;font-size:.82rem}
.pa-paper-body{padding:24px 32px 30px;line-height:1.7;font-size:.95rem;color:#1d2433}

/* stage stepper (LLM pipeline) */
.pa-steps{display:flex;flex-wrap:wrap;gap:6px;margin:4px 0 14px}
.pa-step{padding:5px 10px;border-radius:10px;font-size:.76rem;font-weight:700;border:1px solid var(--pa-line);
  background:color-mix(in srgb,currentColor 3%,transparent)}
.pa-step.llm{background:color-mix(in srgb,var(--pa-violet) 16%,transparent);border-color:color-mix(in srgb,var(--pa-violet) 45%,transparent)}
.pa-step.gate{background:color-mix(in srgb,var(--pa-warn) 16%,transparent);border-color:color-mix(in srgb,var(--pa-warn) 45%,transparent)}

/* clinical entity highlights: pastel fills always carry dark text */
.sent{font-size:.95rem;line-height:2.05;margin:2px 0}
.sn{display:inline-block;min-width:34px;font-weight:800;opacity:.5}
.ent{padding:1px 6px;border-radius:6px;border:1px solid rgba(0,0,0,.08);color:#101828}
.ent small{font-size:.64rem;font-weight:800;color:#475467;margin-left:4px}
.neg{text-decoration:line-through;opacity:.75}

.seqwrap{overflow-x:auto;border:1px solid var(--pa-line);border-radius:16px;padding:8px;background:#fff}

/* sidebar */
.pa-side-brand{display:flex;gap:11px;align-items:center;padding:2px 0 6px}
.pa-side-brand .lg{width:40px;height:40px;border-radius:12px;display:grid;place-items:center;background:var(--pa-grad);font-size:21px}
.pa-side-brand .nm{font-weight:800;font-size:1.08rem;letter-spacing:-.01em}.pa-side-brand .sb{font-size:.74rem;opacity:.6}
.pa-side-label{font-size:.68rem;font-weight:800;letter-spacing:.1em;text-transform:uppercase;opacity:.55;margin:16px 0 4px}
</style>
"""


def h(text) -> str:
    """Escape any text placed inside our HTML."""
    return html.escape(str(text))


def tone(name: str) -> str:
    return f"var(--pa-{name})"


def chip(text, tone_name: str = "neutral") -> str:
    return f'<span class="pa-chip" style="--tone:{tone(tone_name)}"><i></i>{h(text)}</span>'


def section(title: str, subtitle: str = "") -> str:
    return f'<div class="pa-h"><span class="t">{h(title)}</span><span class="s">{h(subtitle)}</span></div>'


def empty_state(icon: str, title: str, subtitle: str) -> str:
    return f'<div class="pa-empty"><div class="ic">{icon}</div><div class="t">{h(title)}</div><div class="s">{h(subtitle)}</div></div>'


def hero(engine: str, simulated: bool, fault: str | None) -> str:
    dot = "fault" if fault else ("sim" if simulated else "")
    engine_txt = f"LLM: {engine}" + (f" · fault: {fault}" if fault else "")
    badges = [f'<span class="pa-hbadge"><span class="dot {dot}"></span>{h(engine_txt)}</span>'] + [
        f'<span class="pa-hbadge">{h(b)}</span>' for b in
        ("🤖 6 cooperating agents", "🔒 PII never leaves the machine", "🧮 Money & rules in deterministic code",
         "🩺 Human in the loop", "🧪 Synthetic data")]
    return (f'<div class="pa-hero"><div class="pa-hero-top"><div class="pa-logo">🏥</div><div>'
            f'<div class="pa-hero-title">ClaimPilot</div>'
            f'<div class="pa-hero-sub">Agentic AI for cashless health-insurance pre-authorisation: a hospital request '
            f'becomes a validated, explained decision and a letter, with a medical officer for risky cases.</div>'
            f'</div></div><div class="pa-hero-badges">{"".join(badges)}</div></div>')


def sidebar_brand() -> str:
    return ('<div class="pa-side-brand"><div class="lg">🏥</div><div><div class="nm">ClaimPilot</div>'
            '<div class="sb">CA-3 · Agentic AI PoC</div></div></div>')


def side_label(text: str) -> str:
    return f'<div class="pa-side-label">{h(text)}</div>'


# ------------------------------------------------------------------ agent flow
FLOW = [("planner", "🧭", "Planner", ["plan"]),
        ("intake", "📄", "Doc Intake", ["check_documents", "preprocess_text", "extract_facts", "verify_accident_docs"]),
        ("research", "🔎", "Policy Research", ["research_policy"]),
        ("executor", "⚙️", "Executor", ["check_eligibility", "calculate_payable"]),
        ("critic", "🧐", "Critic", ["validate"]),
        ("decision", "✅", "Decision & Letter", ["decide", "draft_letter"]),
        ("human", "🩺", "Medical officer", ["human_review"])]


def agent_flow(ctx: dict | None = None) -> str:
    """The six agents and the human as a row of cards, coloured by what happened in the last run."""
    nodes = []
    for key, icon, name, tasks in FLOW:
        if ctx is None:
            cls, state = "idle", "ready"
        else:
            st = [("done" if t == "plan" else ctx["status"].get(t, "")) for t in tasks]
            if "awaiting human" in st:
                cls, state = "wait", "waiting for you"
            elif "done" in st:
                cls = "done"
                state = f"{st.count('done')} task{'s' if st.count('done') != 1 else ''}"
                if key == "critic" and ctx.get("revisions"):
                    state += f" · ↺ {ctx['revisions']} revision"
            else:
                cls, state = "skip", "not needed"
        nodes.append(f'<div class="pa-node {cls}"><div class="ic">{icon}</div><div class="nm">{h(name)}</div>'
                     f'<div class="st">{h(state)}</div></div>')
    return '<div class="pa-flow">' + "".join(nodes) + "</div>"


# ------------------------------------------------------------------ request
def patient_card(claim: dict, product: str, doc_labels: dict, required: list[str]) -> str:
    initials = "".join(p[0] for p in str(claim["patient"]).split()[:2]).upper() or "?"
    chips = (chip(claim["admission_type"], "bad" if claim["admission_type"] == "emergency" else "info")
             + chip(claim["treatment_type"], "violet") + chip(f"{claim['policy_no']} · {product}", "neutral")
             + (chip("re-submission", "warn") if claim.get("resubmission_of") else ""))
    fields = [("Hospital", h(claim["hospital"])),
              ("Diagnosis", f'{h(claim["diagnosis"])} <code>{h(claim["icd10"])}</code>'),
              ("Proposed treatment", h(claim["proposed_treatment"])), ("Requested on", h(claim["request_date"]))]
    grid = "".join(f'<div class="pa-field"><div class="k">{k}</div><div class="v">{v}</div></div>' for k, v in fields)
    docs = claim["documents"]
    doc_chips = "".join(chip(("✓ " if d in docs else "✕ ") + doc_labels.get(d, d), "good" if d in docs else "bad")
                        for d in required)
    doc_chips += "".join(chip("✓ " + doc_labels.get(d, d), "good") for d in docs if d not in required)
    return (f'<div class="pa-card"><div class="pa-kicker">Pre-authorisation request</div>'
            f'<div class="pa-patient"><div class="pa-avatar">{h(initials)}</div><div>'
            f'<div class="pa-name">{h(claim["patient"])}</div>'
            f'<div class="pa-sub">{h(claim["age"])}{h(claim["gender"])} · {h(claim["claim_id"])}</div></div></div>'
            f'<div>{chips}</div><div class="pa-grid">{grid}</div>'
            f'<div class="pa-kicker">Doctor\'s notes · raw, as written by the hospital</div>'
            f'<div class="pa-notes">{h(claim["doctor_notes"])}</div>'
            f'<div class="pa-kicker">Documents</div><div>{doc_chips}</div></div>')


def cost_card(rows: list[tuple[str, str, int]], total: int, inr) -> str:
    top = max([a for *_, a in rows] + [1])
    body = "".join(
        f'<div class="pa-cost-row"><div class="lbl">{h(label)}<span class="basis">{h(basis)}</span></div>'
        f'<div class="amt">{h(inr(amount))}</div><div class="pa-bar"><span style="width:{100 * amount / top:.1f}%"></span></div></div>'
        for label, basis, amount in rows)
    return (f'<div class="pa-card"><div class="pa-kicker">Hospital cost estimate</div>{body}'
            f'<div class="pa-total"><span class="k">Total requested</span><span class="v">{h(inr(total))}</span></div></div>')


# ------------------------------------------------------------------ run + result
def timeline(events: list[dict]) -> str:
    return "".join(
        f'<div class="pa-tl"><div class="pa-tl-dot">{h(e["icon"])}</div><div>'
        f'<div class="pa-tl-head">{h(e["agent"])}<code>{h(e["task"])}</code><span class="pa-tl-ms">{e["ms"]:.1f} ms</span></div>'
        f'<div class="pa-tl-txt">{h(e["summary"])}</div></div></div>' for e in events)


def decision_card(dec: dict, inr) -> str:
    d = dec["decision"]
    t = tone(DECISION_TONE[d])
    by = dec.get("decided_by", "agents")
    who = "the agents" if by == "agents" else by.replace("human: ", "")
    paid = d in ("APPROVE", "PARTIAL")
    if paid:
        amount = f'<div class="v">{h(inr(dec["amount"]))}</div><div class="s">authorised of {h(inr(dec["requested"]))} requested</div>'
    elif d == "ESCALATE" and dec["amount"]:
        amount = (f'<div class="v" style="opacity:.55">{h(inr(dec["amount"]))}</div>'
                  f'<div class="s">provisional, pending review · {h(inr(dec["requested"]))} requested</div>')
    else:
        amount = f'<div class="v" style="opacity:.45">₹0</div><div class="s">nothing authorised · {h(inr(dec["requested"]))} requested</div>'
    conf = dec.get("confidence", 0)
    return (f'<div class="pa-decision" style="--tone:{t}"><div class="pa-dec-icon">{DECISION_GLYPH[d]}</div>'
            f'<div class="pa-dec-main"><div class="pa-dec-kicker">Decision · {h(d)}</div>'
            f'<div class="pa-dec-label">{h(dec["label"])}</div>'
            f'<div class="pa-dec-sub">{h(dec["subtitle"])} · decided by {h(who)}</div></div>'
            f'<div class="pa-dec-amt">{amount}<div class="pa-meter" title="Critic confidence">'
            f'<span style="width:{conf * 100:.0f}%"></span></div>'
            f'<div class="s">Critic confidence {conf:.0%}</div></div></div>')


def item(glyph: str, tone_name: str, title: str, detail: str = "", right: str = "") -> str:
    text_glyph_color = ' style="color:#2a1d00"' if tone_name == "warn" else ""
    return (f'<div class="pa-item" style="--tone:{tone(tone_name)}"><div class="ic"{text_glyph_color}>{glyph}</div>'
            f'<div><div class="t">{h(title)}</div>' + (f'<div class="d">{h(detail)}</div>' if detail else "")
            + f'</div><div class="r">{right}</div></div>')


def check_list(checks: list[dict]) -> str:
    rows = [item(STATUS_GLYPH[c["result"]], STATUS_TONE[c["result"]], c["check"], c["detail"],
                 chip(c["result"], STATUS_TONE[c["result"]]) + (chip(f"clause {c['clause']}", "neutral") if c.get("clause") else ""))
            for c in checks]
    return '<div class="pa-list">' + "".join(rows) + "</div>"


def findings_list(findings: list[dict]) -> str:
    action_tone = {"revise": "bad", "query": "info", "escalate": "violet"}
    rows = []
    for f in findings:
        right = chip(f["status"], STATUS_TONE[f["status"]])
        if f["status"] != "PASS" and f["severity"] != "info":
            right += chip(f"{f['severity']} severity", "neutral")
        if f.get("action"):
            right += chip(f"→ {f['action']}", action_tone.get(f["action"], "neutral"))
        rows.append(item(STATUS_GLYPH[f["status"]], STATUS_TONE[f["status"]], f["check"], f["detail"], right))
    return '<div class="pa-list">' + "".join(rows) + "</div>"


def task_list(tasks: list[dict], status: dict, revisions: int) -> str:
    look = {"done": ("✓", "good", "done"), "skipped": ("–", "neutral", "skipped"), "not needed": ("–", "neutral", "not needed"),
            "awaiting human": ("⏳", "warn", "waiting for human")}
    rows = []
    for i, t in enumerate(tasks, start=1):
        glyph, tn, label = look.get(status.get(t["id"], ""), (str(i), "neutral", "—"))
        if t["id"] == "validate" and revisions:
            label += f" · ↺ {revisions}"
        right = chip(label, tn) + (chip(f"when {t['when']}", "info") if t["when"] != "always" else "")
        rows.append(item(glyph, tn, f"{t['id']}  ·  {t['agent']}", t["goal"], right))
    return '<div class="pa-list">' + "".join(rows) + "</div>"


def letter(text: str, method: str) -> str:
    return (f'<div class="pa-paper"><div class="pa-paper-head">🏥 Pre-Authorisation Team<span>{h(method)}</span></div>'
            f'<div class="pa-paper-body">{h(text).replace(chr(10), "<br>")}</div></div>')  # blank lines would end the HTML block


def stage_stepper(stage_log: list[dict]) -> str:
    steps = [f'<span class="pa-step" title="{h(s["summary"])}">{h(s["stage"])} {h(s["name"])}</span>' for s in stage_log]
    steps += ['<span class="pa-step">P10 Prompt</span>', '<span class="pa-step gate">PII gate</span>',
              '<span class="pa-step llm">LLM</span>', '<span class="pa-step">P11 Parse</span>',
              '<span class="pa-step">P12 Validate</span>', '<span class="pa-step">P13 Ground + compare</span>']
    return '<div class="pa-steps">' + '<span class="pa-arrow">›</span>'.join(steps) + "</div>"
