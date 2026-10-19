"""Draws a UML-style sequence diagram (SVG) of one real run, built from the
orchestrator's audit trace: every task call, every tool / NLP / LLM /
guardrail call made inside it, every return and every Critic revision."""
from __future__ import annotations

from html import escape

LANES = ["Streamlit UI", "Orchestrator", "Planner", "Doc Intake", "NLP pipeline", "Guardrails", "LLM",
         "Policy Research", "Tools & data", "Coverage Executor", "Critic", "Decision & Letter", "Medical officer"]
KIND_LANE = {"nlp": "NLP pipeline", "llm": "LLM", "tool": "Tools & data", "guardrail": "Guardrails", "data": "Tools & data"}
COLORS = {"request": "#0B2E3C", "task": "#1E3A5F", "return": "#667085", "nlp": "#0E7C7B", "llm": "#6B3FA0",
          "tool": "#1F5FAE", "guardrail": "#B54708", "revise": "#D92D20", "escalate": "#6B3FA0", "human": "#8A5A00"}
LANE_FILL = {"Streamlit UI": "#F2F4F7", "Orchestrator": "#0B2E3C", "NLP pipeline": "#E6F4F4", "Guardrails": "#FEF0E6",
             "LLM": "#F2ECFA", "Tools & data": "#E7F0FB", "Critic": "#FFF4E5", "Decision & Letter": "#E8F6EE",
             "Medical officer": "#FFF4DC"}


def build_messages(ctx: dict, collapse_nlp: bool = False) -> list[dict]:
    msgs = [{"from": "Streamlit UI", "to": "Orchestrator", "label": f"run({ctx['claim']['claim_id']})", "kind": "request"}]
    for e in ctx["trace"]:
        lane = e["short"] if e["short"] in LANES else "Orchestrator"
        start = len(msgs)
        is_revision = "revision" in e["task"]
        msgs.append({"from": "Orchestrator", "to": lane, "label": e["task"], "kind": "revise" if is_revision else ("human" if lane == "Medical officer" else "task")})
        calls = e["calls"]
        if collapse_nlp and sum(c["kind"] == "nlp" for c in calls) > 1:
            nlp = [c for c in calls if c["kind"] == "nlp"]
            calls = [{"kind": "nlp", "name": f"{nlp[0]['name'].split()[0]}–{nlp[-1]['name'].split()[0]} preprocess ({len(nlp)} stages)", "detail": ""}] + \
                    [c for c in calls if c["kind"] != "nlp"]
        for c in calls:
            tgt = KIND_LANE.get(c["kind"], "Tools & data")
            label = c["name"] + (f" · {c['detail']}" if c.get("detail") else "")
            msgs.append({"from": lane, "to": tgt, "label": label, "kind": c["kind"]})
        ret = e["summary"]
        kind = "return"
        if e["task"] == "validate" and e["details"].get("verdict") == "REVISE":
            ret, kind = "REVISE → send back to Executor (cap at tariff ceiling)", "revise"
        msgs.append({"from": lane, "to": "Orchestrator", "label": ret, "kind": kind, "dashed": kind == "return",
                     "activation": (start, len(msgs))})
    if ctx.get("status", {}).get("human_review") == "awaiting human":
        msgs.append({"from": "Orchestrator", "to": "Medical officer", "label": "escalate: hold for human review", "kind": "escalate"})
    msgs.append({"from": "Orchestrator", "to": "Streamlit UI", "label": f"decision: {ctx['decision']['decision']} · letter · audit trace",
                 "kind": "request", "dashed": True})
    return msgs


def _w(text: str, size: float) -> float:
    return len(text) * size * 0.56


def render_svg(ctx: dict, start: int = 0, end: int | None = None, collapse_nlp: bool = False,
               title: str | None = None, row_h: int = 27, lane_w: int = 152, max_label: int = 64) -> str:
    msgs = build_messages(ctx, collapse_nlp)
    end = len(msgs) if end is None else end
    used = {m["from"] for m in msgs} | {m["to"] for m in msgs}
    lanes = [l for l in LANES if l in used]
    x = {l: 70 + i * lane_w + lane_w / 2 for i, l in enumerate(lanes)}
    top = 92 if title else 58
    rows = msgs[start:end]
    width = 70 + len(lanes) * lane_w + 20
    height = top + 40 + len(rows) * row_h + 50
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
         'font-family="Inter, Segoe UI, Helvetica, Arial, sans-serif">',
         f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
         '<defs>' + "".join(
             f'<marker id="m{k}" markerWidth="9" markerHeight="9" refX="8" refY="4.5" orient="auto"><path d="M0,0 L9,4.5 L0,9 z" fill="{c}"/></marker>'
             for k, c in COLORS.items()) + '</defs>']
    if title:
        o.append(f'<text x="20" y="34" font-size="21" font-weight="800" fill="#0B2E3C">{escape(title)}</text>')
    # lane headers + lifelines
    for l in lanes:
        fill = LANE_FILL.get(l, "#EEF4FF")
        fg = "#FFFFFF" if l == "Orchestrator" else "#101828"
        o.append(f'<rect x="{x[l] - lane_w / 2 + 6}" y="{top - 34}" width="{lane_w - 12}" height="30" rx="7" fill="{fill}" stroke="#98A2B3"/>')
        o.append(f'<text x="{x[l]}" y="{top - 14}" font-size="12.5" font-weight="700" text-anchor="middle" fill="{fg}">{escape(l)}</text>')
        o.append(f'<line x1="{x[l]}" y1="{top}" x2="{x[l]}" y2="{height - 40}" stroke="#D0D5DD" stroke-dasharray="4 4"/>')
    # activation bars
    for i, m in enumerate(msgs):
        if "activation" in m:
            a, b = m["activation"]
            a, b = max(a, start), min(b, end - 1)
            if a <= b and m["from"] in x:
                y1 = top + 26 + (a - start) * row_h - 6
                y2 = top + 26 + (b - start) * row_h + 4
                o.append(f'<rect x="{x[m["from"]] - 5}" y="{y1}" width="10" height="{y2 - y1}" fill="#EEF4FF" stroke="#98A2B3"/>')
    # messages
    for i, m in enumerate(rows):
        n = start + i + 1
        y = top + 26 + i * row_h
        x1, x2 = x[m["from"]], x[m["to"]]
        color = COLORS[m["kind"]]
        dash = ' stroke-dasharray="5 4"' if m.get("dashed") else ""
        sw = 2.2 if m["kind"] in ("task", "revise", "request", "escalate", "human") else 1.5
        direction = 1 if x2 > x1 else -1
        o.append(f'<line x1="{x1 + 6 * direction}" y1="{y}" x2="{x2 - 7 * direction}" y2="{y}" stroke="{color}" stroke-width="{sw}"{dash} marker-end="url(#m{m["kind"]})"/>')
        label = m["label"] if len(m["label"]) <= max_label else m["label"][:max_label - 1] + "…"
        size = 11.5
        tw = _w(label, size)
        lx = x1 + 12 if direction > 0 else x1 - 12 - tw
        lx = max(70, min(lx, width - tw - 10))
        o.append(f'<rect x="{lx - 3}" y="{y - 16}" width="{tw + 6}" height="14" fill="#ffffff" opacity="0.92"/>')
        weight = "700" if m["kind"] in ("task", "revise", "escalate", "human") else "400"
        o.append(f'<text x="{lx}" y="{y - 5}" font-size="{size}" font-weight="{weight}" fill="{color}">{escape(label)}</text>')
        o.append(f'<text x="22" y="{y + 4}" font-size="11" font-weight="700" fill="#98A2B3">{n}</text>')
    # legend
    ly = height - 18
    lx = 20
    for k, name in [("task", "task call"), ("return", "result"), ("nlp", "NLP stage"), ("llm", "LLM call"),
                    ("tool", "tool / data"), ("guardrail", "guardrail"), ("revise", "revision loop")]:
        dash = ' stroke-dasharray="5 4"' if k == "return" else ""
        o.append(f'<line x1="{lx}" y1="{ly - 4}" x2="{lx + 26}" y2="{ly - 4}" stroke="{COLORS[k]}" stroke-width="2"{dash}/>')
        o.append(f'<text x="{lx + 32}" y="{ly}" font-size="11.5" fill="#475467">{name}</text>')
        lx += 40 + _w(name, 11.5) + 18
    o.append("</svg>")
    return "\n".join(o)


def message_count(ctx: dict, collapse_nlp: bool = False) -> int:
    return len(build_messages(ctx, collapse_nlp))
