"""Clinical text preprocessing pipeline (stages P1-P9).

Every piece of free text is run through these stages BEFORE anything is sent
to the LLM. The same output also feeds the rule-based extractor, so the two
extractors see identical, cleaned text and can be cross-checked.

P1  Assemble      diagnosis + proposed treatment + doctor's notes, kept as sections
P2  Normalise     Unicode NFKC, quotes/dashes, number-unit spacing, whitespace
P3  Expand        medical abbreviations (k/c/o, HTN, T2DM, RTA, ...) -> full words
P4  De-identify   patient name, hospital, policy/claim no., phone, email, ID -> placeholders
P5  Segment       split into numbered sentences (S1, S2, ...)
P6  Tag entities  dictionary (gazetteer) match: conditions, procedures, drugs, tests, flags
P7  Negation      NegEx-style scope: "no history of diabetes" -> diabetes is NEGATED
P8  Durations     "since 2018", "for 4 years", "x 3 months" -> duration in years
P9  Budget        estimate tokens and trim low-value sentences to fit the context window
"""
from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import asdict, dataclass, field

from ..trace import record

# ---------------------------------------------------------------- P3 data
# (abbreviation, expansion, case_sensitive). Longer / more specific entries first.
ABBREVIATIONS: list[tuple[str, str, bool]] = [
    ("k/c/o", "known case of", False), ("c/o", "complaints of", False), ("h/o", "history of", False),
    ("s/p", "status post", False), ("o/e", "on examination", False),
    ("T2DM", "type 2 diabetes mellitus", True), ("DM", "diabetes mellitus", True),
    ("HTN", "hypertension", True), ("CAD", "coronary artery disease", True),
    ("CAG", "coronary angiography", True), ("CKD", "chronic kidney disease", True),
    ("COPD", "chronic obstructive pulmonary disease", True), ("OA", "osteoarthritis", True),
    ("RTA", "road traffic accident", True), ("MLC", "medico-legal case", True),
    ("USG", "ultrasonography", True), ("CXR", "chest X-ray", True), ("CBC", "complete blood count", True),
    ("ECG", "electrocardiogram", True), ("RUQ", "right upper quadrant", True), ("RIF", "right iliac fossa", True),
    ("SOB", "shortness of breath", True), ("TKR", "total knee replacement", True),
    ("PTCA", "percutaneous transluminal coronary angioplasty", True), ("DES", "drug-eluting stent", True),
    ("LAD", "left anterior descending artery", True), ("IOL", "intraocular lens", True),
    ("ORIF", "open reduction internal fixation", True), ("TEP", "totally extraperitoneal", True),
    ("GA", "general anaesthesia", True), ("IV", "intravenous", True),
    ("tab.", "tablet", False), ("tab", "tablet", False), ("inj.", "injection", False),
    ("yrs", "years", False), ("yr", "year", False), ("mths", "months", False), ("mos", "months", False),
    ("wks", "weeks", False), ("pt", "patient", False), ("lap", "laparoscopic", False),
]

# ---------------------------------------------------------------- P6 gazetteer
CONDITIONS = {
    "hypertension": ["hypertension", "hypertensive", "high blood pressure"],
    "diabetes": ["type 2 diabetes mellitus", "diabetes mellitus", "diabetes", "diabetic"],
    "osteoarthritis": ["osteoarthritis"],
    "asthma": ["asthma", "asthmatic"],
    "thyroid disorder": ["hypothyroidism", "hyperthyroidism", "hypothyroid", "thyroid disorder"],
    "chronic kidney disease": ["chronic kidney disease"],
    "copd": ["chronic obstructive pulmonary disease"],
}
PROCEDURES = {
    "appendectomy": ["appendectomy", "appendicectomy", "appendicitis", "appendix"],
    "cataract": ["phacoemulsification", "cataract", "intraocular lens"],
    "joint replacement": ["total knee replacement", "knee replacement", "hip replacement", "joint replacement", "arthroplasty"],
    "hernia": ["hernia repair", "hernioplasty", "hernia"],
    "rhinoplasty": ["rhinoplasty"],
    "angioplasty": ["percutaneous transluminal coronary angioplasty", "angioplasty", "drug-eluting stent", "stent"],
    "gallstone": ["cholecystectomy", "cholelithiasis", "gallstone"],
    "fracture fixation": ["open reduction internal fixation", "intramedullary nail", "fracture"],
    "dental surgery": ["third molar", "odontectomy", "tooth extraction", "dental"],
}
MEDICATIONS = ["amlodipine", "metformin", "insulin", "telmisartan", "levothyroxine", "inhaler", "antibiotics",
               "intravenous fluids", "intravenous antibiotics", "paracetamol"]
INVESTIGATIONS = ["ultrasonography", "chest x-ray", "x-ray", "coronary angiography", "angiography",
                  "complete blood count", "platelet count", "ns1", "electrocardiogram", "visual acuity"]
FLAGS = {
    "cosmetic": ["aesthetic", "cosmetic", "appearance"],
    "accident": ["road traffic accident", "accident", "injury", "trauma", "fall from"],
    "functional": ["nasal obstruction", "breathing difficulty", "obstruction", "reconstruction", "functional"],
    "declared": ["declared at proposal", "declared"],
}
ALL_PROCEDURE_KEYS = list(PROCEDURES) + ["medical management"]
ALL_CONDITION_KEYS = list(CONDITIONS)

# ---------------------------------------------------------------- P7 NegEx-lite
PRE_NEGATION = ["no history of", "no evidence of", "negative for", "absence of", "free of",
                "denies", "denied", "without", "not", "nil", "no"]
POST_NEGATION = ["ruled out", "absent", "not seen"]
TERMINATORS = {"but", "however", "although", "except", "apart", "aside", ";", ":"}
UNCERTAINTY = ["suspected", "possible", "probable", "likely", "rule out", "?"]
NEG_WINDOW = 8  # tokens after a trigger that can be negated

CONTEXT_TOKEN_BUDGET = 1200  # tokens reserved for clinical text inside the prompt

# ---------------------------------------------------------------- P4 patterns
# Contact details and IDs (masked as [LABEL_n]); order matters: e-mails can contain digits.
CONTACT_PATTERNS = [
    (r"[\w.+-]+@[\w-]+\.[\w.]+", "EMAIL"),
    (r"(?<!\d)(?:\+91[\s-]?)?[6-9]\d{4}[\s-]?\d{5}(?!\d)", "PHONE"),   # mobile
    (r"(?<![\d-])0\d{2,4}[\s-]\d{6,8}(?!\d)", "PHONE"),                 # landline with STD code
    (r"(?<!\d)\d{4}[\s-]\d{4}[\s-]\d{4}(?!\d)", "ID_NUMBER"),           # Aadhaar-style
    (r"\b[A-Z]{5}\d{4}[A-Z]\b", "ID_NUMBER"),                           # PAN
]
# A personal name after a title: relatives, doctors, the patient under another spelling.
TITLED_NAME = re.compile(r"\b(Mr|Mrs|Ms|Miss|Dr|Shri|Smt|Kumari)(\.?)\s+((?:[A-Z]\.\s*)*[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)")


@dataclass
class Entity:
    text: str
    type: str          # CONDITION | PROCEDURE | MEDICATION | INVESTIGATION | FLAG
    canonical: str
    sentence: int      # 1-based sentence number (S1, S2, ...)
    start: int
    end: int
    negated: bool = False
    uncertain: bool = False
    duration_years: float | None = None


@dataclass
class PrepResult:
    sections: dict                     # P1 raw sections
    normalized: dict                   # P2
    expanded: dict                     # P3
    expansions: list                   # [(abbr, expansion, count)]
    masked: dict                       # P4
    pii_map: dict                      # placeholder -> original value
    sentences: list                    # P5 [{"n", "section", "text"}]
    entities: list                     # P6-P8 [Entity as dict]
    durations: list                    # P8 [{"sentence", "text", "years"}]
    tokens_est: int                    # P9
    budget: int
    dropped_sentences: list
    stage_log: list = field(default_factory=list)

    def entities_of(self, etype: str, include_negated: bool = False) -> list[dict]:
        return [e for e in self.entities if e["type"] == etype and (include_negated or not e["negated"])]

    def numbered_text(self) -> str:
        return "\n".join(f"S{s['n']}: {s['text']}" for s in self.sentences)


# ---------------------------------------------------------------- stages
def p2_normalize(text: str) -> str:
    t = unicodedata.normalize("NFKC", text or "")
    t = t.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
    t = re.sub(r"[‐-―−]", "-", t)
    t = re.sub(r"[\x00-\x08\x0b-\x1f]", " ", t)
    t = re.sub(r"(\d)(mg|mcg|ml|mm|cm|kg|g)\b", r"\1 \2", t)
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\s+([.,;:])", r"\1", t)
    return t.strip()


def _abbr_pattern(abbr: str) -> str:
    core = re.escape(abbr)
    if abbr.endswith("."):
        return rf"(?<![A-Za-z0-9/]){core}"
    if "/" in abbr:  # k/c/o, c/o ... must not match inside a longer slash form
        return rf"(?<![A-Za-z0-9/]){core}(?![A-Za-z0-9/])"
    return rf"(?<![A-Za-z0-9]){core}(?![A-Za-z0-9])"  # allows lists like DM/HTN/asthma


def p3_expand(text: str) -> tuple[str, list]:
    used = []
    out = text
    for abbr, full, cs in ABBREVIATIONS:
        pat = re.compile(_abbr_pattern(abbr), 0 if cs else re.IGNORECASE)
        count = 0

        def repl(m, full=full, cs=cs):
            nonlocal count
            count += 1
            # keep sentence-initial capitals for case-insensitive forms ("C/o" -> "Complaints of")
            if not cs and m.group(0)[0].isupper():
                return full[0].upper() + full[1:]
            return full

        out = pat.sub(repl, out)
        if count:
            used.append((abbr, full, count))
    # "x 4 years" -> "for 4 years"
    out, n = re.subn(r"(?<![A-Za-z])x\s*(\d+(?:\.\d+)?)\s*(years?|months?|weeks?|days?)\b", r"for \1 \2", out)
    if n:
        used.append(("x <n> <unit>", "for <n> <unit>", n))
    out = re.sub(r"\s{2,}", " ", out)
    return out, used


def p4_deidentify(texts: dict, claim: dict) -> tuple[dict, dict]:
    """Replace direct identifiers with placeholders. Returns (masked_texts, placeholder->original)."""
    pii: dict[str, str] = {}
    rules: list[tuple[str, str]] = []
    name = claim.get("patient", "")
    if name:
        rules.append((re.escape(name), "[PATIENT]"))
        for part in name.split():
            if len(part) >= 3:
                rules.append((rf"\b{re.escape(part)}\b", "[PATIENT]"))
    hosp = claim.get("hospital", "")
    if hosp:
        rules.append((re.escape(hosp), "[HOSPITAL]"))
        short = hosp.split(",")[0].strip()
        if short and short != hosp:
            rules.append((re.escape(short), "[HOSPITAL]"))
    if claim.get("policy_no"):
        rules.append((re.escape(claim["policy_no"]), "[POLICY_NO]"))
    if claim.get("claim_id"):
        rules.append((re.escape(claim["claim_id"]), "[CLAIM_ID]"))
    masked = {}
    counters: dict[str, int] = {}
    for key, text in texts.items():
        t = text
        for pat, ph in rules:
            for m in re.finditer(pat, t, flags=re.IGNORECASE):
                pii.setdefault(ph, name if ph == "[PATIENT]" else (hosp if ph == "[HOSPITAL]" else m.group(0)))
            t = re.sub(pat, ph, t, flags=re.IGNORECASE)
        t = mask_identifiers(t, pii, counters)
        t = re.sub(r"(\[PATIENT\])(\s+\[PATIENT\])+", r"\1", t)
        masked[key] = t
    return masked, pii


def mask_identifiers(text: str, pii: dict, counters: dict) -> str:
    """Mask contact details, ID numbers and titled personal names ("Dr. Mehta" -> "Dr. [PERSON_1]").
    Adds placeholder -> original to `pii`. Shared by P4 and the letter de-identification."""
    for pat, label in CONTACT_PATTERNS:
        def repl(m, label=label):
            counters[label] = counters.get(label, 0) + 1
            ph = f"[{label}_{counters[label]}]"
            pii[ph] = m.group(0)
            return ph
        text = re.sub(pat, repl, text)

    def name_repl(m):
        person = m.group(3)
        ph = next((k for k, v in pii.items() if k.startswith("[PERSON_") and v == person), None)
        if ph is None:
            counters["PERSON"] = counters.get("PERSON", 0) + 1
            ph = f"[PERSON_{counters['PERSON']}]"
            pii[ph] = person
        return f"{m.group(1)}{m.group(2)} {ph}"
    return TITLED_NAME.sub(name_repl, text)


_PROTECT = ["Dr.", "Mr.", "Mrs.", "Ms.", "Smt.", "Shri.", "vs.", "e.g.", "i.e.", "approx.", "No."]


def p5_segment(masked: dict) -> list[dict]:
    sentences = []
    n = 0
    labels = {"diagnosis": "Diagnosis", "treatment": "Proposed treatment"}
    for section in ("diagnosis", "treatment", "notes"):
        text = masked.get(section, "").strip()
        if not text:
            continue
        if section in labels:
            parts = [f"{labels[section]}: {text.rstrip('.')}."]
        else:
            tmp = text
            for i, p in enumerate(_PROTECT):
                tmp = tmp.replace(p, f"__P{i}__")
            parts = [s.strip() for s in re.split(r"(?<=[.!?;])\s+|\n+", tmp) if s.strip()]
            for i, p in enumerate(_PROTECT):
                parts = [s.replace(f"__P{i}__", p) for s in parts]
        for s in parts:
            s = s[0].upper() + s[1:] if s and s[0].islower() else s
            n += 1
            sentences.append({"n": n, "section": section, "text": s})
    return sentences


def _find_terms(sentence: str, terms: list[str]) -> list[tuple[int, int, str]]:
    hits = []
    low = sentence.lower()
    for term in sorted(terms, key=len, reverse=True):
        for m in re.finditer(rf"(?<![a-z0-9]){re.escape(term)}s?(?![a-z0-9])", low):
            hits.append((m.start(), m.end(), term))
    return hits


def p6_tag(sentences: list[dict]) -> list[Entity]:
    groups: list[tuple[str, dict]] = [
        ("CONDITION", CONDITIONS), ("PROCEDURE", PROCEDURES),
        ("MEDICATION", {m: [m] for m in MEDICATIONS}), ("INVESTIGATION", {i: [i] for i in INVESTIGATIONS}),
        ("FLAG", FLAGS),
    ]
    entities: list[Entity] = []
    for s in sentences:
        taken: list[tuple[int, int]] = []
        cands = []
        for etype, table in groups:
            for canonical, syns in table.items():
                for st, en, _ in _find_terms(s["text"], syns):
                    cands.append((st, en, etype, canonical))
        # longest match wins, no overlaps
        for st, en, etype, canonical in sorted(cands, key=lambda c: (-(c[1] - c[0]), c[0])):
            if any(st < b and en > a for a, b in taken):
                continue
            taken.append((st, en))
            entities.append(Entity(text=s["text"][st:en], type=etype, canonical=canonical,
                                   sentence=s["n"], start=st, end=en))
    entities.sort(key=lambda e: (e.sentence, e.start))
    return entities


def _tokens(text: str) -> list[tuple[str, int, int]]:
    return [(m.group(0).lower(), m.start(), m.end()) for m in re.finditer(r"[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*|[^\sA-Za-z0-9]", text)]


def p7_negation(sentences: list[dict], entities: list[Entity]) -> int:
    by_n = {s["n"]: s["text"] for s in sentences}
    flagged = 0
    for n, text in by_n.items():
        toks = _tokens(text)
        words = [t[0] for t in toks]
        scopes: list[tuple[int, int]] = []
        uncertain_scopes: list[tuple[int, int]] = []
        i = 0
        while i < len(words):
            matched = None
            for trig in sorted(PRE_NEGATION, key=lambda x: -len(x.split())):
                tw = trig.split()
                if words[i:i + len(tw)] == tw:
                    matched = tw
                    break
            if matched:
                j = i + len(matched)
                end = j
                while end < len(words) and end - j < NEG_WINDOW and words[end] not in TERMINATORS and words[end] != ".":
                    end += 1
                if end > j:
                    scopes.append((toks[j][1], toks[end - 1][2]))
                i = j
                continue
            for trig in UNCERTAINTY:
                tw = trig.split()
                if words[i:i + len(tw)] == tw:
                    k = i + len(tw)
                    uncertain_scopes.append((toks[min(k, len(toks) - 1)][1], toks[min(k + 5, len(toks) - 1)][2]))
            i += 1
        post = [(m.start(), m.end()) for trig in POST_NEGATION for m in re.finditer(re.escape(trig), text.lower())]
        for e in entities:
            if e.sentence != n:
                continue
            if any(a <= e.start < b for a, b in scopes) or any(0 <= ps - e.end <= 25 for ps, _ in post):
                e.negated = True
                flagged += 1
            if any(a <= e.start < b for a, b in uncertain_scopes):
                e.uncertain = True
    return flagged


_DURATION_PATTERNS = [
    (r"since (\d{4})", "year"),
    (r"(?:for|over|past|x)\s+(\d+(?:\.\d+)?)\s*(years?|months?|weeks?)", "span"),
    (r"(\d+(?:\.\d+)?)\s*(years?|months?)\s+ago", "span"),
]


def _to_years(value: float, unit: str) -> float:
    unit = unit.rstrip("s")
    y = round(value if unit == "year" else value / 12 if unit == "month" else value / 52, 2)
    return int(y) if float(y).is_integer() else y


def p8_durations(sentences: list[dict], entities: list[Entity], request_year: int) -> list[dict]:
    found = []
    for s in sentences:
        low = s["text"].lower()
        spans = []
        for pat, kind in _DURATION_PATTERNS:
            for m in re.finditer(pat, low):
                years = (request_year - int(m.group(1))) if kind == "year" else _to_years(float(m.group(1)), m.group(2))
                spans.append((m.start(), m.group(0), years))
        for st, txt, years in spans:
            found.append({"sentence": s["n"], "text": txt, "years": years})
        conds = [e for e in entities if e.sentence == s["n"] and e.type == "CONDITION"]
        for e in conds:
            after = [sp for sp in spans if sp[0] >= e.end]
            before = [sp for sp in spans if sp[0] < e.start]
            pick = min(after, key=lambda sp: sp[0] - e.end) if after else (max(before, key=lambda sp: sp[0]) if before else None)
            if pick:
                e.duration_years = pick[2]
    return found


def estimate_tokens(text: str) -> int:
    """~4 characters per token for English with Llama-family BPE tokenizers."""
    return math.ceil(len(text) / 4)


def p9_budget(sentences: list[dict], entities: list[Entity], budget: int) -> tuple[list[dict], list[dict], int]:
    total = sum(estimate_tokens(f"S{s['n']}: {s['text']}") + 1 for s in sentences)
    if total <= budget:
        return sentences, [], total
    with_ent = {e.sentence for e in entities}
    ranked = sorted(sentences, key=lambda s: (s["section"] != "notes", s["n"] in with_ent), reverse=True)
    kept, used = [], 0
    for s in ranked:
        cost = estimate_tokens(f"S{s['n']}: {s['text']}") + 1
        if used + cost <= budget:
            kept.append(s)
            used += cost
    kept.sort(key=lambda s: s["n"])
    dropped = [s for s in sentences if s not in kept]
    return kept, dropped, used


# ---------------------------------------------------------------- pipeline
def preprocess_claim(claim: dict, budget: int = CONTEXT_TOKEN_BUDGET) -> PrepResult:
    log = []

    def stage(code, name, summary):
        log.append({"stage": code, "name": name, "summary": summary})
        record("nlp", f"{code} {name}", summary)

    sections = {"diagnosis": claim.get("diagnosis", ""), "treatment": claim.get("proposed_treatment", ""),
                "notes": claim.get("doctor_notes", "")}
    stage("P1", "Assemble", f"{sum(len(v) for v in sections.values())} characters in 3 sections")

    normalized = {k: p2_normalize(v) for k, v in sections.items()}
    stage("P2", "Normalise", "NFKC, punctuation, units, whitespace")

    expanded, used = {}, {}
    for k, v in normalized.items():
        expanded[k], u = p3_expand(v)
        for abbr, full, c in u:
            prev = used.get(abbr, (full, 0))
            used[abbr] = (full, prev[1] + c)
    expansions = [(a, f, c) for a, (f, c) in used.items()]
    stage("P3", "Expand abbreviations", f"{sum(c for *_, c in expansions)} expansions ({len(expansions)} distinct)")

    masked, pii_map = p4_deidentify(expanded, claim)
    stage("P4", "De-identify (PII)", f"{len(pii_map)} identifier(s) masked: {', '.join(pii_map) or 'none'}")

    sentences = p5_segment(masked)
    stage("P5", "Segment sentences", f"{len(sentences)} numbered sentences")

    ents = p6_tag(sentences)
    stage("P6", "Tag clinical entities", f"{len(ents)} entities")

    neg = p7_negation(sentences, ents)
    stage("P7", "Detect negation", f"{neg} entities negated")

    req_year = int(str(claim.get("request_date", "2026"))[:4])
    durations = p8_durations(sentences, ents, req_year)
    stage("P8", "Normalise durations", f"{len(durations)} duration expression(s)")

    kept, dropped, tokens = p9_budget(sentences, ents, budget)
    stage("P9", "Token budget", f"~{tokens} tokens of {budget} budget; {len(dropped)} sentence(s) dropped")

    return PrepResult(sections=sections, normalized=normalized, expanded=expanded, expansions=expansions,
                      masked=masked, pii_map=pii_map, sentences=kept,
                      entities=[asdict(e) for e in ents if e.sentence in {s['n'] for s in kept}],
                      durations=durations, tokens_est=tokens, budget=budget, dropped_sentences=dropped,
                      stage_log=log)


def analyze_numbered_text(numbered: str, request_year: int) -> tuple[list[dict], list[dict]]:
    """Run P6-P8 on already-numbered sentences ("S1: ..."). Used by the simulated LLM."""
    sentences = []
    for line in numbered.splitlines():
        m = re.match(r"S(\d+):\s*(.*)", line.strip())
        if m:
            text = m.group(2)
            section = "diagnosis" if text.startswith("Diagnosis:") else "treatment" if text.startswith("Proposed treatment:") else "notes"
            sentences.append({"n": int(m.group(1)), "section": section, "text": text})
    ents = p6_tag(sentences)
    p7_negation(sentences, ents)
    p8_durations(sentences, ents, request_year)
    return sentences, [asdict(e) for e in ents]


def facts_from_entities(sentences: list[dict], entities: list[dict]) -> dict:
    """Rule-based extractor: turns tagged entities into the facts the agents need."""
    live = [e for e in entities if not e["negated"]]
    section_of = {s["n"]: s["section"] for s in sentences}
    procedure = "medical management"
    for section in ("treatment", "diagnosis", "notes"):
        procs = [e for e in live if e["type"] == "PROCEDURE" and section_of.get(e["sentence"]) == section]
        if procs:
            procedure = procs[0]["canonical"]
            break
    conditions, seen = [], set()
    declared_sents = {e["sentence"] for e in live if e["type"] == "FLAG" and e["canonical"] == "declared"}
    for e in live:
        if e["type"] == "CONDITION" and e["canonical"] not in seen:
            seen.add(e["canonical"])
            conditions.append({"condition": e["canonical"], "duration_years": e["duration_years"],
                               "evidence_sentence": e["sentence"],
                               "stated_as_declared": e["sentence"] in declared_sents})
    flag = lambda c: any(e["type"] == "FLAG" and e["canonical"] == c for e in live)  # noqa: E731
    return {"procedure_key": procedure, "chronic_conditions": conditions,
            "cosmetic_intent": flag("cosmetic"), "functional_need": flag("functional"), "accident": flag("accident")}
