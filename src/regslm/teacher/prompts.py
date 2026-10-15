"""Teacher and gold-pass prompts (paper §3.4 "Silver labels", §3.3 "Gold set").

Three prompt families, deliberately different in wording and example order so that the
two gold passes are independent annotations rather than one prompt run twice:

- ``SILVER``  — the teacher prompt for silver labels; two variants (label-only and
  label + rationale) that differ *only* in the rationale instruction, so RQ2 compares
  training targets, not prompts.
- ``GOLD_A``  — gold pass A (14B teacher): rule-list framing.
- ``GOLD_B``  — gold pass B (8B model): question-driven framing, examples in another order.

All prompts are generated from docs/ANNOTATION_GUIDE.md v1; ``PROMPT_VERSION`` is part of
the teacher cache key, so editing a prompt invalidates the cache.
"""

from __future__ import annotations

from typing import Literal

PROMPT_VERSION = "v3"  # v2: citation.quote (D-020); v3: quote from TEXT, not HEADING

Variant = Literal["label", "rationale", "gold_a", "gold_b"]

_RULES = """\
An obligation is a statement that directs an identified party (a firm, institution, person,
or the authority itself) to do, not do, or be permitted to do something.
- modality: "must" (shall, must, is required to, ต้อง, ให้…), "must_not" (shall not, must not,
  may not, no person shall, ห้าม, มิให้), "may" (may, is permitted to, อาจ, มีสิทธิ — a
  permission, not a possibility), "should" (should, is expected to, ควร).
- NOT obligations: definitions ("means", "includes", "for the purposes of"), scope and
  purpose statements, commencement, amending instructions ("is replaced by"), "may" meaning
  possibility, and guidance that merely restates a rule found elsewhere. For these return
  an empty list.
- Duties of the authority (e.g. "The competent authority shall publish…") ARE obligations.
- If the unit is a list point, subject and modality come from the lead-in given as HEADING.
- If a "shall" sentence names no duty-bearer ("the risk weight shall be 100 %"), the subject
  is the party the instrument addresses in the heading (e.g. "institutions").
- One obligation per (subject, modality, action); split when objects, thresholds or
  deadlines differ; keep coordinated verbs together ("establish and maintain").
- Copy phrases from the text; do not paraphrase. subject = shortest noun phrase; action =
  verb phrase without modal and object; condition = when/if/where clause (join nested
  conditions with " AND "); exceptions = each unless/except/other than clause.
- threshold = the binding number the party must meet: value as a number (4.5 for "4.5 %"),
  unit ("%", "USD", "EUR", "THB", "days", "times"), comparator (">=" at least/not less than/
  minimum; ">" more than/exceed in a prohibition; "<=" not exceed/no more than in a duty;
  "<" less than; "==" exactly). Numbers that only decide WHETHER the rule applies (e.g.
  "where total assets exceed $250 billion") stay inside condition, not threshold.
- deadline = time limit or frequency, copied ("within 30 days", "annually", "at all times").
- citation: source and unit_id exactly as given; quote = the sentence or clause of TEXT that
  contains the trigger word, copied character for character (do not shorten words, do not
  fix typos). If TEXT is a single short item, quote all of it. Always quote from TEXT,
  never from HEADING — for a list point, quote the point itself.
- confidence: your probability (0–1) that the obligation is correct.
Use only numbers that appear in TEXT. Never use values you remember from other versions of
this rule."""

_EXAMPLES = [
    (
        "Institutions shall at all times satisfy a Common Equity Tier 1 capital ratio of 4.5 %.",
        '{"subject":"Institutions","modality":"must","action":"satisfy","object":"a Common '
        'Equity Tier 1 capital ratio","threshold":{"value":4.5,"unit":"%","comparator":">="},'
        '"deadline":"at all times"}',
    ),
    (
        "'Credit institution' means an undertaking the business of which is to take deposits "
        "from the public.",
        "no obligations (definition)",
    ),
    (
        "No person shall act as a broker unless registered with the Commission.",
        '{"subject":"person","modality":"must_not","action":"act","object":"as a broker",'
        '"exceptions":["unless registered with the Commission"]}',
    ),
    (
        "Where a covered company's total consolidated assets exceed $250 billion, it must "
        "conduct a stress test semi-annually.",
        '{"subject":"it","modality":"must","action":"conduct","object":"a stress test",'
        '"condition":"Where a covered company\'s total consolidated assets exceed $250 '
        'billion","deadline":"semi-annually"}',
    ),
    (
        "An institution may, with the prior permission of the competent authority, use "
        "internal models to calculate own funds requirements.",
        '{"subject":"An institution","modality":"may","action":"use","object":"internal '
        'models to calculate own funds requirements","condition":"with the prior permission '
        'of the competent authority"}',
    ),
]


def _examples(order: list[int]) -> str:
    lines = []
    for i in order:
        text, out = _EXAMPLES[i]
        lines.append(f"TEXT: {text}\nOBLIGATIONS (fields only, citation omitted): {out}")
    return "\n\n".join(lines)


SYSTEM_SILVER = (
    "You extract legal obligations from one paragraph of a financial regulation and return "
    "JSON that matches the given schema exactly.\n\n"
    f"Rules:\n{_RULES}\n\nExamples:\n{_examples([0, 1, 2, 3, 4])}"
)
RATIONALE_INSTRUCTION = (
    "\n\nAlso fill `rationale` with at most three short sentences explaining: which trigger "
    "words you used, why any modal sentence is NOT an obligation, and where the condition, "
    "threshold or exception came from. Write the rationale before deciding the obligations."
)
NO_RATIONALE_INSTRUCTION = "\n\nSet `rationale` to null."

SYSTEM_GOLD_A = (
    "You are a careful legal annotator. Follow the annotation rules strictly and return JSON "
    f"that matches the schema.\n\nAnnotation rules:\n{_RULES}"
)
SYSTEM_GOLD_B = (
    "Read the regulatory paragraph and answer, as JSON matching the schema: Who is required, "
    "forbidden, permitted or expected to do something? What exactly? Under what condition, by "
    "when, with what numeric limit, and with what exceptions? Return an empty list if the "
    "paragraph defines terms, describes scope, amends another text, or creates no duty or "
    f"permission.\n\nGuidance:\n{_RULES}\n\nExamples:\n{_examples([4, 2, 1, 3, 0])}"
)


def system_prompt(variant: Variant) -> str:
    if variant == "label":
        return SYSTEM_SILVER + NO_RATIONALE_INSTRUCTION
    if variant == "rationale":
        return SYSTEM_SILVER + RATIONALE_INSTRUCTION
    if variant == "gold_a":
        return SYSTEM_GOLD_A + NO_RATIONALE_INSTRUCTION
    return SYSTEM_GOLD_B + NO_RATIONALE_INSTRUCTION


def user_prompt(source: str, unit_id: str, heading: str, text: str) -> str:
    """The unit as the model sees it. ``text`` must be the exact stored unit text so that
    character offsets in ``citation.span`` resolve."""
    return f"source: {source}\nunit_id: {unit_id}\nHEADING: {heading or '-'}\nTEXT:\n{text}"
