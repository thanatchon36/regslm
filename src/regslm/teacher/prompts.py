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

PROMPT_VERSION = "v5"  # v2 quote (D-020); v3 quote from TEXT; v4 guide v2.0 (D-036); v5 lead-in points (D-038)

Variant = Literal["label", "rationale", "gold_a", "gold_b"]

_RULES = """\
An obligation is a MAIN CLAUSE that directs a NAMED party (a firm, institution, person, or an
authority) to do, not do, or be permitted or expected to do something.
- modality: "must" (shall, must, is required to, ต้อง, มีหน้าที่, directive ให้ + actor + action,
  ให้ถือปฏิบัติ); "must_not" (shall not, must not, may not, no person shall, ห้าม, มิให้); "may"
  (may, is permitted to, มีสิทธิ, อาจ/สามารถ…ได้ only when a party is granted a right — not a
  possibility or capability); "should" (should, is expected to, ควร).
- The subject must be named in TEXT or in a lead-in sentence in HEADING ("A bank must:"). Never
  invent a subject.
- LIST POINTS: when HEADING ends with a lead-in sentence that contains a trigger and a named
  party (e.g. "… | Competent authorities shall determine whether consolidation is to be carried
  out in the following cases:", "… | an institution may exclude the following exposures:",
  "… | institutions shall report the following information:"), the point in TEXT IS an
  obligation even though it has no verb of its own: subject and modality come from the lead-in,
  action from the lead-in, and the point's content goes in object (a thing: "assets representing
  claims on the central bank") or condition (a case: "where … an institution exercises a
  significant influence …"). Quote the point from TEXT. If the lead-in itself is a calculation,
  deeming or delegation rule (no named party), the point is not an obligation. A "shall" sentence with no addressee ("the risk weight shall be 100 %", "the
  value shall be equal to the sum of…") is a calculation rule: return no obligation for it.
  Section titles never supply modality.
- Return an EMPTY list for: definitions ("means", "includes", "for the purposes of", "X = …");
  modals inside a noun phrase or relative clause ("the amount that the bank must return");
  scope/applicability ("shall apply to", "shall not apply to"); deeming ("shall be treated as",
  "is deemed", ให้ถือว่า); commencement, repeal (ให้ยกเลิก), amending instructions; legislative
  delegation ("power is delegated to the Commission to adopt…", "EBA shall develop draft
  standards"); official interpretations/commentary or guidance restating a rule; composition or
  term of office (ให้…ประกอบด้วย, ให้…อยู่ในตำแหน่ง); table cells, form codes, fragments with no main
  clause (unless a lead-in in HEADING applies, see LIST POINTS); "may" or อาจ meaning possibility. Exception: "an institution shall treat X as Y" or
  "shall apply Article X" directs a party → obligation.
- Thai: ให้ is NOT a trigger when it means "give/grant" (ให้สินเชื่อ) or in เพื่อให้. If the
  subject is in a preceding ในกรณีที่… clause of the same sentence, copy it from there.
- Operative duties and powers of an authority ARE obligations ("The competent authority shall
  publish…", "The Commission may exempt…").
- One obligation per (subject, modality, action); split when objects, thresholds or deadlines
  differ; keep coordinated verbs together ("establish and maintain"). "must ensure that others
  do not…" is ONE obligation on the party that must ensure.
- Copy phrases from TEXT; do not paraphrase. subject = shortest noun phrase (replace pronouns like
  "it" with the noun phrase they refer to; "No person shall" → subject "person"); action = verb
  phrase without modal and object ("be" for a limit on a quantity); object includes the base of a
  limit; condition = when/if/where/provided-that clause, nested conditions joined with " AND ";
  exceptions = each unless/except/other-than carve-out ("notwithstanding" is NOT an exception).
- threshold = the BINDING number defining compliance. value = the number as written converted to
  Arabic digits (4.5 for "4,5 %", 25 for "๒๕" or "ยี่สิบห้า", 250 for "$250 billion"); unit includes
  any scale word ("%", "USD billion", "EUR million", "days", "THB"). Comparator: for a LIMIT ON A
  QUANTITY ("X shall not exceed Y", "at least Y", "maintain a ratio of at least Y") use modality
  "must" and the compliant region ("<=" not exceed/no more than, ">=" at least/not less than/
  minimum, "==" equal to). For a PROHIBITED ACT defined by a limit ("shall not grant credit
  exceeding Y") use "must_not" and the prohibited region (">" exceeding/more than, "<" below).
  A number that only decides WHETHER the duty applies ("where total assets exceed $250 billion")
  stays in condition, not threshold.
- deadline = time limit, frequency or duration, copied verbatim ("within 30 days", "annually",
  "at all times", "ภายในสามสิบวัน"). Time limits are never thresholds.
- delegations ("as specified by the Bank of Thailand", ตามที่ธนาคารแห่งประเทศไทยประกาศกำหนด): keep the
  phrase verbatim in object or condition; do not invent numbers.
- citation: source and unit_id exactly as given; quote = the sentence or clause of TEXT that
  contains the trigger, copied character for character. Quote from TEXT, never from HEADING —
  for a list point, quote the point itself. If TEXT is a single short item, quote all of it.
- confidence: your probability (0-1) that the obligation is correct.
Use only numbers that appear in TEXT (digits, Thai digits or number words). Never use values you
remember from other versions of this rule."""

_EXAMPLES = [
    (
        "Large exposures shall not exceed 25 % of Tier 1 capital.",
        '{"subject":"Large exposures","modality":"must","action":"be","object":"of Tier 1 '
        'capital","threshold":{"value":25,"unit":"%","comparator":"<="}}',
    ),
    (
        "The value of the aggregated long position shall be equal to the sum of the positions.",
        "no obligations (no named addressee: calculation rule)",
    ),
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
        '{"subject":"a covered company","modality":"must","action":"conduct","object":'
        '"a stress test",'
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
    f"Rules:\n{_RULES}\n\nExamples:\n{_examples([2, 3, 0, 4, 5, 1, 6])}"
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
    f"permission.\n\nGuidance:\n{_RULES}\n\nExamples:\n{_examples([6, 4, 1, 3, 5, 0, 2])}"
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
