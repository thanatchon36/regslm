"""Student prompts (docs/specs/train.md): short system prompt, teacher user prompt."""

from __future__ import annotations

from regslm.corpus.base import Unit
from regslm.teacher.prompts import user_prompt

STUDENT_SYSTEM = (
    "Extract the legal obligations from the regulatory TEXT as JSON with keys unit_id, "
    "rationale, obligations. Each obligation has subject, modality (must, must_not, may, "
    "should), action, object, condition, threshold (value, unit, comparator), deadline, "
    "exceptions, citation (source, unit_id, quote) and confidence. citation.quote must "
    "copy the trigger sentence from TEXT. Use an empty obligations list if there is no "
    "obligation."
)


def student_messages(unit: Unit) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": STUDENT_SYSTEM},
        {
            "role": "user",
            "content": user_prompt(unit.source, unit.unit_id, unit.heading, unit.text),
        },
    ]
