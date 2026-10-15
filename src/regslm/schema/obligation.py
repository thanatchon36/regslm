"""Obligation schema (paper §3.2 "Obligation schema").

An *obligation* is a deontic statement extracted from one hierarchically-addressed
corpus unit: who (subject) must / must not / may / should do what (action, object),
under which condition, subject to which numeric threshold and deadline, with which
exceptions, citing the exact source span.

Design rules (see docs/DECISIONS.md D-003..D-006 and docs/ANNOTATION_GUIDE.md):
- Every text field is a *verbatim or near-verbatim* phrase from the unit; the
  ``citation.span`` is the character offset of the trigger sentence and must
  resolve against the unit text (checked by ``rewards.span_grounded``).
- Numeric thresholds are normalised (``value`` is a float, ``unit`` a short
  token like ``"%"``, ``"EUR"``, ``"days"``) so perturbation tests can compare
  exact values.
- A unit with no obligation yields ``[]`` — definitions, recitals and purely
  descriptive text are not obligations.
"""

from __future__ import annotations

import json
from enum import StrEnum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Modality(StrEnum):
    """Deontic force. ``should`` covers guidance language ("should", "is expected to")."""

    MUST = "must"
    MUST_NOT = "must_not"
    MAY = "may"
    SHOULD = "should"


class Comparator(StrEnum):
    GE = ">="
    GT = ">"
    LE = "<="
    LT = "<"
    EQ = "=="


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)


class Threshold(_Strict):
    value: float
    unit: Annotated[str, Field(max_length=32)] = ""
    comparator: Comparator = Comparator.GE


class Span(_Strict):
    """Half-open character offsets ``[start, end)`` into the unit text."""

    start: Annotated[int, Field(ge=0)]
    end: Annotated[int, Field(ge=0)]

    @model_validator(mode="after")
    def _ordered(self) -> Span:
        if self.end <= self.start:
            raise ValueError("span.end must be > span.start")
        return self


class Citation(_Strict):
    source: Annotated[str, Field(min_length=1)]
    unit_id: Annotated[str, Field(min_length=1)]
    span: Span


class Obligation(_Strict):
    subject: Annotated[str, Field(min_length=1, max_length=300)]
    modality: Modality
    action: Annotated[str, Field(min_length=1, max_length=300)]
    object: Annotated[str, Field(max_length=500)] = ""
    condition: Annotated[str, Field(max_length=800)] = ""
    threshold: Threshold | None = None
    deadline: Annotated[str, Field(max_length=200)] = ""
    exceptions: list[Annotated[str, Field(max_length=500)]] = Field(default_factory=list)
    citation: Citation
    confidence: Annotated[float, Field(ge=0.0, le=1.0)] = 1.0


class Extraction(_Strict):
    """All obligations from one unit; the unit of model output."""

    unit_id: str
    # Before ``obligations`` on purpose: guided decoding emits fields in schema order, so the
    # rationale is generated first and conditions the labels (D-016).
    rationale: str | None = None
    obligations: list[Obligation] = Field(default_factory=list)


def json_schema(model: type[BaseModel] = Extraction) -> str:
    """JSON Schema used for vLLM guided decoding and exported to ``schemas/``."""
    return json.dumps(model.model_json_schema(), indent=2, ensure_ascii=False)
