"""Obligation schema, validators and perturbation utilities (paper §3.2)."""

from regslm.schema.generation import (
    GenCitation,
    GenExtraction,
    GenObligation,
    Resolved,
    resolve,
    resolve_span,
)
from regslm.schema.obligation import (
    Citation,
    Comparator,
    Extraction,
    Modality,
    Obligation,
    Span,
    Threshold,
    json_schema,
)

__all__ = [
    "Citation",
    "Comparator",
    "Extraction",
    "GenCitation",
    "GenExtraction",
    "GenObligation",
    "Modality",
    "Obligation",
    "Resolved",
    "Span",
    "Threshold",
    "json_schema",
    "resolve",
    "resolve_span",
]
