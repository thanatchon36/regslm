"""Registry of source adapters, keyed by ``SourceAdapter.name``."""

from __future__ import annotations

from regslm.corpus.base import SourceAdapter
from regslm.corpus.ecfr import EcfrAdapter
from regslm.corpus.eurlex import EurLexAdapter
from regslm.corpus.thai import ThaiAdapter

ADAPTERS: dict[str, type[SourceAdapter]] = {
    EcfrAdapter.name: EcfrAdapter,
    EurLexAdapter.name: EurLexAdapter,
    ThaiAdapter.name: ThaiAdapter,
}
