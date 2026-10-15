"""Gold set construction: compare two annotation passes, queue disagreements, finalise.

See docs/specs/gold.md. Gold units are never used for training or prompt tuning; training
data builders call :func:`assert_no_gold`.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

from regslm.eval.agreement import inter_pass_agreement
from regslm.eval.metrics import FIELDS, _field_equal
from regslm.rewards.normalize import match_obligations
from regslm.schema.obligation import Extraction

GOLD_DIR = Path("data/gold")
INSTRUCTION = (
    "Choose A, B, or write a corrected `gold` field (same JSON shape as candidate_a); "
    "see docs/ANNOTATION_GUIDE.md."
)
DECISIONS = ("A", "B", "edit")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def read_pass(path: Path) -> tuple[dict[str, Extraction | None], dict[str, int]]:
    """Extractions (None = invalid) and unresolved-quote counts from a ``labels.jsonl``."""
    ex: dict[str, Extraction | None] = {}
    unres: dict[str, int] = {}
    for row in _read_jsonl(path):
        uid = row["unit_id"]
        e = row.get("extraction")
        ex[uid] = Extraction.model_validate(e) if row.get("valid") and e else None
        unres[uid] = int(row.get("n_unresolved", 0))
    return ex, unres


def diff_summary(a: Extraction | None, b: Extraction | None, n_unres: tuple[int, int]) -> list[str]:
    """Short human-readable differences between two passes on one unit (empty = agree)."""
    if a is None or b is None:
        return [f"pass {n} invalid" for n, e in (("A", a), ("B", b)) if e is None]
    out: list[str] = []
    oa, ob = a.obligations, b.obligations
    pairs = match_obligations(oa, ob)
    ma, mb = {i for i, _ in pairs}, {j for _, j in pairs}
    out += [
        f"only A: obligation {i} ({oa[i].subject}: {oa[i].action})"
        for i in range(len(oa))
        if i not in ma
    ]
    out += [
        f"only B: obligation {j} ({ob[j].subject}: {ob[j].action})"
        for j in range(len(ob))
        if j not in mb
    ]
    for i, j in pairs:
        if oa[i].modality != ob[j].modality:
            out.append(f"A{i}/B{j} modality: {oa[i].modality.value} vs {ob[j].modality.value}")
        out += [f"A{i}/B{j} {f} differs" for f in FIELDS if not _field_equal(oa[i], ob[j], f)]
    for name, n in zip("AB", n_unres, strict=True):
        if n:
            out.append(f"pass {name} has {n} unresolved-quote obligation(s)")
    return out


def _with_quotes(e: Extraction | None, text: str) -> dict[str, Any] | None:
    if e is None:
        return None
    d = e.model_dump(mode="json")
    for o in d["obligations"]:
        s = o["citation"]["span"]
        o["quote"] = text[s["start"] : s["end"]]
    return d


def _strip_quotes(d: Mapping[str, Any]) -> Extraction:
    d = json.loads(json.dumps(d))
    for o in d.get("obligations", []):
        o.pop("quote", None)
    return Extraction.model_validate(d)


def build(
    run_a: Path,
    run_b: Path,
    sample: Path,
    units: Mapping[str, Mapping[str, Any]],
    out_dir: Path = GOLD_DIR,
) -> dict[str, Any]:
    """Compare passes; write agreement.json, to_adjudicate.jsonl and agreed.jsonl."""
    pa, ua = read_pass(run_a / "labels.jsonl")
    pb, ub = read_pass(run_b / "labels.jsonl")
    strata = {r["unit_id"]: r.get("stratum", "") for r in _read_jsonl(sample)}
    ids = [u for u in strata if u in pa and u in pb]
    agr = inter_pass_agreement({u: pa[u] for u in ids}, {u: pb[u] for u in ids})
    adjudicate: list[dict[str, Any]] = []
    agreed: list[dict[str, Any]] = []
    for u in ids:
        info = units[u]
        diffs = diff_summary(pa[u], pb[u], (ua[u], ub[u]))
        if diffs:
            text = info.get("text", "")
            adjudicate.append(
                {
                    "unit_id": u,
                    "stratum": strata[u],
                    "language": info.get("language", ""),
                    "heading": info.get("heading", ""),
                    "text": text,
                    "candidate_a": _with_quotes(pa[u], text),
                    "candidate_b": _with_quotes(pb[u], text),
                    "diff_summary": diffs,
                    "instruction": INSTRUCTION,
                    "decision": None,
                    "gold": None,
                }
            )
        else:
            e = pa[u]
            assert e is not None
            agreed.append({"unit_id": u, "stratum": strata[u], "gold": e.model_dump(mode="json")})
    out_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "n_sample": len(strata),
        "n_both_passes": len(ids),
        "n_agreed": len(agreed),
        "n_to_adjudicate": len(adjudicate),
        "agreement": asdict(agr),
    }
    (out_dir / "agreement.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=True) + "\n", encoding="utf-8"
    )
    _write_jsonl(out_dir / "to_adjudicate.jsonl", adjudicate)
    _write_jsonl(out_dir / "agreed.jsonl", agreed)
    return summary


def finalize(out_dir: Path = GOLD_DIR) -> int:
    """Merge agreed and adjudicated rows into ``gold.jsonl``; raise if any decision is null."""
    rows = _read_jsonl(out_dir / "to_adjudicate.jsonl")
    pending = [r["unit_id"] for r in rows if r.get("decision") is None]
    if pending:
        raise ValueError(f"{len(pending)} of {len(rows)} rows have decision null")
    out: list[dict[str, Any]] = []
    for r in _read_jsonl(out_dir / "agreed.jsonl"):
        ex = Extraction.model_validate(r["gold"])
        out.append(
            {
                "unit_id": r["unit_id"],
                "stratum": r["stratum"],
                "source": "agreed",
                "gold": ex.model_dump(mode="json"),
            }
        )
    for r in rows:
        dec = r["decision"]
        if dec not in DECISIONS:
            raise ValueError(f"{r['unit_id']}: decision must be one of {DECISIONS}, got {dec!r}")
        src = {"A": r["candidate_a"], "B": r["candidate_b"], "edit": r.get("gold")}[dec]
        if src is None:
            raise ValueError(f"{r['unit_id']}: decision {dec} but no extraction available")
        out.append(
            {
                "unit_id": r["unit_id"],
                "stratum": r["stratum"],
                "source": dec,
                "gold": _strip_quotes(src).model_dump(mode="json"),
            }
        )
    _write_jsonl(out_dir / "gold.jsonl", out)
    return len(out)


def assert_no_gold(unit_ids: Iterable[str], sample: Path = GOLD_DIR / "sample.jsonl") -> None:
    """Raise if any id is a gold unit (gold is never used for training or prompt tuning)."""
    if not sample.exists():
        raise FileNotFoundError(f"gold sample {sample} missing; cannot verify gold exclusion")
    gold = {r["unit_id"] for r in _read_jsonl(sample)}
    bad = sorted(gold.intersection(unit_ids))
    if bad:
        raise ValueError(f"{len(bad)} gold unit(s) in training data, e.g. {bad[:3]}")
