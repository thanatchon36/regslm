"""Gold set construction: compare two annotation passes, queue disagreements, finalise.

See docs/specs/gold.md. Gold units are never used for training or prompt tuning; training
data builders call :func:`assert_no_gold`.
"""

from __future__ import annotations

import csv
import json
import re
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

from regslm.eval.agreement import inter_pass_agreement
from regslm.eval.metrics import FIELDS, _field_equal
from regslm.rewards.normalize import hungarian, match_obligations, normalize_text, thresholds_equal
from regslm.schema.generation import resolve_span
from regslm.schema.obligation import Extraction, Modality, Obligation, Span, Threshold

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


_THAI = re.compile("[\u0e00-\u0e7f]")
SUBJECT_F1 = 0.8
FIELD_F1 = 0.6
TOLERANT_FIELDS = ("action", "object", "condition", "deadline")


def _tokens(s: str) -> set[str]:
    """Whitespace words; character bigrams for Thai (written without spaces)."""
    t = normalize_text(s)
    if _THAI.search(t):
        c = t.replace(" ", "")
        return {c[i : i + 2] for i in range(max(len(c) - 1, 1))} if c else set()
    return set(t.split())


def _tok_f1(a: str, b: str) -> float:
    ta, tb = _tokens(a), _tokens(b)
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    ov = len(ta & tb)
    return 2 * ov / (len(ta) + len(tb))


def _exceptions_f1(a: list[str], b: list[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    free = set(range(len(b)))
    hit = 0
    for x in a:
        best = max(free, key=lambda j: _tok_f1(x, b[j]), default=None)
        if best is not None and _tok_f1(x, b[best]) >= FIELD_F1:
            free.discard(best)
            hit += 1
    return 2 * hit / (len(a) + len(b))


def tolerant_field_diffs(a: Obligation, b: Obligation) -> list[str]:
    """Fields on which two matched obligations disagree under the tolerant test."""
    out: list[str] = []
    if a.modality != b.modality:
        out.append("modality")
    if normalize_text(a.subject) != normalize_text(b.subject) and (
        _tok_f1(a.subject, b.subject) < SUBJECT_F1
    ):
        out.append("subject")
    out += [
        f
        for f in TOLERANT_FIELDS
        if _tok_f1(getattr(a, f), getattr(b, f)) < FIELD_F1
        or (bool(getattr(a, f).strip()) != bool(getattr(b, f).strip()))
    ]
    if not thresholds_equal(a.threshold, b.threshold):
        out.append("threshold")
    if _exceptions_f1(a.exceptions, b.exceptions) < FIELD_F1:
        out.append("exceptions")
    return out


REMATCH_MIN = 0.6


def tolerant_match(oa: list[Obligation], ob: list[Obligation]) -> list[tuple[int, int]]:
    """Span-IoU matching, then a content rematch of the leftovers (tiering only).

    Second pass: Hungarian on mean token-F1 of subject and action, equal modality, >= 0.6.
    """
    pairs = match_obligations(oa, ob)
    ra = [i for i in range(len(oa)) if i not in {p[0] for p in pairs}]
    rb = [j for j in range(len(ob)) if j not in {p[1] for p in pairs}]
    if ra and rb:
        sc = [
            [
                (_tok_f1(oa[i].subject, ob[j].subject) + _tok_f1(oa[i].action, ob[j].action)) / 2
                if oa[i].modality == ob[j].modality
                else 0.0
                for j in rb
            ]
            for i in ra
        ]
        cost = [[-(1000.0 + x) if x >= REMATCH_MIN else 0.0 for x in row] for row in sc]
        pairs += [(ra[i], rb[j]) for i, j in hungarian(cost) if sc[i][j] >= REMATCH_MIN]
    return sorted(pairs)


def tolerant_tier(a: Extraction | None, b: Extraction | None) -> int:
    """0 = agree under the tolerant test; 1 = existence/modality disagreement; 2 = boundaries."""
    if a is None or b is None:
        return 1
    oa, ob = a.obligations, b.obligations
    if not oa and not ob:
        return 0
    if not oa or not ob:
        return 1
    pairs = tolerant_match(oa, ob)
    if len(pairs) < len(oa) or len(pairs) < len(ob):
        return 1
    for i, j in pairs:
        d = tolerant_field_diffs(oa[i], ob[j])
        if "modality" in d:
            return 1
        if d:
            return 2
    return 0


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
    n_strict = 0
    for u in ids:
        info = units[u]
        diffs = diff_summary(pa[u], pb[u], (ua[u], ub[u]))
        n_strict += not diffs
        tier = tolerant_tier(pa[u], pb[u])
        if tier:
            text = info.get("text", "")
            adjudicate.append(
                {
                    "unit_id": u,
                    "stratum": strata[u],
                    "tier": tier,
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
    adjudicate.sort(key=lambda r: (r["tier"], r["language"] != "th", r["language"], r["unit_id"]))
    out_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "n_sample": len(strata),
        "n_both_passes": len(ids),
        "n_agreed": len(agreed),
        "n_to_adjudicate": len(adjudicate),
        "n_agreed_strict": n_strict,
        "n_tier1": sum(r["tier"] == 1 for r in adjudicate),
        "n_tier2": sum(r["tier"] == 2 for r in adjudicate),
        "tolerant": {
            "n_agreed": len(agreed),
            "n_to_adjudicate": len(adjudicate),
            "by_tier_language": {
                f"tier{t}/{lang}": n
                for (t, lang), n in sorted(
                    Counter((r["tier"], r["language"]) for r in adjudicate).items()
                )
            },
        },
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


# ---- CSV adjudication workflow -------------------------------------------------------------

CSV_COLUMNS = (
    "row", "tier", "unit_id", "language", "heading", "text",
    "A", "B", "differences", "decision", "note",
)  # fmt: skip
NO_OBLIGATION = "— no obligation —"
_CMP = {"≥": ">=", "≤": "<="}
_THR_RE = re.compile(r"^(>=|<=|==|>|<|≥|≤)\s*(-?[\d.,]+)\s*(.*)$")


def _clean(s: str) -> str:
    return " ".join(s.replace("|", "/").split())


def format_obligation(o: Mapping[str, Any]) -> str:
    """One-line human-readable obligation (inverse of :func:`parse_obligation_line`)."""
    t = o.get("threshold")
    thr = ""
    if t:
        v = f"{t['value']:g}"
        thr = " ".join(x for x in (t.get("comparator", ">="), v, t.get("unit", "")) if x)
    exc = "; ".join(_clean(x).replace(";", ",") for x in o.get("exceptions", []))
    parts = [
        _clean(o["subject"]),
        o["modality"],
        _clean(o["action"]),
        _clean(o.get("object", "")),
        f"cond: {_clean(o.get('condition', ''))}",
        f"thr: {thr}",
        f"dl: {_clean(o.get('deadline', ''))}",
        f"exc: {exc}",
    ]
    return " | ".join(parts)


def format_candidate(c: Mapping[str, Any] | None) -> str:
    if c is None:
        return "INVALID"
    obs = c.get("obligations", [])
    return "\n".join(format_obligation(o) for o in obs) if obs else NO_OBLIGATION


def parse_obligation_line(line: str) -> dict[str, Any]:
    """Parse one ``format_obligation`` line (without citation); raises ValueError."""
    parts = [p.strip() for p in line.split("|")]
    if len(parts) != 8:
        raise ValueError(f"expected 8 ' | '-separated parts, got {len(parts)}")
    subject, modality, action, obj, cond, thr, dl, exc = parts
    for pre, val in (("cond:", cond), ("thr:", thr), ("dl:", dl), ("exc:", exc)):
        if not val.lower().startswith(pre):
            raise ValueError(f"part {val!r} should start with {pre!r}")
    cond, thr, dl, exc = (v.split(":", 1)[1].strip() for v in (cond, thr, dl, exc))
    try:
        mod = Modality(modality.lower().replace(" ", "_"))
    except ValueError as e:
        raise ValueError(f"unknown modality {modality!r}") from e
    if not subject or not action:
        raise ValueError("subject and action must be non-empty")
    threshold = None
    if thr and thr != "-":
        m = _THR_RE.match(thr)
        if not m:
            raise ValueError(f"cannot parse threshold {thr!r} (use e.g. '<= 5 %')")
        threshold = Threshold(
            value=float(m[2].replace(",", "")), unit=m[3].strip(), comparator=_CMP.get(m[1], m[1])
        ).model_dump(mode="json")
    return {
        "subject": subject,
        "modality": mod.value,
        "action": action,
        "object": obj,
        "condition": cond,
        "threshold": threshold,
        "deadline": dl,
        "exceptions": [x.strip() for x in exc.split(";") if x.strip()],
    }


def export_csv(out_dir: Path = GOLD_DIR) -> dict[str, Any]:
    """Write adjudicate.csv (UTF-8 BOM) and, if openpyxl is installed, adjudicate.xlsx."""
    rows = _read_jsonl(out_dir / "to_adjudicate.jsonl")
    table: list[list[Any]] = []
    for n, r in enumerate(rows, 1):
        dec = {"A": "A", "B": "B", "edit": "E"}.get(r.get("decision") or "", "")
        note = ""
        if dec == "E" and r.get("gold") is not None:
            note = format_candidate(r["gold"]) if r["gold"]["obligations"] else "none"
        table.append(
            [
                n, r.get("tier", ""), r["unit_id"], r["language"], r["heading"][-200:], r["text"],
                format_candidate(r["candidate_a"]), format_candidate(r["candidate_b"]),
                "; ".join(r["diff_summary"]), dec, note,
            ]
        )  # fmt: skip
    path = out_dir / "adjudicate.csv"
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(CSV_COLUMNS)
        w.writerows(table)
    xlsx = None
    try:
        import openpyxl
        from openpyxl.styles import Alignment
    except ImportError:
        pass
    else:
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.append(list(CSV_COLUMNS))
        for t in table:
            ws.append(t)
        widths = (5, 5, 28, 8, 30, 60, 55, 55, 35, 9, 55)
        for col, wd in zip("ABCDEFGHIJK", widths, strict=True):
            ws.column_dimensions[col].width = wd
        for row in ws.iter_rows(min_row=2):
            for c in row:
                c.alignment = Alignment(wrap_text=True, vertical="top")
        ws.freeze_panes = "A2"
        xlsx = out_dir / "adjudicate.xlsx"
        wb.save(xlsx)
    return {"n_rows": len(table), "csv": str(path), "xlsx": str(xlsx) if xlsx else None}


def _edited_gold(row: Mapping[str, Any], note: str) -> dict[str, Any]:
    text = row["text"]
    cands = [
        o
        for c in (row.get("candidate_a"), row.get("candidate_b"))
        if c
        for o in c["obligations"]
    ]
    lines = [ln.strip() for ln in note.splitlines() if ln.strip()]
    obs: list[dict[str, Any]] = []
    if not (len(lines) == 1 and lines[0].lower() == "none"):
        if not lines:
            raise ValueError("decision E needs corrected obligations (or 'none') in note")
        for k, ln in enumerate(lines, 1):
            try:
                o = parse_obligation_line(ln)
            except ValueError as e:
                raise ValueError(f"note line {k}: {e}") from e
            quote = next(
                (
                    c["quote"]
                    for c in cands
                    if normalize_text(c["subject"]) == normalize_text(o["subject"])
                    and normalize_text(c["action"]) == normalize_text(o["action"])
                ),
                text,
            )
            span = resolve_span(text, quote) or Span(start=0, end=len(text))
            src = cands[0]["citation"]["source"] if cands else row["unit_id"].split("/")[0]
            o["citation"] = {
                "source": src,
                "unit_id": row["unit_id"],
                "span": {"start": span.start, "end": span.end},
            }
            o["confidence"] = 1.0
            o["quote"] = text[span.start : span.end]
            obs.append(o)
    d = {"unit_id": row["unit_id"], "rationale": None, "obligations": obs}
    _strip_quotes(d)  # validate against the schema
    return d


def import_csv(csv_path: Path, out_dir: Path = GOLD_DIR, force: bool = False) -> dict[str, Any]:
    """Fill ``decision``/``gold`` in to_adjudicate.jsonl from a filled-in adjudicate.csv."""
    rows = _read_jsonl(out_dir / "to_adjudicate.jsonl")
    by = {r["unit_id"]: r for r in rows}
    with csv_path.open(encoding="utf-8-sig", newline="") as f:
        records = list(csv.DictReader(f))
    errors: list[str] = []
    applied = skipped = 0
    for rec in records:
        uid = (rec.get("unit_id") or "").strip()
        label = f"row {rec.get('row')} ({uid})"
        r = by.get(uid)
        if r is None:
            errors.append(f"{label}: unknown unit_id")
            continue
        d = (rec.get("decision") or "").strip().upper()
        if not d:
            continue
        if d not in ("A", "B", "E"):
            errors.append(f"{label}: decision must be A, B or E, got {d!r}")
            continue
        if r.get("decision") is not None and not force:
            skipped += 1
            continue
        try:
            if d == "E":
                gold_obj: Any = _edited_gold(r, rec.get("note") or "")
                r["decision"] = "edit"
            else:
                gold_obj = r["candidate_a" if d == "A" else "candidate_b"]
                if gold_obj is None:
                    raise ValueError(f"candidate {d} is INVALID and cannot be chosen")
                r["decision"] = d
            r["gold"] = gold_obj
            applied += 1
        except ValueError as e:
            errors.append(f"{label}: {e}")
    _write_jsonl(out_dir / "to_adjudicate.jsonl", rows)
    undecided = [r["unit_id"] for r in rows if r.get("decision") is None]
    return {
        "n_rows": len(rows),
        "applied": applied,
        "skipped_existing": skipped,
        "n_undecided": len(undecided),
        "undecided": undecided,
        "errors": errors,
    }
