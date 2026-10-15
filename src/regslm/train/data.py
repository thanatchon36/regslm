"""Training-data builder (docs/specs/train.md, "Data builder"; D-020, D-027, D-029)."""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from regslm.corpus.base import Unit
from regslm.schema.generation import GenCitation, GenExtraction, GenObligation
from regslm.schema.obligation import Extraction
from regslm.train.prompts import student_messages

CountPrompt = Callable[[list[dict[str, str]]], int]
CountText = Callable[[str], int]


@dataclass
class LabelRow:
    unit_id: str
    valid: bool
    extraction: Extraction | None
    n_unresolved: int = 0
    unit: Unit | None = None  # embedded in fixture files


@dataclass
class Example:
    unit_id: str
    origin: str  # "teacher" | "rule_empty"
    prompt: list[dict[str, str]]
    target: str
    n_target_tokens: int = 0
    n_tokens: int = 0


@dataclass
class Built:
    examples: list[Example]
    report: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------- label files


def read_label_rows(paths: Iterable[str | Path]) -> dict[str, LabelRow]:
    """Teacher ``labels.jsonl`` rows (later files override earlier ones per unit)."""
    out: dict[str, LabelRow] = {}
    for p in paths:
        for ln in Path(p).read_text(encoding="utf-8").splitlines():
            if not ln.strip():
                continue
            r = json.loads(ln)
            ex = Extraction.model_validate(r["extraction"]) if r.get("extraction") else None
            unit = Unit.model_validate(r["unit"]) if r.get("unit") else None
            out[r["unit_id"]] = LabelRow(
                r["unit_id"], bool(r.get("valid")), ex, int(r.get("n_unresolved", 0)), unit
            )
    return out


def hash_order(ids: Iterable[str], seed: int = 0) -> list[str]:
    """Fixed shuffle independent of the pool, so subsets are nested (500 ⊂ 2k ⊂ ...)."""
    return sorted(set(ids), key=lambda i: hashlib.sha256(f"{seed}:{i}".encode()).hexdigest())


def target_from_label(
    ex: Extraction, text: str, rationale: str | None = None, unit_id: str | None = None
) -> GenExtraction:
    """Rebuild the generation target; ``citation.quote = text[span.start:span.end]`` (D-020)."""
    obls = []
    for o in ex.obligations:
        quote = text[o.citation.span.start : o.citation.span.end]
        data = o.model_dump(exclude={"citation"})
        cit = GenCitation(source=o.citation.source, unit_id=o.citation.unit_id, quote=quote)
        obls.append(GenObligation(**data, citation=cit))
    return GenExtraction(unit_id=unit_id or ex.unit_id, rationale=rationale, obligations=obls)


def target_json(g: GenExtraction) -> str:
    """Compact JSON in schema order."""
    return g.model_dump_json()


def empty_target(unit_id: str) -> str:
    return target_json(GenExtraction(unit_id=unit_id, rationale=None, obligations=[]))


# --------------------------------------------------------------------------- gold guard


def check_no_gold(
    unit_ids: Iterable[str], gold_path: str | Path = "data/gold/sample.jsonl"
) -> None:
    ids = list(unit_ids)
    try:
        from regslm.eval.gold import assert_no_gold  # type: ignore[import-not-found,unused-ignore]
    except ImportError:
        from regslm.data.splits import read_gold_ids

        bad = sorted(set(ids) & read_gold_ids(Path(gold_path)))
        if bad:
            raise AssertionError(f"{len(bad)} gold units in training data: {bad[:3]}") from None
        return
    assert_no_gold(ids, Path(gold_path))


# ------------------------------------------------------------------------------ builder


def _usable(
    labels: dict[str, LabelRow],
    train_units: dict[str, Unit],
    need_rationale: bool,
    drops: Counter[str],
) -> list[str]:
    ok: list[str] = []
    for uid in hash_order(labels):
        r = labels[uid]
        if uid not in train_units:
            drops["not_train"] += 1
        elif not r.valid or r.extraction is None:
            drops["invalid_label"] += 1
        elif r.n_unresolved > 0:
            drops["unresolved_obligations"] += 1
        elif need_rationale and not (r.extraction.rationale or "").strip():
            drops["no_rationale"] += 1
        else:
            ok.append(uid)
    return ok


def rule_empty_ratio(train_units: Iterable[Unit], trigger: Callable[[str, str], bool]) -> float:
    flags = [trigger(u.text, u.heading) for u in train_units]
    t = sum(flags)
    return (len(flags) - t) / t if t else 0.0


def _teacher_example(
    u: Unit, r: LabelRow, with_rationale: bool, cp: CountPrompt, ct: CountText, suffix: str
) -> Example | None:
    assert r.extraction is not None
    rat = r.extraction.rationale if with_rationale else None
    try:
        g = target_from_label(r.extraction, u.text, rat, u.unit_id)
    except ValidationError:
        return None
    return _mk(u, "teacher", target_json(g), cp, ct, suffix)


def _mk(u: Unit, origin: str, target: str, cp: CountPrompt, ct: CountText, suffix: str) -> Example:
    p = student_messages(u)
    return Example(u.unit_id, origin, p, target, ct(target), cp(p) + ct(target + suffix))


def build_examples(
    *,
    n_labels: int,
    arm: str,
    seed: int,
    labels: dict[str, LabelRow],
    rationale_labels: dict[str, LabelRow] | None,
    train_units: dict[str, Unit],
    count_prompt: CountPrompt,
    count_text: CountText,
    max_seq_len: int = 2048,
    suffix: str = "",
    match_rationale_tokens: bool = False,
    rule_empty: bool = True,
    trigger: Callable[[str, str], bool] | None = None,
) -> Built:
    if trigger is None:
        from regslm.corpus.sample import has_trigger as trigger
    drops: Counter[str] = Counter()
    use_rat = arm == "rationale"
    pool = rationale_labels if use_rat else labels
    if pool is None:
        raise ValueError("rationale arm needs rationale_labels")
    ok = _usable(pool, train_units, use_rat, drops)

    ratio = rule_empty_ratio(train_units.values(), trigger) if rule_empty else 0.0
    rule_pool = hash_order(
        uid
        for uid, u in train_units.items()
        if uid not in labels
        and uid not in (rationale_labels or {})
        and not trigger(u.text, u.heading)
    )

    def n_rule(n: int) -> int:
        return min(round(ratio * n), len(rule_pool))

    def teacher_ex(uid: str, rationale: bool = use_rat) -> Example | None:
        src = (rationale_labels if rationale else labels) or {}
        return _teacher_example(
            train_units[uid], src[uid], rationale, count_prompt, count_text, suffix
        )

    k = min(n_labels, len(ok))
    budget = None
    if match_rationale_tokens and arm == "label":
        rb = build_examples(
            n_labels=n_labels, arm="rationale", seed=seed, labels=labels,
            rationale_labels=rationale_labels, train_units=train_units,
            count_prompt=count_prompt, count_text=count_text, max_seq_len=max_seq_len,
            suffix=suffix, rule_empty=rule_empty, trigger=trigger,
        )  # fmt: skip
        budget = sum(e.n_target_tokens for e in rb.examples)
        tok_t: list[int] = []
        tok_r: list[int] = []

        def total(kk: int) -> int:
            while len(tok_t) < kk:
                e = teacher_ex(ok[len(tok_t)])
                tok_t.append(e.n_target_tokens if e else 0)
            while len(tok_r) < n_rule(kk):
                tok_r.append(count_text(empty_target(rule_pool[len(tok_r)])))
            return sum(tok_t[:kk]) + sum(tok_r[: n_rule(kk)])

        k = min(n_labels, len(ok))
        while k < len(ok) and total(k) < budget:
            k += 1

    sel = ok[:k]
    exs = [e for uid in sel if (e := teacher_ex(uid)) is not None]
    if len(sel) > len(exs):
        drops["bad_target"] += len(sel) - len(exs)
    for uid in rule_pool[: n_rule(len(exs))]:
        exs.append(
            _mk(train_units[uid], "rule_empty", empty_target(uid), count_prompt, count_text, suffix)
        )

    if arm == "placebo":
        payloads = [GenExtraction.model_validate_json(e.target) for e in exs]
        random.Random(seed).shuffle(payloads)
        permuted = []
        for e, p in zip(exs, payloads, strict=True):
            tgt = target_json(p.model_copy(update={"unit_id": e.unit_id}))
            permuted.append(
                _mk(train_units[e.unit_id], e.origin, tgt, count_prompt, count_text, suffix)
            )
        exs = permuted

    kept = [e for e in exs if e.n_tokens <= max_seq_len]
    report = {
        "arm": arm,
        "n_labels_requested": n_labels,
        "n_labels_selected": len(sel),
        "n_usable_labels": len(ok),
        "dropped": dict(drops),
        "dropped_too_long": len(exs) - len(kept),
        "n_teacher": sum(e.origin == "teacher" for e in kept),
        "n_rule_empty": sum(e.origin == "rule_empty" for e in kept),
        "rule_empty_ratio": ratio,
        "n_train_units": len(train_units),
        "target_tokens": sum(e.n_target_tokens for e in kept),
        "token_budget": budget,
    }
    return Built(kept, report)


# ----------------------------------------------------------------------- IO / tokenizer


def load_train_units(
    splits: str | Path,
    corpus: str | Path = "data/corpus/units.parquet",
    local: str | Path = "data/local/corpus/units_local.parquet",
) -> dict[str, Unit]:
    """All split == train units (tier P only, D-011) keyed by unit_id."""
    import polars as pl

    ids = pl.read_parquet(splits).filter(pl.col("split") == "train").select("unit_id")
    frames = [pl.read_parquet(corpus)]
    if Path(local).exists():
        frames.append(pl.read_parquet(local))
    df = pl.concat(frames, how="diagonal").unique("unit_id").join(ids, on="unit_id", how="inner")
    if "tier" in df.columns:
        df = df.filter(pl.col("tier") == "P")
    fields = list(Unit.model_fields)
    return {r["unit_id"]: Unit.model_validate(r) for r in df.select(fields).iter_rows(named=True)}


def completion_suffix(tokenizer: Any) -> str:
    """Turn-end text the chat template appends after an assistant message (e.g. <|im_end|>)."""
    kw = {"tokenize": False, "enable_thinking": False}
    full = tokenizer.apply_chat_template(
        [{"role": "user", "content": "u"}, {"role": "assistant", "content": "ZZZ"}], **kw
    )
    return str(full.split("ZZZ", 1)[1]).rstrip()


def render_prompt(tokenizer: Any, messages: Sequence[dict[str, str]]) -> str:
    return str(
        tokenizer.apply_chat_template(
            list(messages), tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    )


def encode(tokenizer: Any, ex: Example, suffix: str) -> dict[str, list[int]]:
    """Token ids with a completion mask (loss on target + turn end only)."""
    p = tokenizer(render_prompt(tokenizer, ex.prompt), add_special_tokens=False)["input_ids"]
    c = tokenizer(ex.target + suffix, add_special_tokens=False)["input_ids"]
    return {"input_ids": p + c, "completion_mask": [0] * len(p) + [1] * len(c)}


def tokenizer_counters(tokenizer: Any, suffix: str) -> tuple[CountPrompt, CountText]:
    def cp(msgs: list[dict[str, str]]) -> int:
        return len(tokenizer(render_prompt(tokenizer, msgs), add_special_tokens=False)["input_ids"])

    def ct(s: str) -> int:
        return len(tokenizer(s, add_special_tokens=False)["input_ids"])

    return cp, ct


def fixture_split(
    labels: dict[str, LabelRow], gold_sample: str | Path, eval_n: int
) -> tuple[dict[str, Unit], dict[str, LabelRow], list[LabelRow]]:
    """Fixture mode: units embedded in label rows; gold removed; last ``eval_n`` held out."""
    from regslm.data.splits import read_gold_ids

    gold = read_gold_ids(Path(gold_sample))
    labels = {u: r for u, r in labels.items() if r.unit is not None and u not in gold}
    units = {u: r.unit for u, r in labels.items() if r.unit is not None}
    held: list[LabelRow] = []
    if eval_n:
        good = [u for u in hash_order(labels) if labels[u].valid and labels[u].n_unresolved == 0]
        for u in good[-eval_n:]:
            held.append(labels.pop(u))
            units.pop(u)
    return units, labels, held


def build_from_config(cfg: Any, tokenizer: Any) -> tuple[Built, list[LabelRow]]:
    """Resolve files from a TrainConfig, build examples. Returns (built, held-out eval rows)."""
    labels = read_label_rows(cfg.labels)
    rat = read_label_rows(cfg.rationale_labels) if cfg.rationale_labels else None
    suffix = completion_suffix(tokenizer)
    cp, ct = tokenizer_counters(tokenizer, suffix)
    held: list[LabelRow] = []
    if cfg.splits is not None:
        train_units = load_train_units(cfg.splits)
    else:  # fixture mode: units are embedded in the label rows
        train_units, labels, held = fixture_split(labels, cfg.gold_sample, cfg.eval_n)
    langs = getattr(cfg, "langs", None)
    if langs:  # e.g. the H5c English-only transfer cell; label subsets stay nested per language
        train_units = {u: x for u, x in train_units.items() if x.language in langs}
    built = build_examples(
        n_labels=cfg.n_labels, arm=cfg.arm, seed=cfg.seed, labels=labels, rationale_labels=rat,
        train_units=train_units, count_prompt=cp, count_text=ct, max_seq_len=cfg.max_seq_len,
        suffix=suffix, match_rationale_tokens=cfg.match_rationale_tokens, rule_empty=cfg.rule_empty,
    )  # fmt: skip
    check_no_gold([e.unit_id for e in built.examples], cfg.gold_sample)
    built.report["origin_column"] = dict(Counter(e.origin for e in built.examples))
    return built, held
