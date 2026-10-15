"""Document segmentation and extraction plumbing (CPU, no model)."""

from __future__ import annotations

import io
import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from regslm.serve import extract as ex
from regslm.serve.segment import (
    detect_language,
    read_paragraphs,
    segment,
    segment_file,
    split_long,
)

FIX = Path(__file__).parent / "fixtures" / "thai"

EN_TXT = """Part 2 Capital

Article 5 Buffers

A bank must hold a buffer of at least 2.5% of risk-weighted assets.

A bank shall notify the regulator within 10 days if:
(a) its buffer falls below the minimum;
(b) it breaches a limit.

Article 6 Reporting

Banks must report quarterly.
"""


def test_language_detection() -> None:
    assert detect_language("A bank must hold capital.") == "en"
    assert detect_language("ธนาคารต้องดำรงเงินกองทุน") == "th"
    assert detect_language("ธนาคาร " + "x" * 100) == "en"
    assert detect_language("hello", "th") == "th"


def test_text_segmentation_ids_and_leadins(tmp_path: Path) -> None:
    f = tmp_path / "rule.txt"
    f.write_text(EN_TXT)
    units = segment_file(f)
    ids = [u.unit_id for u in units]
    assert ids == [
        "doc/Part2/Article5/1",
        "doc/Part2/Article5/2",
        "doc/Part2/Article5/3",
        "doc/Part2/Article5/4",
        "doc/Part2/Article6/1",
    ]
    assert len(set(ids)) == len(ids)
    lead = units[1].text
    assert lead.endswith("if:")
    assert units[2].heading == lead and units[3].heading == lead
    assert units[2].text.startswith("(a)")
    assert units[0].heading.startswith("Article 5")
    assert units[4].heading.startswith("Article 6")
    assert units[0].language == "en"


def test_no_headings_flat_ids() -> None:
    units = segment(["First para.", "Second para."], "en")
    assert [u.unit_id for u in units] == ["doc/1", "doc/2"]


def test_thai_headings() -> None:
    paras = ["มาตรา ๕ ผู้ใดประกอบธุรกิจต้องได้รับอนุญาตจากธนาคารแห่งประเทศไทยก่อน", "ข้อ ๓", "ให้ธนาคารรายงาน"]
    units = segment(paras, "th")
    assert [u.unit_id for u in units] == ["doc/มาตรา5/1", "doc/ข้อ3/1"]
    assert units[1].heading == "ข้อ ๓"


def test_section_sign_and_markdown(tmp_path: Path) -> None:
    f = tmp_path / "a.md"
    f.write_text("# § 3.10 Capital\n\nThe firm must keep records.\n")
    units = segment_file(f)
    assert units[0].unit_id == "doc/§3.10/1"


def test_html(tmp_path: Path) -> None:
    f = tmp_path / "a.html"
    f.write_text(
        "<html><head><title>x</title><style>p{}</style></head><body><h2>Article 5</h2>"
        "<p>A bank must  report.</p><ul><li>(a) one</li><li>(b) two</li></ul></body></html>"
    )
    paras, lang = read_paragraphs(f)
    assert paras == ["Article 5", "A bank must report.", "(a) one", "(b) two"]
    assert lang == "en"


def test_split_long_keeps_text() -> None:
    p = ". ".join(f"Sentence number {i} must hold" for i in range(300))
    parts = split_long(p, 500)
    assert all(len(x) <= 500 for x in parts)
    assert " ".join(parts).replace(" ", "") == p.replace(" ", "")


def test_pdf_fixtures() -> None:
    en = segment_file(FIX / "fiba_en_trimmed.pdf")
    assert en and en[0].language == "en"
    th = segment_file(FIX / "botn25550332_trimmed.pdf")
    assert th and th[0].language == "th"
    assert len({u.unit_id for u in th}) == len(th)


def test_unsupported_suffix(tmp_path: Path) -> None:
    f = tmp_path / "a.docx"
    f.write_text("x")
    with pytest.raises(ValueError):
        segment_file(f)


def _fake(units_text: dict[str, str]):
    """Backend answering from the TEXT in each prompt."""

    def fn(batch):
        outs = []
        for msgs in batch:
            user = msgs[1]["content"]
            uid = user.split("unit_id: ")[1].split("\n")[0]
            text = user.split("TEXT:\n", 1)[1]
            if "garbage" in text:
                outs.append("not json")
            elif "none" in text:
                outs.append(json.dumps({"unit_id": uid, "obligations": []}))
            else:
                q = text.split(".")[0] + "." if "." in text else text
                if "ghost" in text:
                    q = "a sentence that is not in the text at all"
                outs.append(
                    json.dumps(
                        {
                            "unit_id": uid,
                            "obligations": [
                                {
                                    "subject": "bank",
                                    "modality": "must",
                                    "action": "report",
                                    "citation": {"source": "doc", "unit_id": uid, "quote": q},
                                }
                            ],
                        }
                    )
                )
        return outs

    return ex.FnBackend(fn)


def test_extract_plumbing_and_summary() -> None:
    units = segment(
        ["A bank must report.", "garbage here.", "none here.", "ghost bank must report."],
        "en",
    )
    rows, summary = ex.extract_units(units, _fake({}), batch_size=2)
    assert summary["units"] == 4
    assert summary["obligations"] == 1 and len(rows) == 1
    assert summary["invalid"] == 1 and summary["invalid_unit_ids"] == ["doc/2"]
    assert summary["unresolved"] == 1
    r = rows[0]
    assert r["citation"]["unit_id"] == "doc/1"
    s, e = r["citation"]["span"]["start"], r["citation"]["span"]["end"]
    assert r["unit_text"][s:e] == r["quote"] == "A bank must report."

    buf = io.StringIO()
    ex.write_jsonl(rows, summary, buf)
    lines = [json.loads(x) for x in buf.getvalue().splitlines()]
    assert len(lines) == 2 and "_summary" in lines[-1]
    assert lines[-1]["_summary"]["disclaimer"] == ex.DISCLAIMER
    assert "disclaimer" not in lines[0]
    assert "1 invalid" in ex.summary_line(summary)


def test_choose_backend() -> None:
    assert ex.choose_backend(None, None, "x/y-awq") == ("vllm", "awq")
    assert ex.choose_backend(None, "gguf", "x") == ("llamacpp", "gguf")
    assert ex.choose_backend(None, None, "m.gguf") == ("llamacpp", "gguf")
    with pytest.raises(ValueError):
        ex.choose_backend("llamacpp", "awq", "x")
    with pytest.raises(ValueError):
        ex.choose_backend("foo", None, "x")


def test_cli_help_and_bad_args(tmp_path: Path) -> None:
    from regslm.cli import app

    r = CliRunner().invoke(app, ["extract", "--help"])
    assert r.exit_code == 0 and "--model" in r.output and "--guided" in r.output
    f = tmp_path / "a.txt"
    f.write_text("x")
    r = CliRunner().invoke(app, ["extract", str(f), "--model", "m", "--backend", "nope"])
    assert r.exit_code != 0
