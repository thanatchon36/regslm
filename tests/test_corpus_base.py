import urllib.robotparser

from regslm.corpus.base import USER_AGENT, make_unit_id, robots_lines

ECFR_LIKE = """User-agent: *
Disallow: /some/path/

  Disallow: /api/versioner/v1/full/
  Disallow: /search
"""


def test_robots_blank_line_inside_group_is_honoured() -> None:
    naive = urllib.robotparser.RobotFileParser()
    naive.parse(ECFR_LIKE.splitlines())
    rp = urllib.robotparser.RobotFileParser()
    rp.parse(robots_lines(ECFR_LIKE))
    url = "https://www.ecfr.gov/api/versioner/v1/full/2026-01-01/title-12.xml"
    assert naive.can_fetch(USER_AGENT, url)  # the stdlib bug this guards against
    assert not rp.can_fetch(USER_AGENT, url)
    assert rp.can_fetch(USER_AGENT, "https://www.ecfr.gov/api/versioner/v1/versions/title-12.json")


def test_make_unit_id() -> None:
    assert make_unit_id("ecfr", "12CFR3", "§ 3.10", "(a)", "") == "ecfr/12CFR3/§3.10/(a)"
