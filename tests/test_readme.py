"""The README's result blocks must match the committed results (they are generated from them)."""

from __future__ import annotations

from scripts import make_readme_tables as mrt


def test_readme_blocks_are_up_to_date():
    assert mrt.main(check=True) == 0, "README result blocks are stale: run scripts/make_readme_tables.py"


def test_every_block_has_markers_in_the_readme():
    text = mrt.README.read_text(encoding="utf-8")
    for name in mrt.BLOCKS:
        assert f"<!-- BEGIN:{name} -->" in text and f"<!-- END:{name} -->" in text
