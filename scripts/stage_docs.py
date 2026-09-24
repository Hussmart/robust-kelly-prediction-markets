"""Helper for the staged commit history: write ``docs/methodology.md`` truncated after stage N.

Usage: ``python scripts/stage_docs.py N`` keeps the sections that exist at stage N
(3 -> sections 1-2, 4 -> 1-3, 5 -> 1-4, 7 -> 1-5, 8 or higher -> everything).
The full document is kept in ``docs/methodology.full.md`` (untracked) between calls.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

DOCS = Path(__file__).resolve().parents[1] / "docs"
FULL = DOCS / "methodology.full.md"
TARGET = DOCS / "methodology.md"
LAST_SECTION = {3: 2, 4: 3, 5: 4, 6: 4, 7: 5, 8: 6, 9: 6}


def main(stage: int) -> None:
    """Write the truncated methodology for ``stage``."""
    if not FULL.exists():
        FULL.write_text(TARGET.read_text(encoding="utf-8"), encoding="utf-8")
    text = FULL.read_text(encoding="utf-8")
    keep = LAST_SECTION[stage]
    match = re.search(rf"\n---\n\n## {keep + 1}\. ", text)
    TARGET.write_text(text[: match.start()] + "\n" if match else text, encoding="utf-8")


if __name__ == "__main__":
    main(int(sys.argv[1]))
