"""Put both the repository root and `runner/` on the path.

The root so that `pytest tests/` works and not only `python -m pytest tests`,
which adds it implicitly; `runner/` so the tests can import `utils.budget`
without installing the experiment layer as a package.
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
for percorso in (REPO, REPO / "runner"):
    if str(percorso) not in sys.path:
        sys.path.insert(0, str(percorso))
