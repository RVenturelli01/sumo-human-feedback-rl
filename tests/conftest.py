"""Make `experiments/` importable so the tests can use utils.budget."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
