"""Make the repo root importable so tests can `from harness...` / `from tasks...`.

pytest's default (prepend) import mode puts each test file's own directory on
sys.path, not the repo root. A top-level conftest guarantees the root is there
regardless of where pytest is invoked from.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
