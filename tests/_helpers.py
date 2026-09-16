"""Shared helpers for the test suite.

Imported as a plain module (``from _helpers import imr``), the same way the
benchmark scripts import ``_common``: pytest's prepend import mode puts each
test file's directory on ``sys.path`` because ``tests/`` has no ``__init__.py``.
"""

import importlib.util
import pathlib
import sys

from scipy import stats

REPO = pathlib.Path(__file__).resolve().parents[1]


def imr(t):
    """Inverse Mills ratio phi(t)/(1-Phi(t)) = mean of N(0,1) above t."""
    return stats.norm.pdf(t) / stats.norm.sf(t)


def load_script(relative):
    """Execute a repo script (``examples/``, ``benchmarks/``) and return it.

    The script's own directory goes on ``sys.path`` for the duration of the
    import, so benchmark scripts can resolve their siblings
    (``from _common import ...``)."""
    path = (REPO / relative).resolve()
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(path.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module
