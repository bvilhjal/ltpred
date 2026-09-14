"""Optional Numba configuration behaves consistently across backends."""

import importlib

import numpy as np
import pytest

numba_mod = importlib.import_module("ltpred._numba")


@pytest.mark.parametrize("ncores,error,match", [
    *[(v, TypeError, "non-boolean integer") for v in (True, False, np.bool_(True))],
    *[(v, TypeError, "ncores must be an integer") for v in (1.0, np.float64(2), "2", None)],
    *[(v, ValueError, "ncores must be positive") for v in (0, -1, np.int64(-2))],
])
def test_set_num_threads_rejects_invalid_counts(monkeypatch, ncores, error, match):
    called = []
    monkeypatch.setattr(numba_mod, "_set_threads", called.append)

    with pytest.raises(error, match=match):
        numba_mod.set_num_threads(ncores)

    assert called == []


@pytest.mark.parametrize("ncores", [1, np.int64(3)])
def test_set_num_threads_delegates_normalized_integer(monkeypatch, ncores):
    called = []
    monkeypatch.setattr(numba_mod, "_set_threads", called.append)

    numba_mod.set_num_threads(ncores)

    assert called == [int(ncores)]
