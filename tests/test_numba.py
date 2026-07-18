"""Optional Numba configuration behaves consistently across backends."""

import importlib

import numpy as np
import pytest

numba_mod = importlib.import_module("ltpred._numba")


@pytest.mark.parametrize("ncores", [True, False, np.bool_(True)])
def test_set_num_threads_rejects_boolean_counts(monkeypatch, ncores):
    called = []
    monkeypatch.setattr(numba_mod, "_set_threads", called.append)

    with pytest.raises(TypeError, match="non-boolean integer"):
        numba_mod.set_num_threads(ncores)

    assert called == []


@pytest.mark.parametrize("ncores", [1.0, np.float64(2), "2", None])
def test_set_num_threads_rejects_non_integer_counts(monkeypatch, ncores):
    called = []
    monkeypatch.setattr(numba_mod, "_set_threads", called.append)

    with pytest.raises(TypeError, match="ncores must be an integer"):
        numba_mod.set_num_threads(ncores)

    assert called == []


@pytest.mark.parametrize("ncores", [0, -1, np.int64(-2)])
def test_set_num_threads_rejects_nonpositive_counts(monkeypatch, ncores):
    called = []
    monkeypatch.setattr(numba_mod, "_set_threads", called.append)

    with pytest.raises(ValueError, match="ncores must be positive"):
        numba_mod.set_num_threads(ncores)

    assert called == []


@pytest.mark.parametrize("ncores", [1, np.int64(3)])
def test_set_num_threads_delegates_normalized_integer(monkeypatch, ncores):
    called = []
    monkeypatch.setattr(numba_mod, "_set_threads", called.append)

    numba_mod.set_num_threads(ncores)

    assert called == [int(ncores)]
