"""Optional Numba acceleration shim.

The truncated-MVN Gibbs sampler's inner sweep dominates runtime; JIT-compiling it
gives a large speed-up, and because families are independent the estimator also
runs many of them in parallel (``_jit_parallel`` + ``prange``). If Numba is not
installed we fall back to no-op decorators (``prange`` becomes ``range``) and run
the identical code in pure Python -- just serial and slower. Every module gets
its JIT decorators from here so the with/without-Numba switch lives in one place,
the same pattern ldpred3 uses.
"""

from __future__ import annotations

from numbers import Integral

import numpy as np

__all__ = ["HAVE_NUMBA", "_jit", "_jit_parallel", "_set_threads", "prange",
           "set_num_threads"]

try:
    from numba import njit as _njit, prange

    HAVE_NUMBA = True

    def _jit(func):
        return _njit(cache=True)(func)

    def _jit_parallel(func):
        return _njit(cache=True, parallel=True)(func)

    def _set_threads(ncores):
        from numba import set_num_threads
        set_num_threads(ncores)

except ImportError:  # pragma: no cover - exercised only without numba
    HAVE_NUMBA = False
    prange = range

    def _jit(func):
        return func

    def _jit_parallel(func):
        return func

    def _set_threads(ncores):
        pass


def set_num_threads(ncores):
    """Set the active thread count for subsequent Numba-parallel kernels.

    ``ncores`` must be a positive, non-boolean integer. With Numba installed,
    this masks its existing thread pool to the requested number of active
    workers and cannot exceed the maximum configured when Numba initialized.
    Setting ``NUMBA_NUM_THREADS`` before import establishes that maximum; it is
    therefore not equivalent to this runtime setter. Without Numba, the value
    is still validated but otherwise has no effect.
    """
    if isinstance(ncores, (bool, np.bool_)):
        raise TypeError("ncores must be a non-boolean integer")
    if not isinstance(ncores, Integral):
        raise TypeError("ncores must be an integer")
    if ncores <= 0:
        raise ValueError("ncores must be positive")
    _set_threads(int(ncores))
