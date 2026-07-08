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
        set_num_threads(int(ncores))

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
    """Set how many threads the Numba-parallel kernels use (no-op without Numba).

    Call before large runs to control CPU use, e.g. ``ltpred.set_num_threads(4)``.
    Equivalent to setting ``NUMBA_NUM_THREADS`` before import."""
    _set_threads(ncores)
