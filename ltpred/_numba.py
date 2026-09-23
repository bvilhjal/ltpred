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

import functools
import threading
from numbers import Integral

import numpy as np

__all__ = ["HAVE_NUMBA", "_jit", "_jit_parallel", "_set_threads", "prange",
           "set_num_threads"]

try:
    import numba
    from numba import njit as _njit, prange

    HAVE_NUMBA = True

    def _jit(func):
        return _njit(cache=True)(func)

    _launch_lock = threading.Lock()
    _layer = []

    def _serialise_launch():
        # Numba's fallback ``workqueue`` layer (all a macOS pip wheel gets: no
        # OpenMP, no TBB) aborts the process when two Python threads launch
        # parallel kernels at once. The layer is fixed at the first launch.
        if not _layer:
            try:
                _layer.append(numba.threading_layer())
            except ValueError:          # nothing launched yet: lock this one
                return True
        return _layer[0] == "workqueue"

    def _jit_parallel(func):
        kernel = _njit(cache=True, parallel=True)(func)

        @functools.wraps(func)
        def launch(*args, **kwargs):
            if _serialise_launch():
                with _launch_lock:
                    return kernel(*args, **kwargs)
            return kernel(*args, **kwargs)

        launch.py_func = func
        return launch

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


def set_num_threads(ncores: int) -> None:
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
