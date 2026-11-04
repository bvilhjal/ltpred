"""Shared Numba decorators and serial pure-Python fallbacks.

Parallel launches are serialised only on Numba's macOS ``workqueue`` layer,
which cannot safely handle concurrent launches. Numerical code is otherwise
identical with and without the JIT.
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
    """Set this thread's active Numba worker count to a positive integer.

    The count cannot exceed the pool maximum fixed at Numba initialisation
    (configure that with ``NUMBA_NUM_THREADS`` before import). Without Numba,
    validation still runs but the setting has no effect.
    """
    if isinstance(ncores, (bool, np.bool_)):
        raise TypeError("ncores must be a non-boolean integer")
    if not isinstance(ncores, Integral):
        raise TypeError("ncores must be an integer")
    if ncores <= 0:
        raise ValueError("ncores must be positive")
    _set_threads(int(ncores))
