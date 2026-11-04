"""Shared validation helpers for public numerical APIs."""

from __future__ import annotations

import numpy as np

from ._mathfun import norm_ppf


def validate_binary(values, *, name="values", ndim=None):
    """Return Boolean data after rejecting missing, sentinel, and non-binary codes."""
    array = np.asarray(values)
    if ndim is not None and array.ndim != ndim:
        raise ValueError(f"{name} must be a {ndim}-dimensional array")
    if array.dtype.kind == "b":
        return array
    if array.dtype.kind not in "iuf" or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only boolean or 0/1 values")
    if np.any((array != 0) & (array != 1)):
        raise ValueError(f"{name} must contain only boolean or 0/1 values")
    return array.astype(bool)


def validate_bounds(lower, upper, *, context="bounds"):
    """Validate liability truncation bounds without rejecting valid infinities.

    Infinite endpoints and finite point pins are valid. NaN endpoints,
    reversed intervals, and coincident *infinite* bounds are not: they can
    otherwise reach a fixed-coordinate mask or numerical kernel and silently
    change the statistical model.
    """
    lower = np.asarray(lower)
    upper = np.asarray(upper)
    if lower.shape != upper.shape:
        raise ValueError(
            f"{context}: lower and upper must have the same shape; got "
            f"{lower.shape} and {upper.shape}")
    if np.any(np.isnan(lower)) or np.any(np.isnan(upper)):
        raise ValueError(f"{context}: lower and upper must not contain NaN")
    reversed_bounds = upper < lower
    if np.any(reversed_bounds):
        locations = np.argwhere(reversed_bounds)
        if reversed_bounds.ndim == 1:
            locations = locations[:, 0].tolist()
        else:
            locations = [tuple(index) for index in locations.tolist()]
        raise ValueError(
            f"{context}: upper must be >= lower at every coordinate; reversed "
            f"bounds at indices {locations}")
    infinite_pin = (lower == upper) & ~np.isfinite(lower)
    if np.any(infinite_pin):
        raise ValueError(
            f"{context}: a point pin (lower == upper) must be finite; a "
            "coincident infinite bound is not an observation")


def validate_mixture_inputs(K_i, K_pop, *, expected_shape=None,
                            require_pair=False, lower=None, upper=None,
                            context="mixture inputs"):
    """Validate cumulative-incidence pairs for the PA censoring mixture.

    ``NaN``/``NaN`` marks unused coordinates. Supplied pairs must be finite
    with ``0 <= K_i <= K_pop < 1`` and positive ``K_pop``. With bounds, pairs
    are allowed only on unpinned controls with finite upper bounds and a lower
    bound below the lifetime threshold. ``require_pair`` demands at least one
    active coordinate. Returns normalised arrays; ``None`` means all NaN.
    """
    shape = None if expected_shape is None else tuple(expected_shape)

    def _coerce(value):
        if value is None:
            return np.full((), np.nan) if shape is None else np.full(shape, np.nan)
        array = np.asarray(value)
        if not np.issubdtype(array.dtype, np.floating):
            try:
                array = np.asarray(value, dtype=float)
            except (TypeError, ValueError):
                raise TypeError(f"{context}: K_i and K_pop must be numeric") from None
        return array

    K_i = _coerce(K_i)
    K_pop = _coerce(K_pop)
    if K_i.shape != K_pop.shape:
        raise ValueError(
            f"{context}: K_i and K_pop must have the same shape; got "
            f"{K_i.shape} and {K_pop.shape}")
    if shape is not None and K_i.shape != shape:
        raise ValueError(
            f"{context}: K_i and K_pop must have shape {shape}; got {K_i.shape}")

    unused = np.isnan(K_i) & np.isnan(K_pop)
    supplied = ~unused
    if np.any(supplied & (~np.isfinite(K_i) | ~np.isfinite(K_pop))):
        raise ValueError(
            f"{context}: every supplied coordinate requires finite K_i and K_pop; "
            "use NaN/NaN where the mixture is unused")
    if np.any(supplied & ((K_i < 0.0) | (K_i > K_pop) |
                          (K_pop <= 0.0) | (K_pop >= 1.0))):
        raise ValueError(
            f"{context}: every supplied pair must satisfy "
            "0 <= K_i <= K_pop < 1 with K_pop > 0")
    if (lower is None) != (upper is None):
        raise ValueError(f"{context}: lower and upper must be given together")
    if lower is not None:
        lower = np.asarray(lower)
        upper = np.asarray(upper)
        if lower.shape != K_i.shape or upper.shape != K_i.shape:
            raise ValueError(
                f"{context}: lower and upper must have shape {K_i.shape}; got "
                f"{lower.shape} and {upper.shape}")
        if np.any(supplied & (lower == upper)):
            raise ValueError(
                f"{context}: K_i/K_pop on a pinned row (lower == upper) is not "
                "meaningful -- a pinned observation needs no censoring mixture; "
                "use NaN/NaN there")
        if np.any(supplied & (upper == np.inf)):
            raise ValueError(
                f"{context}: K_i/K_pop must be NaN/NaN on observed-case or "
                "unbounded rows (upper == +inf); the censoring mixture is only "
                "defined for controls with a finite upper bound")
        active = supplied & np.isfinite(upper)
        split = -norm_ppf(K_pop)
        invalid_active = active & (lower >= split)
        if np.any(invalid_active):
            raise ValueError(
                f"{context}: every active censored-control row must satisfy "
                "lower < Phi^-1(1 - K_pop)")
    else:
        active = supplied
    if require_pair and not np.any(active):
        raise ValueError(
            f"{context}: use_mixture=True requires at least one active "
            "censored-control K_i/K_pop pair")
    return K_i, K_pop
