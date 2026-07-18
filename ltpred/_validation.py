"""Shared validation helpers for public numerical APIs."""

from __future__ import annotations

import numpy as np


def validate_bounds(lower, upper, *, context="bounds"):
    """Validate liability truncation bounds without rejecting valid infinities.

    Infinite endpoints and exact point pins are valid. NaN endpoints and
    reversed intervals are not: both can otherwise reach a fixed-coordinate
    mask or numerical kernel and silently change the statistical model.
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
