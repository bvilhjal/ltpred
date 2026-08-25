"""Copy-unbind role o / target for own_status='out' (use I)."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from ._validation import validate_own_status

def _unbounded_like(value):
    """``(-inf, inf)`` with the same shape as a member's bound."""
    arr = np.asarray(value)
    if arr.ndim == 0:
        return -np.inf, np.inf
    return np.full(arr.shape, -np.inf), np.full(arr.shape, np.inf)


def _raise_unbound_proband_only(ids, n_total, what):
    """Refuse use I with nothing left to condition on (ADuLT + out)."""
    shown = ", ".join(repr(x) for x in ids[:5])
    more = "" if len(ids) <= 5 else f", ... (+{len(ids) - 5} more)"
    raise ValueError(
        f"{len(ids)} of {n_total} {what} have only the proband with "
        f"own_status='out' ({shown}{more}); with nothing to condition on "
        "the estimate would be the prior mean 0, indistinguishable from a "
        "real score. Include relatives, or drop those families.")


def _families_for_own_status(families, own_status):
    """Copy-unbind role ``o`` when ``own_status='out'``; keep the row."""
    own_status = validate_own_status(own_status)
    if own_status == "in":
        return families
    out, only_o = [], []
    for fam in families:
        roles = [m.role for m in fam.members]
        if roles and all(r == "o" for r in roles):
            only_o.append(fam.fam_id)
        members = []
        for m in fam.members:
            if m.role == "o":
                lo, hi = _unbounded_like(m.lower)
                m = replace(m, lower=lo, upper=hi, K_i=None, K_pop=None)
            members.append(m)
        out.append(replace(fam, members=members))
    if only_o:
        _raise_unbound_proband_only(only_o, len(families), "families")
    return out


def _unbind_o_column(roles, lower, upper, own_status, K_i=None, K_pop=None):
    """Rewrite the role-``o`` column to ``(-inf, inf)`` without dropping it."""
    own_status = validate_own_status(own_status)
    if own_status == "in":
        return lower, upper, K_i, K_pop
    roles = list(roles)
    if "o" not in roles:
        return lower, upper, K_i, K_pop
    if all(r == "o" for r in roles):
        raise ValueError(
            "own_status='out' with only role 'o' leaves nothing to condition "
            "on; the estimate would be the prior mean 0, indistinguishable "
            "from a real score. Include relative columns, or drop "
            "own_status='out'.")
    j = roles.index("o")
    lower = np.array(lower, copy=True)
    upper = np.array(upper, copy=True)
    lower[:, j] = -np.inf
    upper[:, j] = np.inf
    if K_i is not None:
        K_i = np.array(K_i, copy=True, dtype=float)
        K_i[:, j] = np.nan
    if K_pop is not None:
        K_pop = np.array(K_pop, copy=True, dtype=float)
        K_pop[:, j] = np.nan
    return lower, upper, K_i, K_pop
