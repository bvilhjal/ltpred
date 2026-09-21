"""Simulate families under the liability-threshold model.

Draws each family's liabilities from the liability-threshold family covariance (genetic ``g``, full
``o``, and relatives jointly multivariate normal), assigns case/control status by
thresholding the full liabilities, and packages the per-member truncation bounds
as ready-to-estimate :class:`~ltpred.family.Family` objects. Two flavours:

* ``use_age=False`` -- classic LT-FH: a single prevalence threshold ``T``, cases
  ``(T, inf)`` and controls ``(-inf, T)``. Self-consistent, handy for validating
  the estimator against the simulated truth.
* ``use_age=True`` -- age-aware simulation with generation-consistent current
  ages. Status is ``onset <= current age`` (a young person with high liability
  is a censored control, not a case). ``onset_model`` chooses how onset is
  generated; ``case_encoding`` chooses the bound written for observed cases.

The returned :class:`Simulation` also keeps the true liabilities so downstream
tests can check that the estimated genetic liability tracks the simulated one.
Port of LTFHPlus::simulate_under_LTM(_single), with coherent follow-up.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import warnings

import numpy as np
from numpy.typing import ArrayLike

from .covariance import construct_covmat_single, kinship_from_pedigree
from .family import Family, Member, families_from_columns
from .fit import _component_matrix
from ._mathfun import norm_cdf, norm_ppf
from .thresholds import (liability_threshold, convert_liability_to_aoo,
                         convert_age_to_thresh, prevalence_thresholds,
                         _convert_cir_to_age)

__all__ = ["Simulation", "simulate_under_LTM_single",
           "pedigree_birth_times", "simulate_pedigree",
           "RegisterSimulation", "simulate_register_liabilities",
           "FollowupSimulation", "simulate_followup_records",
           "MultiTraitSimulation", "simulate_under_LTM_multi"]

# Mean years older than the proband. Used only when use_age=True, so parents
# are not drawn independently younger than their children.
_AGE_GAP = {
    "o": 0.0, "s": 0.0, "mhs": 0.0, "phs": 0.0,
    "m": 29.0, "f": 31.0, "mau": 29.0, "pau": 31.0,
    "mgm": 56.0, "mgf": 58.0, "pgm": 56.0, "pgf": 58.0,
    "c": -28.0,
}

_ONSET_MODELS = ("threshold_crossing", "stochastic", "liability_dependent")
_CASE_ENCODINGS = ("pin", "interval", "lifetime")
_DEFAULT_ONSET_RHO = 0.6


def _role_stem(role):
    # Strip the numbering to get the role stem. Children carry a two-part index
    # (``c1.2``), so the dot has to go too -- ``"c1.2".rstrip("0123456789")``
    # stops at the dot and yields ``"c1."``, which never matched the ``"c"`` key
    # and silently gave simulated children the 10-60 default.
    return role.rstrip("0123456789").rstrip(".").rstrip("0123456789")


def _draw_ages(roles, n, rng):
    """Generation-consistent current ages: parents older than the proband."""
    age_o = rng.uniform(20.0, 55.0, size=n)
    ages = {}
    for role in roles:
        stem = _role_stem(role)
        gap = _AGE_GAP.get(stem, 0.0)
        sd = 0.0 if stem == "o" else (4.0 if abs(gap) >= 20.0 else 3.0)
        ages[role] = np.clip(age_o + gap + rng.normal(0.0, sd, n), 1.0, 110.0)
    return ages


@dataclass
class Simulation:
    """Output of :func:`simulate_under_LTM_single`.

    ``liabilities`` is ``(n_sim, d)`` with columns in ``roles`` order (``g``, ``o``,
    relatives); ``genetic``/``full`` are its first two columns for convenience.
    ``status`` maps each non-``g`` role to its boolean *observed* case array.
    ``ages`` / ``onset`` are per-role arrays when ``use_age=True`` (onset is
    ``inf`` for never-affected people). ``families`` is the list of one-proband
    families (bounds only) to pass to the estimator."""
    roles: list
    covmat: np.ndarray
    liabilities: np.ndarray
    status: dict
    families: list
    pop_prev: float
    ages: dict = None
    onset: dict = None
    onset_model: str = None
    case_encoding: str = None
    onset_rho: float = None

    @property
    def genetic(self):
        return self.liabilities[:, self.roles.index("g")]

    @property
    def full(self):
        return self.liabilities[:, self.roles.index("o")]


def _record_onset(aoo, onset_resolution):
    """The onset age a register would *record*, at ``onset_resolution`` years.

    Registers record onset on a grid (a year, a quarter), not to the instant, and
    every case bound is derived from that recorded value -- so the same rounding
    must apply to whichever ``case_encoding`` is in use, otherwise ``pin`` and
    ``interval`` would silently describe different observation processes.
    ``onset_resolution=None`` records the exact simulated onset. That makes the
    pin reproduce the true liability up to the round trip through the CIP curve,
    which is lossless except at the extreme upper tail, where
    :func:`~ltpred.thresholds._convert_cir_to_age` clamps the implied onset age
    at 0 and the liability cannot be recovered from it."""
    if onset_resolution is None:
        return aoo
    return round(aoo / onset_resolution) * onset_resolution


def _threshold_crossing_case_bounds(aoo, current_age, onset_resolution,
                                    case_encoding, pop_prev, mid_point, slope):
    """Bounds implied by an exact or nearest-grid crossing age.

    If a register reports ``r`` to the nearest grid width ``w``, the latent
    onset lies in ``[r - w/2, r + w/2]``.  The liability threshold decreases
    with age, so this age bin maps to the reversed liability interval
    ``[T(r + w/2), T(r - w/2)]``.  At the age-zero clamp the upper liability is
    unbounded.  ``case_encoding='interval'`` deliberately keeps only the
    conservative lower edge, preserving its one-sided semantics.
    """
    if onset_resolution is None:
        thr = float(convert_age_to_thresh(
            aoo, pop_prev=pop_prev, mid_point=mid_point, slope=slope))
        return (thr, np.inf) if case_encoding == "interval" else (thr, thr)

    recorded = _record_onset(aoo, onset_resolution)
    half_width = onset_resolution / 2.0
    # Observed case status also says onset preceded the current/censoring age.
    latest_onset = min(recorded + half_width, current_age)
    lower = float(convert_age_to_thresh(
        latest_onset, pop_prev=pop_prev, mid_point=mid_point, slope=slope))
    if case_encoding == "interval":
        return lower, np.inf

    earliest_onset = recorded - half_width
    if earliest_onset <= 0.0:
        return lower, np.inf
    upper = float(convert_age_to_thresh(
        earliest_onset, pop_prev=pop_prev, mid_point=mid_point, slope=slope))
    return lower, upper


def _validate_onset_resolution(onset_resolution):
    if onset_resolution is None:
        return None
    if isinstance(onset_resolution, (bool, np.bool_)):
        raise TypeError("onset_resolution must be a positive number or None, "
                        "not bool")
    try:
        onset_resolution = float(onset_resolution)
    except (TypeError, ValueError):
        raise TypeError(
            "onset_resolution must be a positive number or None") from None
    if not np.isfinite(onset_resolution) or onset_resolution <= 0.0:
        raise ValueError("onset_resolution must be finite and > 0, or None")
    return onset_resolution


def _validate_onset_rho(onset_rho, default=_DEFAULT_ONSET_RHO):
    if onset_rho is None:
        return float(default)
    if isinstance(onset_rho, (bool, np.bool_)):
        raise TypeError("onset_rho must be a number in [0, 1]")
    try:
        rho = float(onset_rho)
    except (TypeError, ValueError) as exc:
        raise TypeError("onset_rho must be a number in [0, 1]") from exc
    if not np.isfinite(rho) or rho < 0.0 or rho > 1.0:
        raise ValueError("onset_rho must be in [0, 1]")
    return rho


def _resolve_age_options(use_age, onset_model, case_encoding, onset_rho=None):
    if not use_age:
        if (onset_model is not None or case_encoding is not None
                or onset_rho is not None):
            raise ValueError(
                "onset_model, case_encoding and onset_rho apply only when "
                "use_age=True")
        return None, None, None
    if onset_model is None:
        onset_model = "threshold_crossing"
    if onset_model not in _ONSET_MODELS:
        raise ValueError(
            f"onset_model must be one of {_ONSET_MODELS}, got {onset_model!r}")
    if case_encoding is None:
        case_encoding = "pin" if onset_model == "threshold_crossing" else "lifetime"
    if case_encoding not in _CASE_ENCODINGS:
        raise ValueError(
            f"case_encoding must be one of {_CASE_ENCODINGS}, got {case_encoding!r}")
    if onset_model == "liability_dependent":
        onset_rho = _validate_onset_rho(onset_rho)
    elif onset_rho is not None:
        raise ValueError(
            "onset_rho applies only when onset_model='liability_dependent'")
    if case_encoding == "pin" and onset_model != "threshold_crossing":
        warnings.warn(
            f"case_encoding='pin' with onset_model={onset_model!r} pins each "
            "case at T(onset), but under this onset model onset is not "
            "deterministic given liability, so the pin records information the "
            "generating process does not contain (use 'interval' or "
            "'lifetime')", stacklevel=2)
    return onset_model, case_encoding, onset_rho


def _onset_times(liab, pop_prev, mid_point, slope, onset_model, lifetime_t, rng,
                 onset_rho=0.0):
    """Per-person onset age; ``inf`` means never affected in this lifetime."""
    n = liab.shape[0]
    onset = np.full(n, np.inf)
    if onset_model == "threshold_crossing":
        aoo = convert_liability_to_aoo(liab, pop_prev=pop_prev,
                                       mid_point=mid_point, slope=slope)
        finite = np.isfinite(aoo)
        onset[finite] = np.maximum(aoo[finite], 0.0)
        return onset
    # Lifetime case if l > T. Among those cases, onset is drawn from the CIP:
    # independently of l when onset_rho = 0 (``stochastic``), or with a
    # Gaussian copula of strength onset_rho toward the crossing quantile
    # (``liability_dependent``). Pinning at T(onset) is then *not* the true
    # liability unless onset_rho = 1.
    lifetime_case = liab > lifetime_t
    if not np.any(lifetime_case):
        return onset
    u = rng.random(int(lifetime_case.sum()))
    rho = 0.0 if onset_model == "stochastic" else float(onset_rho)
    if rho == 0.0:
        cir = np.clip(u * pop_prev, 1e-12, pop_prev * (1.0 - 1e-12))
    else:
        l_case = liab[lifetime_case]
        q_dep = np.clip((1.0 - norm_cdf(l_case)) / pop_prev, 1e-12, 1.0 - 1e-12)
        z_dep = norm_ppf(q_dep)
        z_ind = norm_ppf(np.clip(u, 1e-12, 1.0 - 1e-12))
        z = rho * z_dep + np.sqrt(max(0.0, 1.0 - rho * rho)) * z_ind
        cir = np.clip(norm_cdf(z), 1e-12, 1.0 - 1e-12) * pop_prev
    onset[lifetime_case] = _convert_cir_to_age(
        cir, pop_prev=pop_prev, mid_point=mid_point, slope=slope)
    return onset


def _stable_factor(cov):
    """A platform-stable lower-triangular ``L`` with ``L @ L.T == cov``.

    ``Generator.multivariate_normal`` factors the covariance with an SVD by
    default, and an SVD has no canonical sign: the factor -- and therefore
    every draw from a given seed -- depends on the LAPACK build underneath
    NumPy. The same seed then gives different families on macOS and Linux,
    which silently invalidates any figure quoted from a seeded run. The
    Cholesky factor of a positive-definite matrix is unique (lower triangular,
    positive diagonal), so seeded output agrees everywhere up to floating-point
    rounding.

    ``h2=1`` leaves the covariance exactly singular -- ``g`` and ``o`` are then
    the same variable -- and a singular matrix has no Cholesky factor. An
    eigendecomposition is *not* the way out: LAPACK picks an arbitrary basis
    inside a repeated eigenvalue's eigenspace, which is the same platform
    dependence in another guise, and the default seven-relative pedigree at
    ``h2=1`` does have a repeated eigenvalue. Lift the spectrum off zero
    instead and keep the unique factor. Eigen*values* are basis-independent, so
    the lift is itself stable; it is ~1e-12 of the mean variance, orders of
    magnitude below the Monte-Carlo error of anything drawn from the result.
    """
    cov = np.asarray(cov, dtype=float)
    try:
        return np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        pass
    evals = np.linalg.eigvalsh(cov)
    scale = max(float(np.trace(cov)) / cov.shape[0], 1.0)
    # eigvalsh returns eigenvalues a rounding step below zero for a singular
    # covariance; anything materially negative is a real input problem, and
    # used to surface as `multivariate_normal`'s own check_valid="warn".
    if evals.min() < -1e-8 * scale:
        warnings.warn("covariance is not positive-semidefinite; the spectrum "
                      "was lifted to make it factorable", RuntimeWarning,
                      stacklevel=2)
    base = max(0.0, -float(evals.min()))
    eye = np.eye(cov.shape[0])
    for step in (1e-12, 1e-10, 1e-8, 1e-6, 1e-4):
        try:
            return np.linalg.cholesky(cov + (base + step * scale) * eye)
        except np.linalg.LinAlgError:
            continue
    raise np.linalg.LinAlgError(
        "covariance could not be factored even after lifting its spectrum")


def simulate_under_LTM_single(fam_vec: Sequence[str] | None = ("m", "f", "s1", "mgm",
                                                              "mgf", "pgm", "pgf"),
                              n_fam: Mapping[str, int] | None = None,
                              add_ind: bool = True, h2: float = 0.5,
                              n_sim: int = 1000, pop_prev: float = 0.1,
                              use_age: bool = False, mid_point: float = 60.0,
                              slope: float = 1.0 / 8.0, seed: int | None = None,
                              onset_model: str | None = None,
                              case_encoding: str | None = None,
                              onset_resolution: float | None = 1.0,
                              onset_rho: float | None = None) -> Simulation:
    """Simulate ``n_sim`` families for a single trait.

    Builds the covariance from ``fam_vec``/``n_fam`` (``g``/``o`` prepended when
    ``add_ind``), draws liabilities, and returns a :class:`Simulation`.

    With ``use_age=False`` (default) the bounds are classic LT-FH: one lifetime
    threshold, status ``l > T``.

    With ``use_age=True`` current ages are generation-consistent, and a person
    is an observed case only if their onset age is at most their current age.
    ``onset_model`` is ``"threshold_crossing"`` (default: onset is the CIP
    inverse of true liability — the LT-FH++ convention), ``"stochastic"``
    (lifetime status at ``T``, onset drawn from the CIP independently of
    ``l`` given being a lifetime case), or ``"liability_dependent"`` (same
    lifetime status, but onset is coupled to ``l`` by a Gaussian copula of
    strength ``onset_rho``, default 0.6: higher liability tends to earlier
    onset, with residual noise). ``rho = 0`` recovers ``stochastic``;
    ``rho = 1`` recovers ``threshold_crossing`` among lifetime cases.
    ``case_encoding`` is ``"pin"`` (default for threshold-crossing),
    ``"interval"`` (age-specific ``[T(onset), inf)``), or ``"lifetime"``
    (default for stochastic and liability-dependent: ``[T, inf)``). This
    helper has one logistic CIP and does not simulate the full
    sex/birth-cohort personalisation of LT-FH++.

    ``onset_resolution`` is the grid a register is taken to **record** onset on,
    in years (default ``1.0``, whole years; ``None`` records the exact simulated
    onset). Under ``threshold_crossing``, a finite recording grid is represented
    by its full age bin. A ``"pin"`` therefore becomes the corresponding
    two-sided liability interval, while ``"interval"`` keeps its one-sided
    meaning but uses the bin's conservative lower edge. Both contain the
    generating liability. With ``onset_resolution=None``, the old exact-onset
    behavior is preserved: ``"pin"`` is a point and ``"interval"`` starts at
    that point. The age-0 clamp remains an upper-open liability interval because
    arbitrarily high liabilities map to onset age zero.

    A given ``seed`` reproduces the same families on every platform, up to
    floating-point rounding: the liabilities are drawn through a canonical
    factorisation of the covariance (:func:`_stable_factor`) rather than
    NumPy's default SVD, whose signs vary with the LAPACK build."""
    onset_resolution = _validate_onset_resolution(onset_resolution)
    onset_model, case_encoding, onset_rho = _resolve_age_options(
        use_age, onset_model, case_encoding, onset_rho)
    cov_obj = construct_covmat_single(fam_vec=fam_vec, n_fam=n_fam,
                                      add_ind=add_ind, h2=h2)
    roles = cov_obj.roles
    if "g" not in roles or "o" not in roles:
        raise ValueError("simulation requires add_ind=True (needs g and o)")
    d = len(roles)
    rng = np.random.default_rng(seed)
    # Not `rng.multivariate_normal`: its default SVD factorisation is not
    # sign-canonical, so a seed would not reproduce across LAPACK builds.
    liab = rng.standard_normal((n_sim, d)) @ _stable_factor(cov_obj.matrix).T

    t = float(liability_threshold(pop_prev))
    non_g = [r for r in roles if r != "g"]

    if use_age:
        ages = _draw_ages(non_g, n_sim, rng)
        onset = {
            r: _onset_times(liab[:, roles.index(r)], pop_prev, mid_point, slope,
                            onset_model, t, rng,
                            onset_rho=0.0 if onset_rho is None else onset_rho)
            for r in non_g
        }
        status = {r: onset[r] <= ages[r] for r in non_g}
    else:
        ages = onset = None
        status = {r: liab[:, roles.index(r)] > t for r in non_g}

    families = []
    for i in range(n_sim):
        members = []
        for r in non_g:
            is_case = bool(status[r][i])
            if not use_age:
                lower, upper = (t, np.inf) if is_case else (-np.inf, t)
            elif is_case:
                aoo = float(onset[r][i])
                if case_encoding == "lifetime":
                    lower, upper = t, np.inf
                elif onset_model == "threshold_crossing":
                    lower, upper = _threshold_crossing_case_bounds(
                        aoo, ages[r][i], onset_resolution, case_encoding,
                        pop_prev, mid_point, slope)
                else:
                    # An observed case has a finite onset at or before their
                    # current age; the guard covers only a direct call with a
                    # hand-built onset array.
                    aoo_r = (0.0 if not np.isfinite(aoo)
                             else _record_onset(aoo, onset_resolution))
                    thr = float(convert_age_to_thresh(
                        aoo_r, pop_prev=pop_prev, mid_point=mid_point,
                        slope=slope))
                    lower, upper = (thr, np.inf) if case_encoding == "interval" \
                        else (thr, thr)
            else:
                thr = float(convert_age_to_thresh(
                    ages[r][i], pop_prev=pop_prev, mid_point=mid_point,
                    slope=slope))
                lower, upper = -np.inf, thr
            members.append(Member(role=r, lower=lower, upper=upper,
                                   pid=f"fam_{i}_{r}"))
        families.append(Family(fam_id=f"fam_{i}", members=members))

    return Simulation(roles=roles, covmat=cov_obj.matrix, liabilities=liab,
                      status=status, families=families, pop_prev=pop_prev,
                      ages=ages, onset=onset, onset_model=onset_model,
                      case_encoding=case_encoding, onset_rho=onset_rho)


# ---------------------------------------------------------------------------
# Population-register simulation (use I / use III input, with known truth)
# ---------------------------------------------------------------------------

def simulate_pedigree(rng: np.random.Generator, n_founder_pairs: int = 150,
                      gens: int = 3,
                      remarry: float = 0.10) -> tuple[list, list, list]:
    """Simulate a multi-generation pedigree as trio columns.

    Returns ``(ids, father, mother)`` — the three columns
    :func:`~ltpred.pipeline.estimate_liabilities`,
    :func:`~ltpred.pedigree.build_parent_graph` and
    :func:`~ltpred.covariance.kinship_from_pedigree` all take directly.
    Founders have ``None`` parents. ``gens`` generations are built from
    ``n_founder_pairs`` unrelated founder couples; each couple has 2–4 children,
    and with probability ``remarry`` a partner forms a second union, which is
    what produces half-siblings. Recorded full siblings are never paired.

    ``rng`` is a :class:`numpy.random.Generator`; pass
    ``np.random.default_rng(seed)`` for a reproducible pedigree."""
    ids, father, mother = [], [], []

    def add(f, m):
        pid = f"p{len(ids)}"
        ids.append(pid)
        father.append(f)
        mother.append(m)
        return pid

    couples = [(add(None, None), add(None, None)) for _ in range(n_founder_pairs)]
    prev_children = []
    for g in range(gens):
        if g > 0:
            pool = list(prev_children)
            rng.shuffle(pool)
            couples = []
            i = 0
            while i + 1 < len(pool):
                a, b = pool[i], pool[i + 1]
                i += 2
                # avoid mating recorded siblings (same recorded parent)
                fa, ma = father[ids.index(a)], mother[ids.index(a)]
                fb, mb = father[ids.index(b)], mother[ids.index(b)]
                if fa is not None and (fa in (fb, mb) or ma in (fb, mb)):
                    continue
                couples.append((a, b))
        next_children = []
        for fa, mo in couples:
            for _ in range(int(rng.integers(2, 5))):
                next_children.append(add(fa, mo))
            if rng.uniform() < remarry:           # second union -> half-sibs
                mate = add(None, None)
                next_children.append(add(fa, mate))
        prev_children = next_children
    return ids, father, mother


def pedigree_birth_times(ids: Sequence, father: Sequence, mother: Sequence, *,
                         base_birth_year: float = 1920.0,
                         generation_years: float = 30.0) -> np.ndarray:
    """Assign generation-coherent calendar birth times to a pedigree.

    Co-parents are placed in the same generation (a union-find over couples) and
    every child one generation later than its recorded parents, so birth times
    are ``base_birth_year + generation_years * generation``. Use this to obtain
    the ``birth_time`` column :func:`~ltpred.pipeline.estimate_liabilities`
    needs for calendar-time (use I) censoring when the pedigree itself carries no
    dates.

    Raises ``ValueError`` on a generational cycle. Unlike
    :func:`simulate_pedigree` this consumes no randomness, so it is deterministic
    for a given pedigree."""
    n = len(ids)
    pos = {pid: i for i, pid in enumerate(ids)}
    representative = list(range(n))

    def find(i):
        while representative[i] != i:
            representative[i] = representative[representative[i]]
            i = representative[i]
        return i

    def union(i, j):
        left, right = find(i), find(j)
        if left != right:
            representative[right] = left

    for fa, mo in zip(father, mother):
        if fa in pos and mo in pos:
            union(pos[fa], pos[mo])

    component = np.array([find(i) for i in range(n)], dtype=np.intp)
    members = {root: [] for root in set(component.tolist())}
    for i, root in enumerate(component):
        members[root].append(i)
    children = {root: set() for root in members}
    indegree = {root: 0 for root in members}
    for child, (fa, mo) in enumerate(zip(father, mother)):
        child_root = int(component[child])
        for parent_id in (fa, mo):
            if parent_id not in pos:
                continue
            parent_root = int(component[pos[parent_id]])
            if child_root != parent_root and child_root not in children[parent_root]:
                children[parent_root].add(child_root)
                indegree[child_root] += 1

    frontier = [root for root, count in indegree.items() if count == 0]
    generation = {root: 0 for root in frontier}
    visited = 0
    while frontier:
        root = frontier.pop()
        visited += 1
        for child_root in children[root]:
            generation[child_root] = max(
                generation.get(child_root, 0), generation[root] + 1)
            indegree[child_root] -= 1
            if indegree[child_root] == 0:
                frontier.append(child_root)
    if visited != len(members):
        raise ValueError("pedigree contains a generational cycle")
    return np.array([
        base_birth_year + generation_years * generation[int(root)]
        for root in component
    ])


@dataclass
class RegisterSimulation:
    """Output of :func:`simulate_register_liabilities`.

    ``ids``/``father``/``mother``/``status``/``age``/``birth_time`` are the
    columns :func:`~ltpred.pipeline.estimate_liabilities` consumes, aligned by
    position. ``onset`` is the age at diagnosis (``inf`` if never affected);
    ``genetic`` is the **true** standardised additive genetic liability and
    ``residual_var`` the matching residual variance, so an estimate can be scored
    against the truth rather than only against itself."""
    ids: list
    father: list
    mother: list
    status: np.ndarray
    age: np.ndarray
    onset: np.ndarray
    birth_time: np.ndarray
    genetic: np.ndarray
    residual_var: np.ndarray


def simulate_register_liabilities(rng: np.random.Generator, ids: Sequence,
                                  father: Sequence, mother: Sequence, *,
                                  h2: float, cip_ages: ArrayLike,
                                  cip_values: ArrayLike,
                                  eval_age: float) -> RegisterSimulation:
    """Simulate one **consistent** population liability field, then records.

    Draws raw genetic liabilities ``G ~ N(0, h2 * A_full)`` once for the whole
    pedigree plus independent residuals ``E``, then divides both the full
    liability ``G + E`` and its genetic coordinate ``G`` by each person's raw
    full-liability SD ``sqrt(h2 * A_ii + (1 - h2))``. That division is a no-op
    for non-inbred people and is what keeps an inbred person (``A_ii > 1``) on
    the unit-variance liability scale the public kinship path assumes.

    Drawing the population once — rather than each proband's pedigree separately
    — is the point: a person shared between two probands' pedigrees then has one
    status and one genetic value, so status and ``g`` stay correlated as they are
    in reality. Independent per-proband draws would fix the shared person's
    status from one draw and the proband's ``g`` from another, silently
    decorrelating them.

    Onset follows the threshold-crossing model against ``cip_values``: a person
    is a lifetime case when ``1 - Phi(liability)`` is at most the curve's horizon
    value, with onset the age at which the curve reaches that probability.
    Everyone is observed through ``eval_age`` unless diagnosed earlier, so
    ``status`` is *observed* case at ``eval_age`` and ``age`` is onset for cases
    and ``eval_age`` otherwise.

    ``cip_ages``/``cip_values`` are a strictly increasing cumulative-incidence
    curve (see :func:`~ltpred.cip.kaplan_meier_cip`). ``rng`` is a
    :class:`numpy.random.Generator`; the pedigree usually comes from
    :func:`simulate_pedigree` and ``birth_time`` from
    :func:`pedigree_birth_times`, which this function calls for you.

    A given ``rng`` reproduces the same register on every platform, up to
    floating-point rounding: the population liability field is drawn through
    the canonical factorisation of :func:`_stable_factor`, not NumPy's
    sign-arbitrary SVD."""
    cip_ages = np.asarray(cip_ages, dtype=float)
    cip_values = np.asarray(cip_values, dtype=float)
    if cip_ages.shape != cip_values.shape:
        raise ValueError(
            f"cip_ages and cip_values must have the same shape; got "
            f"{cip_ages.shape} and {cip_values.shape}")
    if cip_values.size < 2 or np.any(np.diff(cip_values) <= 0.0):
        raise ValueError("cip_values must have at least two strictly increasing "
                         "entries to be invertible by interpolation")
    _, A_full = kinship_from_pedigree(ids, father, mother)
    n = len(ids)
    # Not `rng.multivariate_normal`: its default SVD factorisation is not
    # sign-canonical, so a seed would not reproduce across LAPACK builds.
    raw_genetic = rng.standard_normal(n) @ _stable_factor(h2 * A_full).T
    residual = rng.standard_normal(n) * np.sqrt(1.0 - h2)
    scale = np.sqrt(h2 * np.diag(A_full) + (1.0 - h2))
    genetic = raw_genetic / scale
    liability = (raw_genetic + residual) / scale
    residual_var = (1.0 - h2) / scale ** 2

    onset = np.full(n, np.inf)
    need = 1.0 - norm_cdf(liability)
    event = need <= cip_values[-1]
    # Tiny positive floor avoids a zero-length follow-up event in the
    # Aalen-Johansen estimate for the rare extreme liability.
    onset[event] = np.maximum(
        np.interp(need[event], cip_values, cip_ages), 1e-9)
    status = onset <= eval_age
    age = np.where(status, onset, float(eval_age))
    birth_time = pedigree_birth_times(ids, father, mother)
    return RegisterSimulation(
        ids=list(ids), father=list(father), mother=list(mother), status=status,
        age=age, onset=onset, birth_time=birth_time, genetic=genetic,
        residual_var=residual_var)


# ---------------------------------------------------------------------------
# Follow-up records for cumulative-incidence (step 2) estimation
# ---------------------------------------------------------------------------

@dataclass
class FollowupSimulation:
    """Output of :func:`simulate_followup_records`.

    ``age_entry``/``age_exit``/``event`` are the three columns
    :func:`~ltpred.cip.aalen_johansen_cip` and
    :func:`~ltpred.cip.kaplan_meier_cip` consume, with ``event`` coded
    0 = administratively censored, 1 = diagnosed, 2 = died. ``onset`` and
    ``death_age`` are the underlying latent times (``inf`` when the event never
    happens), kept so an estimate can be checked against the generating curve.
    ``n_dropped`` counts records removed because ``age_exit <= age_entry``."""
    age_entry: np.ndarray
    age_exit: np.ndarray
    event: np.ndarray
    onset: np.ndarray
    death_age: np.ndarray
    n_dropped: int


def simulate_followup_records(rng: np.random.Generator, n: int, *,
                              pop_prev: float, mid_point: float, slope: float,
                              mortality: bool = False,
                              gompertz_log_a: float = -9.0,
                              gompertz_b: float = 0.085,
                              birth_year_range: tuple[float, float] = (
                                  1900.0, 2000.0),
                              admin_end: float = 2015.0,
                              register_start_year: float | None = None
                              ) -> FollowupSimulation:
    """Simulate registry follow-up records with a known incidence curve.

    Each person gets a birth year uniform on ``birth_year_range``, a lifetime
    case indicator with probability ``pop_prev``, and — if a case — an onset age
    from the inverse CDF of the logistic cumulative-incidence curve with
    ``mid_point``/``slope`` (the same parameterisation
    :func:`~ltpred.thresholds.thresholds_from_cip` expects). With
    ``register_start_year`` given, follow-up is left-truncated: nobody is
    observed before that calendar year, which is what a register opened mid-life
    looks like. Everyone is administratively censored at ``admin_end``.

    With ``mortality=True`` a competing Gompertz death time is drawn with hazard
    ``exp(gompertz_log_a + gompertz_b * age)`` and coded ``event = 2`` when it
    precedes onset. This is the case that makes the difference between the two
    estimators visible: :func:`~ltpred.cip.kaplan_meier_cip` treats death as
    independent censoring and therefore **overestimates** incidence, while
    :func:`~ltpred.cip.aalen_johansen_cip` targets the crude cumulative
    incidence in the presence of the competing risk.

    Records with ``age_exit <= age_entry`` are dropped and counted in
    ``n_dropped``; they are people whose whole follow-up falls outside the
    observation window."""
    birth_year = rng.uniform(birth_year_range[0], birth_year_range[1], n)
    entry_age = np.zeros(n)
    if register_start_year is not None:
        entry_age = np.maximum(0.0, register_start_year - birth_year)

    u = rng.uniform(size=n)
    is_case = u < pop_prev
    onset = np.full(n, np.inf)
    v = rng.uniform(size=int(is_case.sum()))
    onset[is_case] = mid_point + np.log(v / (1 - v)) / slope
    onset = np.maximum(onset, 0.0)

    if mortality:
        w = rng.uniform(size=n)
        death_age = (np.log(1.0 + gompertz_b * (-np.log(w))
                            / np.exp(gompertz_log_a)) / gompertz_b)
    else:
        death_age = np.full(n, np.inf)

    admin = np.maximum(entry_age, admin_end - birth_year)
    exit_age = np.minimum(np.minimum(onset, death_age), admin)
    event = np.zeros(n, dtype=int)
    event[(onset <= exit_age) & np.isfinite(onset)] = 1
    event[(death_age <= exit_age) & (onset > death_age)] = 2
    ok = exit_age > entry_age
    return FollowupSimulation(
        age_entry=entry_age[ok], age_exit=exit_age[ok], event=event[ok],
        onset=onset[ok], death_age=death_age[ok],
        n_dropped=int(n - int(ok.sum())))


# ---------------------------------------------------------------------------
# Multi-trait families under the liability-threshold model
# ---------------------------------------------------------------------------

_MULTI_ROLES = ("o", "m", "f", "s1", "s2")
_MULTI_FATHER = ("f", None, None, "f", "f")
_MULTI_MOTHER = ("m", None, None, "m", "m")


@dataclass
class MultiTraitSimulation:
    """Output of :func:`simulate_under_LTM_multi`.

    ``liabilities`` is ``(n_families, d, n_traits)`` with ``roles`` ordering the
    ``d`` people; ``status`` is the ``(n_families * d, n_traits)`` person-major
    boolean case matrix the bounds were built from; ``families`` is the list to
    pass to :func:`~ltpred.fit.fit_pairwise_multi`. ``truth`` echoes the
    generating variance parameters, so a fit can be scored against them."""
    roles: list
    phen_names: tuple
    liabilities: np.ndarray
    status: np.ndarray
    families: list
    pop_prev: np.ndarray
    covmat: np.ndarray
    truth: dict


def simulate_under_LTM_multi(n_families: int = 1000, *,
                             seed: int | None = None,
                             roles: Sequence[str] = _MULTI_ROLES,
                             father: Sequence = _MULTI_FATHER,
                             mother: Sequence = _MULTI_MOTHER,
                             h2: ArrayLike = (0.35, 0.40), rg: float = 0.50,
                             sib_shared: ArrayLike = (
                                 (0.15, 0.075), (0.075, 0.15)),
                             couple_shared: ArrayLike = (
                                 (0.10, 0.02), (0.02, 0.10)),
                             residual_re: float = -0.35,
                             pop_prev: ArrayLike = (0.10, 0.20),
                             phen_names: Sequence[str] = ("trait_1", "trait_2")
                             ) -> MultiTraitSimulation:
    """Simulate ``n_families`` independent families with two observed traits.

    Builds the ``d * n_traits`` covariance as an explicit Kronecker sum over one
    relationship matrix per variance component —
    ``kron(G, A) + kron(S, C) + kron(M_mat, M) + kron(E, I)`` — where ``A`` is
    the additive relationship from
    :func:`~ltpred.covariance.kinship_from_pedigree` and ``C``/``M`` are the
    shared-sibship and shared-couple indicators from the same component matrices
    :func:`~ltpred.fit.fit_variance_components` uses. Liabilities are drawn once
    per family, thresholded per trait at ``pop_prev``, and packaged as
    :class:`~ltpred.family.Family` objects with one interval per trait.

    ``h2`` is the per-trait additive variance, ``rg`` the genetic correlation,
    ``sib_shared``/``couple_shared`` the per-trait shared-sibship and
    shared-couple covariance matrices, and ``residual_re`` the residual
    environmental correlation. The defaults are the design
    ``docs/vignette.md`` and the tutorial document, so
    ``simulate_under_LTM_multi(n_families=3000, seed=1)`` reproduces that
    cohort bit for bit.

    Each trait's liability has unit variance (the four component diagonals sum
    to 1), so the implied full-liability correlation between traits is
    ``G12 + S12 + M12 + E12`` — 0.1511 for the defaults, echoed in
    ``truth["target_trait_corr"]``. The correlation of the *binary* statuses is
    much smaller, because thresholding at 0.10 and 0.20 discards most of it.
    Pass ``roles`` together with matching ``father``/``mother`` to use a
    different family structure."""
    roles = list(roles)
    d = len(roles)
    h2 = np.asarray(h2, dtype=float)
    n_traits = h2.size
    pop_prev = np.asarray(pop_prev, dtype=float)
    if pop_prev.size != n_traits:
        raise ValueError(
            f"pop_prev must have one prevalence per trait ({n_traits}); got "
            f"{pop_prev.size}")
    _, a = kinship_from_pedigree(roles, father, mother)
    c = np.asarray(_component_matrix(roles, "C"), dtype=float)
    m = np.asarray(_component_matrix(roles, "M"), dtype=float)

    g = rg * np.sqrt(np.outer(h2, h2))
    np.fill_diagonal(g, h2)
    s = np.asarray(sib_shared, dtype=float)
    t = np.asarray(couple_shared, dtype=float)
    for name, mat in (("sib_shared", s), ("couple_shared", t)):
        if mat.shape != (n_traits, n_traits):
            raise ValueError(
                f"{name} must be ({n_traits}, {n_traits}); got {mat.shape}")
    e = np.diag(1.0 - np.diag(g + s + t))
    e[0, 1] = e[1, 0] = residual_re * np.sqrt(e[0, 0] * e[1, 1])
    sigma = (np.kron(g, a) + np.kron(s, c) + np.kron(t, m)
             + np.kron(e, np.eye(d)))

    rng = np.random.default_rng(seed)
    latent = rng.standard_normal((n_families, n_traits * d))
    # `_stable_factor` is `cholesky` for a positive-definite sigma -- the usual
    # case, numerically identical -- and lifts the spectrum instead of failing
    # when a component choice makes it exactly singular.
    latent = latent @ _stable_factor(sigma).T
    # Covariance coordinates are trait-major; family rows are people.
    latent = latent.reshape(n_families, n_traits, d).transpose(0, 2, 1)
    person = latent.reshape(-1, n_traits)
    status = person > -norm_ppf(pop_prev)
    bounds = [prevalence_thresholds(status[:, p], pop_prev=float(pop_prev[p]))
              for p in range(n_traits)]
    lower = np.column_stack([lo for lo, hi in bounds])
    upper = np.column_stack([hi for lo, hi in bounds])
    families = families_from_columns(
        fam_id=np.repeat(np.arange(n_families), d),
        role=np.tile(roles, n_families), lower=lower, upper=upper,
        pid=np.arange(d * n_families))
    return MultiTraitSimulation(
        roles=roles, phen_names=tuple(phen_names), liabilities=latent,
        status=status, families=families, pop_prev=pop_prev, covmat=sigma,
        truth={"h2": h2.copy(), "rg": float(rg), "sib_shared": s.copy(),
               "couple_shared": t.copy(), "residual_re": float(residual_re),
               "target_trait_corr": float(g[0, 1] + s[0, 1] + t[0, 1]
                                          + e[0, 1])})

