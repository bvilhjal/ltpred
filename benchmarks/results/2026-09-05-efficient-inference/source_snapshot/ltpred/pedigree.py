"""Discovering relatives from parent-offspring (trio) records.

The role grammar and :func:`~ltpred.covariance.kinship_from_pedigree` start
from a **specified** pedigree. Register data instead arrive as population-scale
parent-offspring records (``id, father, mother`` per person), and the pedigree
must be *discovered*: which people are this proband's relatives, and how are
they connected? This module builds the parent-child graph once
(:func:`build_parent_graph`) and then extracts each proband's pedigree by
breadth-first traversal (:func:`extract_pedigree`) — the graph-based relative
extraction of Pedersen
et al. (2025, *Front Genet*): a relative is anyone within ``max_degree``
relationship degrees of the proband.

Degree conventions follow the standard relationship-degree scale (and
Pedersen et al. 2025): **first-degree** relatives (parents, children, full
siblings) are graph distance 1; **second-degree** (grandparents,
grandchildren, half-siblings, aunts/uncles, nieces/nephews) distance 2;
**third-degree** (first cousins, great-grandparents) distance 3. This is
achieved by adding explicit edges between full siblings (two shared parents)
to the parent-child graph, so traversal distance equals the standard degree
rather than the raw meiotic count. Mates enter through shared children: the
proband's own mate is graph distance 2 (proband -> child -> mate), while a
relative's mate is distance 3 via the relative's child; they are genetically
unrelated to the proband but belong to the pedigree (and to the
spousal-environment component of `fit_variance_components`).

The extracted :class:`Pedigree` carries ``ids``/``father``/``mother`` in the
exact form :func:`~ltpred.covariance.kinship_from_pedigree` consumes (a parent
outside the records is a founder), plus each member's relationship ``degree``
from the proband and a ``closure_only`` mask. Extraction then closes on **all
recorded ancestors** of the set, so the pedigree's kinship is exact for every
member pair (equal to the full-population kinship restricted to the set); the
degree limit truncates only which *relatives* are included, never the kinship
among them. ``closure_only`` distinguishes those structural ancestors from the
people reached within ``max_degree`` whose observations may enter an analysis.
Deterministic ordering: the proband first, then by (degree, id), so identical
inputs give identical pedigrees regardless of record order.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

__all__ = ["ParentGraph", "Pedigree", "build_parent_graph",
           "extract_pedigree"]


@dataclass
class ParentGraph:
    """Validated parent-child index of a population's trio records.

    ``ids`` are the unique person ids; ``sire``/``dam`` give each person's
    father/mother as indices into ``ids`` (or -1 for unknown/founder).
    ``children[i]`` lists the indices of person ``i``'s recorded children;
    ``sibs[i]`` the indices of person ``i``'s **full** siblings (two shared
    parents) — the sib edges that make traversal distance equal the standard
    relationship degree (Pedersen et al. 2025). ``n_unresolved_parents``
    counts non-null father/mother values that matched no id; declared-unknown
    parents (``None``/``nan``) are not counted."""
    ids: list
    sire: list
    dam: list
    children: list
    sibs: list
    index: dict
    n_unresolved_parents: int


@dataclass
class Pedigree:
    """One proband's extracted pedigree.

    ``ids``/``father``/``mother`` feed
    :func:`~ltpred.covariance.kinship_from_pedigree` directly (parents outside
    the extracted set are founders). ``degree[i]`` is the relationship-degree
    distance of member ``i`` from the proband (0 = proband).
    ``closure_only[i]`` is true when the member was added only to preserve exact
    kinship, after the ``max_degree`` traversal; it is an explicit warning that
    the person's diagnosis is outside the requested observation set unless a
    caller deliberately opts in. Ordering is deterministic: proband first, then
    by (degree, id)."""
    proband: object
    ids: list
    father: list
    mother: list
    degree: np.ndarray
    closure_only: np.ndarray


def build_parent_graph(ids: Sequence, father: Sequence,
                       mother: Sequence) -> ParentGraph:
    """Index population trio records for traversal.

    ``ids`` must be unique; a parent not among ``ids`` (``None``, ``nan``, or
    any unlisted value) is an unknown founder. Unlisted **non-null** values are
    counted in ``n_unresolved_parents``: a register boundary makes some of
    them inevitable, but the count is what lets a caller distinguish that
    boundary from an id-format mismatch or a failed join. Raises on duplicate
    ids and on a person recorded as their own parent. (Cycle detection -- a
    person being their own ancestor -- happens in
    :func:`~ltpred.covariance.kinship_from_pedigree`, which raises on it.)
    """
    ids = list(ids)
    father = list(father)
    mother = list(mother)
    n = len(ids)
    if not (len(father) == len(mother) == n):
        raise ValueError("ids, father and mother must share length")
    if len(set(ids)) != n:
        raise ValueError("ids must be unique")
    index = {pid: i for i, pid in enumerate(ids)}

    n_unresolved = 0

    def _idx(p):
        nonlocal n_unresolved
        if p is None or (isinstance(p, float) and np.isnan(p)):
            return -1
        j = index.get(p, -1)
        if j == -1:
            n_unresolved += 1
        return j

    sire = [_idx(p) for p in father]
    dam = [_idx(p) for p in mother]
    children = [[] for _ in range(n)]
    for i in range(n):
        if sire[i] == i or dam[i] == i:
            raise ValueError(f"individual {ids[i]!r} is its own parent")
        if sire[i] != -1:
            children[sire[i]].append(i)
        if dam[i] != -1:
            children[dam[i]].append(i)
    # full-sibling edges: two shared parents (both known)
    sibs = [set() for _ in range(n)]
    by_parents = {}
    for i in range(n):
        if sire[i] != -1 and dam[i] != -1:
            by_parents.setdefault((sire[i], dam[i]), []).append(i)
    for group in by_parents.values():
        for i in group:
            sibs[i].update(j for j in group if j != i)
    sibs = [sorted(s) for s in sibs]
    return ParentGraph(ids=ids, sire=sire, dam=dam, children=children,
                       sibs=sibs, index=index, n_unresolved_parents=n_unresolved)


def extract_pedigree(graph: ParentGraph, proband: object,
                     max_degree: int = 2) -> Pedigree:
    """One proband's pedigree by breadth-first traversal of ``graph``.

    Visits every person within ``max_degree`` relationship-degrees of
    ``proband`` (see the module docstring for the degree conventions), then
    closes on **all recorded ancestors** of the extracted set. Kinship flows
    through common ancestors, so with every ancestor present the extracted
    pedigree's kinship is **exact**: it equals the full-population kinship
    restricted to the extracted members, inbreeding included. The degree
    limit truncates only *which* relatives are included (lateral and
    descendant links beyond ``max_degree``), never the kinship among those
    included. ``closure_only`` marks ancestors added after the traversal, so a
    downstream scorer can retain them for kinship without automatically using
    their diagnoses. A parent not present in the records at all is an unknown
    founder, as :func:`~ltpred.covariance.kinship_from_pedigree` expects.
    """
    if isinstance(max_degree, bool) or not isinstance(max_degree,
                                                       (int, np.integer)):
        raise TypeError("max_degree must be an integer")
    if max_degree < 1:
        raise ValueError("max_degree must be at least 1")
    start = graph.index.get(proband)
    if start is None:
        raise ValueError(f"proband {proband!r} is not in the parent graph")

    degree = {start: 0}
    frontier = [start]
    for step in range(1, max_degree + 1):
        nxt = []
        for i in frontier:
            for j in graph.children[i]:
                if j not in degree:
                    degree[j] = step
                    nxt.append(j)
            for j in (graph.sire[i], graph.dam[i]):
                if j != -1 and j not in degree:
                    degree[j] = step
                    nxt.append(j)
            for j in graph.sibs[i]:
                if j not in degree:
                    degree[j] = step
                    nxt.append(j)
        frontier = nxt

    traversed = set(degree)

    # Ancestral closure: recursively include every extracted member's recorded
    # ancestors. Kinship between any two people flows through their common
    # ancestors, so with all ancestors present the extracted pedigree's
    # kinship is EXACT -- it equals the full-population kinship restricted to
    # the extracted members (the tabular method in kinship_from_pedigree
    # computes inbreeding correctly whenever the connecting paths are present).
    # Without this, boundary members who are related to each other (e.g. two
    # closure parents who are siblings) would be treated as unrelated
    # founders. The degree limit thus truncates only *lateral/descendant*
    # links (which relatives are included), never the kinship among those
    # included.
    # Relax rather than first-write: an ancestor reachable by two routes (say a
    # grandparent who is also a more distant ancestor on the other side) must
    # keep the shortest relationship-degree distance, which is what ``degree``
    # documents and what pipeline diagnostics summarise. Assigning on first
    # visit instead recorded whichever route the traversal happened to reach.
    queue = list(degree)
    while queue:
        i = queue.pop()
        step = degree[i] + 1
        for j in (graph.sire[i], graph.dam[i]):
            if j != -1 and (j not in degree or step < degree[j]):
                degree[j] = step
                queue.append(j)

    members = sorted(degree, key=lambda j: (degree[j], str(graph.ids[j])))
    members.remove(start)
    members = [start] + members
    id_at = [graph.ids[j] for j in members]
    pos = {j: k for k, j in enumerate(members)}

    def _parent(pidx):
        return graph.ids[pidx] if pidx != -1 and pidx in pos else None

    father = [_parent(graph.sire[j]) for j in members]
    mother = [_parent(graph.dam[j]) for j in members]
    return Pedigree(
        proband=proband,
        ids=id_at,
        father=father,
        mother=mother,
        degree=np.array([degree[j] for j in members]),
        closure_only=np.array([j not in traversed for j in members], dtype=bool),
    )
