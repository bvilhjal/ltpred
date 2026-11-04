"""Discover pedigrees from population parent-offspring records.

`build_parent_graph` indexes parents, children and full-sibling edges;
`extract_pedigree` performs a breadth-first search to ``max_degree`` and then
adds every recorded ancestor needed for exact kinship. Full-sib edges make
parents, children and full siblings distance one; grandparents, half-sibs and
avuncular relatives distance two; first cousins distance three.

Closure-only ancestors retain their shortest degree and are marked so their
diagnoses need not enter the observation set. Output order is proband first,
then ``(degree, str(id))``, independent of record order.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .covariance import _parent_links

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
    parents (``None``, NaN, pandas NA/NaT or empty/NA strings) are not counted.
    Zero/``"0"`` is unknown only when it is not an explicitly listed person id."""
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
    `ltpred.covariance.kinship_from_pedigree` directly; a parent is ``None``
    only when unrecorded, since closure includes every recorded parent.
    ``degree[i]`` is the relationship-degree distance of member ``i`` from the
    proband (0 = proband; closure ancestors carry their shortest ancestral
    chain distance, always above ``max_degree``).
    ``closure_only[i]`` is true when the member was added only to preserve exact
    kinship, after the ``max_degree`` traversal; it is an explicit warning that
    the person's diagnosis is outside the requested observation set unless a
    caller deliberately opts in. Ordering is deterministic: proband first, then
    by ``(degree, str(id))``.

    ``member_index`` maps each member to its index in the parent graph.
    ``sire_index``/``dam_index`` give each member's father/mother as a
    *position in this pedigree* (``-1`` when unrecorded), so relationships can
    be computed without rebuilding an id-to-index map. All three are ``None``
    only for a hand-assembled ``Pedigree``."""
    proband: object
    ids: list
    father: list
    mother: list
    degree: np.ndarray
    closure_only: np.ndarray
    member_index: np.ndarray = None
    sire_index: np.ndarray = None
    dam_index: np.ndarray = None


def build_parent_graph(ids: Sequence, father: Sequence,
                       mother: Sequence) -> ParentGraph:
    """Index population trio records for traversal.

    ``ids`` must be unique; a parent not among ``ids`` (``None``, ``nan``, or
    any unlisted value) is an unknown founder. Unlisted **non-null** values are
    counted in ``n_unresolved_parents``: a register boundary makes some of
    them inevitable, but the count is what lets a caller distinguish that
    boundary from an id-format mismatch or a failed join. Raises
    ``ValueError`` when ``ids``/``father``/``mother`` differ in length, when
    ``ids`` contains a missing value (``None``, NaN, pandas NA/NaT or a
    missing-marker string such as ``""``/``"NA"``), on duplicate ids and on a
    person recorded as their own parent. Longer cycles (a person being their
    own ancestor) are not checked here;
    `ltpred.covariance.kinship_from_pedigree` raises on them.
    """
    ids, index, sire, dam, children, n_unresolved = _parent_links(
        ids, father, mother)
    n = len(ids)
    # Each person belongs to at most one full-sibling group. Groups arrive in
    # row order, so their adjacency lists need neither sets nor sorting.
    sibs = [[] for _ in range(n)]
    by_parents = {}
    for i in range(n):
        if sire[i] != -1 and dam[i] != -1:
            by_parents.setdefault((sire[i], dam[i]), []).append(i)
    for group in by_parents.values():
        if len(group) > 1:
            for i in group:
                sibs[i] = [j for j in group if j != i]
    return ParentGraph(ids=ids, sire=sire, dam=dam, children=children,
                       sibs=sibs, index=index, n_unresolved_parents=n_unresolved)


def extract_pedigree(graph: ParentGraph, proband: object,
                     max_degree: int = 2) -> Pedigree:
    """Extract one proband's relatives and complete ancestral closure.

    BFS visits relatives through ``max_degree``; closure then adds every
    recorded ancestor, making extracted kinship exact even with inbreeding.
    ``closure_only`` identifies added ancestors. ``max_degree`` must be a
    positive integer and ``proband`` must be present.
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
    # documents.
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
        """Parent id when recorded and extracted, else ``None`` (founder)."""
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
        member_index=np.array(members, dtype=np.intp),
        sire_index=np.array([pos.get(graph.sire[j], -1) for j in members],
                            dtype=np.intp),
        dam_index=np.array([pos.get(graph.dam[j], -1) for j in members],
                           dtype=np.intp),
    )
