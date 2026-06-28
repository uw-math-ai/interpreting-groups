"""Finite-group construction and structural partitions.

Generalizes the monoid J-class machinery in ``core.algebra`` to arbitrary finite
groups. The headline object is the full icosahedral group

    I_h  =  A5 x C2          (order 120)

where A5 is the icosahedral *rotation* group (even permutations of 5 points) and
the C2 factor is the central inversion (proper rotations vs. rotoreflections).

The key interoperability contract: :func:`group_partition` returns the *same dict
shape* as ``core.algebra.algebraic_jclass_order`` / ``jclass_permutation`` so the
existing analysis scripts work with a one-line swap. For a group the natural
analog of a monoid J-class is a **conjugacy class** (mode="conjugacy"); the
composition-series view (mode="coset") splits elements by the cosets of the
normal A5 subgroup.
"""
import numpy as np


# --------------------------------------------------------------------------
# Group construction
# --------------------------------------------------------------------------

def _a5_elements():
    """Return the 60 elements of A5 as sympy Permutations, identity first.

    Falls back to a dependency-free generation by closing two generators of A5
    under composition if sympy is unavailable.
    """
    try:
        from sympy.combinatorics.named_groups import AlternatingGroup
        els = list(AlternatingGroup(5).generate(af=False))
    except Exception:
        from sympy.combinatorics import Permutation
        # 5-cycle and a 3-cycle generate A5
        gens = [Permutation([1, 2, 3, 4, 0]), Permutation([0, 2, 3, 1, 4])]
        seen = {}
        frontier = [Permutation(list(range(5)))]
        while frontier:
            nxt = []
            for x in frontier:
                if x in seen:
                    continue
                seen[x] = True
                for g in gens:
                    nxt.append(x * g)
            frontier = [y for y in nxt if y not in seen]
        els = list(seen.keys())

    # Put the identity first for a stable, interpretable index 0.
    ident = [e for e in els if e.is_Identity][0]
    els = [ident] + [e for e in els if not e.is_Identity]
    return els


def icosahedral_group():
    """Build the full icosahedral group I_h = A5 x C2 (order 120).

    Indexing convention:
        index = a5_index + 60 * sign_bit
        sign_bit 0 -> proper rotations  (the normal A5 subgroup, indices 0..59)
        sign_bit 1 -> rotoreflections   (indices 60..119)
    The identity is index 0.

    Returns:
        dict with:
            n: 120
            table: [120, 120] int array, table[a, b] = a . b
            elements: list of (a5_index, sign) with sign in {+1, -1}
            identity_index: 0
            a5_indices: list[int] of the order-60 normal rotation subgroup
            name: "I_h"
    """
    a5 = _a5_elements()
    a5_idx = {e: i for i, e in enumerate(a5)}
    n5 = len(a5)              # 60
    n = 2 * n5               # 120

    # Precompute the A5 multiplication table once.
    a5_table = np.empty((n5, n5), dtype=np.int64)
    for i in range(n5):
        for j in range(n5):
            a5_table[i, j] = a5_idx[a5[i] * a5[j]]

    table = np.empty((n, n), dtype=np.int64)
    for a in range(n):
        i, s = a % n5, a // n5          # s in {0, 1} is the sign bit
        for b in range(n):
            j, t = b % n5, b // n5
            k = a5_table[i, j]
            sign_bit = s ^ t            # C2 multiplication
            table[a, b] = k + n5 * sign_bit

    elements = [(a % n5, 1 if a // n5 == 0 else -1) for a in range(n)]

    return {
        "n": n,
        "table": table,
        "elements": elements,
        "identity_index": 0,
        "a5_indices": list(range(n5)),
        "name": "I_h",
    }


# --------------------------------------------------------------------------
# Group-theoretic structure from a Cayley table (no monoid assumptions)
# --------------------------------------------------------------------------

def inverses(table):
    """Return inv[g] such that table[g, inv[g]] == identity, for a group table.

    Identity is detected as the index e with table[e, x] == x for all x.
    """
    n = table.shape[0]
    ident = next(e for e in range(n) if np.all(table[e] == np.arange(n)))
    inv = np.empty(n, dtype=np.int64)
    for g in range(n):
        inv[g] = int(np.where(table[g] == ident)[0][0])
    return inv, ident


def conjugacy_classes(table):
    """Partition group elements into conjugacy classes.

    Conjugate of x by g is g . x . g^{-1}. Returns classes sorted by
    (size, min element) so labels are deterministic.

    Returns:
        list of sorted index lists, one per conjugacy class.
    """
    n = table.shape[0]
    inv, _ = inverses(table)
    seen = np.zeros(n, dtype=bool)
    classes = []
    for x in range(n):
        if seen[x]:
            continue
        orbit = set()
        for g in range(n):
            # g . x . g^{-1}
            orbit.add(int(table[table[g, x], inv[g]]))
        for y in orbit:
            seen[y] = True
        classes.append(sorted(orbit))
    classes.sort(key=lambda c: (len(c), c[0]))
    return classes


# --------------------------------------------------------------------------
# Partition dict (drop-in for algebraic_jclass_order / jclass_permutation)
# --------------------------------------------------------------------------

def _partition_dict(class_lists, n, labels):
    """Assemble the standard perm dict from an ordered list of index-blocks."""
    sorted_indices = np.array([i for blk in class_lists for i in blk], dtype=np.int64)
    class_counts = np.array([len(blk) for blk in class_lists], dtype=np.int64)

    # Per-element class id, in original index order.
    elem_class = np.empty(n, dtype=np.int64)
    for cid, blk in enumerate(class_lists):
        for i in blk:
            elem_class[i] = cid

    boundaries = np.cumsum(class_counts)[:-1]
    tick_positions = np.concatenate(([0], boundaries)) + class_counts / 2 - 0.5
    return {
        "sorted_indices": sorted_indices,
        "boundaries": boundaries,
        "tick_positions": tick_positions,
        "tick_labels": list(labels),
        "j_classes": elem_class,          # analog of gcd-based j_classes
        "unique_j": np.arange(len(class_lists)),
        "class_counts": class_counts,
    }


def group_partition(table, mode="conjugacy", a5_indices=None, coset_labels=None):
    """Group analog of ``algebraic_jclass_order`` returning the same dict shape.

    Args:
        table: [n, n] Cayley table.
        mode: "conjugacy" sorts elements by conjugacy class; "coset" gives the
            composition-series view (cosets of the normal A5, conjugacy-sorted
            within each coset).
        a5_indices: indices of the normal A5 subgroup (required for mode="coset";
            defaults to the first n/2 indices, matching ``icosahedral_group``).

    Returns:
        perm dict: sorted_indices, boundaries, tick_positions, tick_labels,
        j_classes, unique_j, class_counts.
    """
    n = table.shape[0]
    classes = conjugacy_classes(table)

    if mode == "conjugacy":
        labels = [f"$K_{{{cid}}}$ ({len(blk)})" for cid, blk in enumerate(classes)]
        return _partition_dict(classes, n, labels)

    if mode == "coset":
        if a5_indices is None:
            a5_indices = list(range(n // 2))
        a5_set = set(a5_indices)
        # Two cosets; within each, keep conjugacy-class grouping for clean blocks.
        coset0, coset1 = [], []
        for blk in classes:
            coset0.extend([i for i in blk if i in a5_set])
            coset1.extend([i for i in blk if i not in a5_set])
        labels = list(coset_labels) if coset_labels is not None else \
            ["A5 (rotations)", r"A5$\cdot\sigma$ (rotoreflections)"]
        return _partition_dict([coset0, coset1], n, labels)

    raise ValueError(f"unknown mode {mode!r} (expected 'conjugacy' or 'coset')")


def symmetric_group(d=5):
    """Build the symmetric group S_d (Weyl group of A_{d-1}) as a Cayley table.

    For d=5 this is order 120, with the standard (d-1)-dim irrep realizing the
    A4 root system. Returns the standard group dict plus geometry (root-system
    orbit coordinates) and the alternating subgroup as the normal subgroup.
    """
    from sympy.combinatorics.named_groups import SymmetricGroup
    perms = []
    for p in SymmetricGroup(d).generate(af=False):
        af = list(p.array_form) + list(range(len(p.array_form), d))
        perms.append(tuple(af))
    # identity first
    ident_t = tuple(range(d))
    perms = [ident_t] + [p for p in perms if p != ident_t]
    idx = {p: i for i, p in enumerate(perms)}
    n = len(perms)

    def comp(a, b):
        return tuple(a[b[x]] for x in range(d))

    table = np.empty((n, n), dtype=np.int64)
    for i, a in enumerate(perms):
        for j, b in enumerate(perms):
            table[i, j] = idx[comp(a, b)]

    # parity -> even (A_d, normal index 2) vs odd
    def parity(p):
        s = 0
        for i in range(d):
            for j in range(i + 1, d):
                if p[i] > p[j]:
                    s ^= 1
        return s
    normal = [i for i, p in enumerate(perms) if parity(p) == 0]

    # standard (d-1)-dim rep: permutation matrices restricted to sum-zero hyperplane.
    # Orthonormal basis B (d x (d-1)) of {x: sum x = 0} via QR of centered identity.
    M = np.eye(d) - np.ones((d, d)) / d
    B = np.linalg.qr(M)[0][:, :d - 1]
    v0 = B.T @ (np.eye(d)[0] - np.eye(d)[1])     # image of root e0 - e1
    coords = np.empty((n, d - 1))
    for i, p in enumerate(perms):
        Pm = np.zeros((d, d))
        for x in range(d):
            Pm[p[x], x] = 1.0                    # P e_x = e_{p(x)}
        rho = B.T @ Pm @ B
        coords[i] = rho @ v0
    return {
        "n": n, "table": table, "name": f"S{d}", "identity_index": 0,
        "normal_subgroup": normal, "coset_labels": ("A_d (even)", "odd"),
        "geometry": {"irrep_dim": d - 1, "polytope": f"A{d-1}_roots",
                     "coords4d": coords if d - 1 == 4 else None,
                     "coords3d": coords if d - 1 == 3 else None,
                     "coordsND": coords, "seed3d": None},
    }


# Registry: task name -> group dict (Cayley table + geometry + partition metadata).
def get_group(task):
    """Return the standard group dict for a task name (drop-in across the repo).

    Supported: 'icosahedral' (I_h=A5xC2), '2I','2O','2T' (binary polyhedral),
    'A5' (icosahedral rotations), 'S5' (symmetric / Weyl group of A4).
    """
    if task == "icosahedral":
        G = icosahedral_group()
        G["normal_subgroup"] = G["a5_indices"]
        G["coset_labels"] = ("A5 (rotations)", "rotoreflections")
        G.setdefault("geometry", {"irrep_dim": 3, "polytope": "icosahedron",
                                  "coords4d": None, "coords3d": None, "seed3d": None})
        return G
    if task in ("2I", "2O", "2T", "A5"):
        from core import quaternions as q
        return {"2I": q.binary_icosahedral, "2O": q.binary_octahedral,
                "2T": q.binary_tetrahedral, "A5": q.alternating5_quat}[task]()
    if task in ("S5", "S4", "S6"):
        return symmetric_group(int(task[1:]))
    from core.weyl import BUILDERS
    if task in BUILDERS:
        return BUILDERS[task]()
    raise ValueError(f"unknown group task {task!r}")


GROUP_TASKS = ["icosahedral", "2I", "2O", "2T", "A5", "S5",
               "WA3", "WB3", "WD4", "WB4", "WG2", "PSL27", "A6"]


def analysis_partition(task, P, mode="conjugacy"):
    """Return the structural ``perm`` dict for any task (drop-in interface).

    For task="modmult" this is the monoid J-class ordering; for a finite group it
    is the conjugacy-class (mode="conjugacy") or composition-series coset
    (mode="coset", requires an index-2 normal subgroup) ordering. The perm dict
    has identical keys in all cases so analysis scripts consume it uniformly.

    Returns:
        (perm dict or None, group dict or None). perm is None if mode="coset" is
        requested for a group without a suitable normal subgroup. group dict is
        None only for modmult.
    """
    if task == "modmult":
        from core.algebra import algebraic_jclass_order
        return algebraic_jclass_order(P), None
    G = get_group(task)
    if mode == "coset" and G.get("normal_subgroup") is None:
        return None, G
    perm = group_partition(G["table"], mode=mode,
                           a5_indices=G.get("normal_subgroup"),
                           coset_labels=G.get("coset_labels"))
    return perm, G
