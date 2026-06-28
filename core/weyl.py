"""Weyl / reflection groups and two exotic groups, as Cayley tables + geometry.

Reflection groups are built directly as their standard (reflection) representation
-- a set of n x n orthogonal matrices closed under multiplication -- so:
  * the Cayley table is matrix multiplication, and
  * the geometric target for the decode is the ROOT SYSTEM: coords[g] = M_g @ v0
    for a chosen root v0 (the W-orbit of v0).

Groups:
  W(A3)=S4      order 24,  rank 3  -> cuboctahedron (A3 roots)
  W(B3)=W(C3)   order 48,  rank 3  -> octahedron (B3 short roots)  [= O_h]
  W(D4)         order 192, rank 4  -> 24-cell (D4 roots)
  W(B4)=W(C4)   order 384, rank 4  -> 16-cell (B4 short roots)
  W(G2)         order 12,  rank 2  -> hexagon (G2 short roots)
  PSL(2,7)=GL(3,2)  order 168      -> NO Euclidean polytope (complex 3-dim irreps; Klein quartic)
  A6                order 360      -> NO low-dim polytope (smallest faithful irrep is 5-dim)
"""
import numpy as np
from itertools import permutations, product


# ----------------------------- matrix-group helper ----------------------------

def _build_matrix_group(mats):
    """Cayley table of a multiplicatively-closed set of matrices (identity first)."""
    mats = np.asarray(mats, float)
    m = len(mats)
    I = np.eye(mats.shape[1])
    i0 = int(np.abs(mats - I).reshape(m, -1).sum(1).argmin())
    mats[[0, i0]] = mats[[i0, 0]]
    key = {tuple(np.round(M, 5).ravel()): i for i, M in enumerate(mats)}
    table = np.empty((m, m), dtype=np.int64)
    flat = mats.reshape(m, -1)
    for i in range(m):
        for j in range(m):
            P = mats[i] @ mats[j]
            k = tuple(np.round(P, 5).ravel())
            table[i, j] = key.get(k, int(np.abs(flat - P.ravel()).sum(1).argmin()))
    return mats, table


def _orbit_coords(mats, v0):
    v0 = np.asarray(v0, float); v0 = v0 / np.linalg.norm(v0)
    return np.array([M @ v0 for M in mats])


def _signed_perm_mats(n, even_only=False):
    out = []
    for p in permutations(range(n)):
        for s in product((1.0, -1.0), repeat=n):
            if even_only and np.prod(s) < 0:
                continue
            M = np.zeros((n, n))
            for j in range(n):
                M[p[j], j] = s[j]
            out.append(M)
    return np.array(out)


def _Sn_reflection_mats(d):
    """S_d acting on the (d-1)-dim sum-zero space = reflection rep of W(A_{d-1})."""
    M0 = np.eye(d) - np.ones((d, d)) / d
    B = np.linalg.qr(M0)[0][:, :d - 1]            # d x (d-1) orthonormal
    out = []
    for p in permutations(range(d)):
        P = np.zeros((d, d))
        for j in range(d):
            P[p[j], j] = 1.0
        out.append(B.T @ P @ B)
    return np.array(out), B


def _pack(name, mats, table, polytope, coords, dim):
    geo = {"polytope": polytope, "irrep_dim": dim,
           "coords4d": None, "coords3d": None, "coordsND": coords}
    if dim == 3:
        geo["coords3d"] = coords
    elif dim == 4:
        geo["coords4d"] = coords
    return {"n": len(mats), "table": table, "name": name, "identity_index": 0,
            "normal_subgroup": None, "coset_labels": None,
            "geometry": geo, "_mats": mats}


# ------------------------------- the groups -----------------------------------

def weyl_A3():
    mats, B = _Sn_reflection_mats(4)              # S4, 3-dim, order 24
    mats, table = _build_matrix_group(mats)
    v0 = B.T @ (np.eye(4)[0] - np.eye(4)[1])      # an A3 root e0-e1
    return _pack("W(A3)", mats, table, "cuboctahedron", _orbit_coords(mats, v0), 3)


def weyl_B3():
    mats = _signed_perm_mats(3)                    # order 48 = O_h
    mats, table = _build_matrix_group(mats)
    return _pack("W(B3)", mats, table, "octahedron", _orbit_coords(mats, [1, 0, 0]), 3)


def weyl_D4():
    mats = _signed_perm_mats(4, even_only=True)    # order 192
    mats, table = _build_matrix_group(mats)
    v0 = np.array([1, 1, 0, 0]) / np.sqrt(2)       # a D4 root e0+e1
    return _pack("W(D4)", mats, table, "24-cell", _orbit_coords(mats, v0), 4)


def weyl_B4():
    mats = _signed_perm_mats(4)                    # order 384
    mats, table = _build_matrix_group(mats)
    return _pack("W(B4)", mats, table, "16-cell", _orbit_coords(mats, [1, 0, 0, 0]), 4)


def weyl_G2():
    th = np.pi / 3
    rot = lambda a: np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    R = np.diag([1.0, -1.0])
    mats = np.array([rot(k * th) for k in range(6)] + [rot(k * th) @ R for k in range(6)])
    mats, table = _build_matrix_group(mats)        # dihedral order 12
    return _pack("W(G2)", mats, table, "hexagon", _orbit_coords(mats, [1, 0]), 2)


def gl3_2():
    """GL(3,2) = PSL(2,7), order 168. 3x3 invertible matrices over F2."""
    els = []
    for bits in range(512):
        M = np.array([[(bits >> (3 * r + c)) & 1 for c in range(3)] for r in range(3)], dtype=np.int64)
        # det over F2
        d = (M[0, 0]*(M[1, 1]*M[2, 2] ^ M[1, 2]*M[2, 1])
             ^ M[0, 1]*(M[1, 0]*M[2, 2] ^ M[1, 2]*M[2, 0])
             ^ M[0, 2]*(M[1, 0]*M[2, 1] ^ M[1, 1]*M[2, 0])) & 1
        if d == 1:
            els.append(M)
    key = {M.tobytes(): i for i, M in enumerate(els)}
    # identity first
    I = np.eye(3, dtype=np.int64)
    i0 = key[I.tobytes()]; els[0], els[i0] = els[i0], els[0]
    key = {M.tobytes(): i for i, M in enumerate(els)}
    n = len(els)
    table = np.empty((n, n), dtype=np.int64)
    for i in range(n):
        for j in range(n):
            table[i, j] = key[((els[i] @ els[j]) & 1).tobytes()]
    return {"n": n, "table": table, "name": "PSL(2,7)", "identity_index": 0,
            "normal_subgroup": None, "coset_labels": None,
            "geometry": {"polytope": None, "irrep_dim": 3,
                         "coords4d": None, "coords3d": None, "coordsND": None}}


def alt6():
    """A6 (order 360) via sympy even permutations of 6 points."""
    from sympy.combinatorics.named_groups import AlternatingGroup
    els = []
    for p in AlternatingGroup(6).generate(af=False):
        af = list(p.array_form) + list(range(len(p.array_form), 6))
        els.append(tuple(af))
    ident = tuple(range(6))
    els = [ident] + [e for e in els if e != ident]
    idx = {e: i for i, e in enumerate(els)}
    n = len(els)
    table = np.empty((n, n), dtype=np.int64)
    for i, a in enumerate(els):
        for j, b in enumerate(els):
            table[i, j] = idx[tuple(a[b[x]] for x in range(6))]
    return {"n": n, "table": table, "name": "A6", "identity_index": 0,
            "normal_subgroup": None, "coset_labels": None,
            "geometry": {"polytope": None, "irrep_dim": 5,
                         "coords4d": None, "coords3d": None, "coordsND": None}}


BUILDERS = {"WA3": weyl_A3, "WB3": weyl_B3, "WD4": weyl_D4, "WB4": weyl_B4,
            "WG2": weyl_G2, "PSL27": gl3_2, "A6": alt6}
