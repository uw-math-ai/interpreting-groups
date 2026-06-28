"""Binary polyhedral groups as unit quaternions (their elements ARE polytopes).

The binary tetrahedral/octahedral/icosahedral groups 2T, 2O, 2I are finite
subgroups of the unit quaternions S^3 = SU(2). Their element sets are, literally,
the vertices of the 24-cell (2T), the 24-cell+dual (2O), and the **600-cell** (2I).
So building them from explicit quaternions gives, for free, the "true" geometric
coordinate of every group element — the target a grokked model should recover.

Each builder returns the standard group dict used across the repo (n, table, ...)
plus a ``geometry`` block: 4-D quaternion coordinates (the polytope), a 3-D
rotation-rep realization (quaternion conjugation), and the faithful/geometric
irrep dimension to project onto.
"""
import numpy as np

PHI = (1.0 + np.sqrt(5.0)) / 2.0


# ----------------------------- quaternion algebra -----------------------------

def qmul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ])


def rotation_matrix(q):
    """3x3 SO(3) matrix of the rotation that unit quaternion q induces (v -> q v q*)."""
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
    ])


# ----------------------------- vertex constructions ---------------------------

def _perm_sign(p):
    p = list(p)
    s = 1
    for i in range(len(p)):
        for j in range(i + 1, len(p)):
            if p[i] > p[j]:
                s = -s
    return s


def _even_perms4():
    import itertools
    return [p for p in itertools.permutations(range(4)) if _perm_sign(p) == 1]


def verts_2T():
    """24 unit quaternions = 24-cell vertices (binary tetrahedral group)."""
    V = []
    # 8: (+-1,0,0,0) and coordinate permutations
    for i in range(4):
        for s in (1.0, -1.0):
            v = [0.0, 0.0, 0.0, 0.0]
            v[i] = s
            V.append(v)
    # 16: (+-1/2, +-1/2, +-1/2, +-1/2)
    import itertools
    for signs in itertools.product((0.5, -0.5), repeat=4):
        V.append(list(signs))
    return np.array(V)


def verts_2O():
    """48 unit quaternions (binary octahedral group)."""
    V = list(verts_2T())
    import itertools
    r = 1.0 / np.sqrt(2.0)
    # 24: two coords = +-1/sqrt2, other two zero
    for i, j in itertools.combinations(range(4), 2):
        for si in (r, -r):
            for sj in (r, -r):
                v = [0.0, 0.0, 0.0, 0.0]
                v[i] = si
                v[j] = sj
                V.append(v)
    return np.array(V)


def verts_2I():
    """120 unit quaternions = 600-cell vertices (binary icosahedral group)."""
    import itertools
    V = []
    # 8: (+-1,0,0,0) & perms
    for i in range(4):
        for s in (1.0, -1.0):
            v = [0.0, 0.0, 0.0, 0.0]
            v[i] = s
            V.append(v)
    # 16: (+-1/2,...)
    for signs in itertools.product((0.5, -0.5), repeat=4):
        V.append(list(signs))
    # 96: even permutations of (+-phi/2, +-1/2, +-1/(2 phi), 0), signs on the 3 nonzero
    base = [PHI / 2.0, 0.5, 1.0 / (2.0 * PHI), 0.0]
    for perm in _even_perms4():
        arranged = [base[perm[k]] for k in range(4)]
        nz = [k for k in range(4) if arranged[k] != 0.0]
        for signs in itertools.product((1.0, -1.0), repeat=3):
            v = list(arranged)
            for k, sgn in zip(nz, signs):
                v[k] *= sgn
            V.append(v)
    return np.array(V)


# ----------------------------- group assembly ---------------------------------

def _build_from_vertices(V, name, geom_irrep_dim, polytope, seed3d=None):
    """Cayley table from an explicit, multiplicatively-closed quaternion vertex set.

    The element set must contain -1 (binary groups do); we do NOT identify q ~ -q.
    """
    m = len(V)
    key = {tuple(np.round(v, 6)): i for i, v in enumerate(V)}

    def lookup(v):
        k = tuple(np.round(v, 6))
        if k in key:
            return key[k]
        # tolerant fallback
        d = np.abs(V - v).sum(1)
        i = int(np.argmin(d))
        assert d[i] < 1e-4, f"product not in vertex set (min dist {d[i]:.2e})"
        return i

    table = np.empty((m, m), dtype=np.int64)
    for i in range(m):
        for j in range(m):
            table[i, j] = lookup(qmul(V[i], V[j]))

    ident = lookup(np.array([1.0, 0.0, 0.0, 0.0]))
    # rotation (3-D) coordinate per element: conjugation acting on a seed point
    coords3d = None
    if seed3d is not None:
        coords3d = np.array([rotation_matrix(V[i]) @ seed3d for i in range(m)])

    return {
        "n": m, "table": table, "name": name, "identity_index": int(ident),
        "normal_subgroup": None, "coset_labels": None,
        "geometry": {
            "irrep_dim": geom_irrep_dim,
            "polytope": polytope,
            "coords4d": V.copy(),          # the polytope vertices (faithful geometry)
            "coords3d": coords3d,          # rotation-rep orbit of a seed (icosahedron/etc.)
            "seed3d": None if seed3d is None else np.array(seed3d),
        },
    }


def icosahedron_vertices():
    """12 icosahedron vertices (cyclic perms of (0,+-1,+-phi)), unit-normalized."""
    V = []
    for a, b in ((1, PHI), (1, -PHI), (-1, PHI), (-1, -PHI)):
        V += [(0, a, b), (a, b, 0), (b, 0, a)]
    V = np.array(V, dtype=float)
    return V / np.linalg.norm(V[0])


def octahedron_vertices():
    return np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], float)


def binary_icosahedral():
    """2I = SL(2,5), order 120; elements = 600-cell vertices."""
    seed = icosahedron_vertices()[0]
    return _build_from_vertices(verts_2I(), "2I", geom_irrep_dim=4,
                                polytope="600cell", seed3d=seed)


def binary_octahedral():
    """2O, order 48; elements = 24-cell + dual vertices.

    The first 24 vertices are 2T (binary tetrahedral), which is a normal index-2
    subgroup (kernel of the det / 'which sheet' map) -> a clean coset partition.
    """
    seed = octahedron_vertices()[0]
    G = _build_from_vertices(verts_2O(), "2O", geom_irrep_dim=4,
                             polytope="24cell_plus_dual", seed3d=seed)
    G["normal_subgroup"] = list(range(24))
    G["coset_labels"] = ("2T (24-cell)", r"2O$\setminus$2T")
    return G


def binary_tetrahedral():
    """2T = SL(2,3), order 24; elements = 24-cell vertices."""
    seed = octahedron_vertices()[0]
    return _build_from_vertices(verts_2T(), "2T", geom_irrep_dim=4,
                                polytope="24cell", seed3d=seed)


def _quat_a5():
    """Canonical icosian A5 reps, their 3-D rotation matrices, table, and the
    sqrt5->-sqrt5 (Galois) permutation on the reps."""
    Vf = verts_2I()
    def canon(v):
        v = np.round(v, 6)
        for c in v:
            if abs(c) > 1e-9:
                if c < 0: v = -v
                break
        return v
    reps, key = [], {}
    for v in Vf:
        cv = canon(v); k = tuple(np.round(cv, 6))
        if k not in key: key[k] = len(reps); reps.append(cv)
    reps = np.array(reps)
    def look(v):
        k = tuple(np.round(canon(v), 6))
        return key.get(k, int(np.abs(reps - canon(v)).sum(1).argmin()))
    table = np.array([[look(qmul(reps[i], reps[j])) for j in range(60)] for i in range(60)])
    Rq = [rotation_matrix(r) for r in reps]
    def fc(c):
        a = abs(c)
        if abs(a - PHI / 2) < 1e-3: return -np.sign(c) * (PHI - 1) / 2
        if abs(a - (PHI - 1) / 2) < 1e-3: return -np.sign(c) * PHI / 2
        return c
    gp = np.array([look(np.array([fc(c) for c in v])) for v in reps])
    return reps, Rq, table, gp


def _order(table, g):
    e = next(x for x in range(len(table)) if all(table[x, y] == y for y in range(len(table))))
    x, o = g, 1
    while x != e:
        x = table[x, g]; o += 1
    return o, e


def ih_icosahedron_coords():
    """3-D icosahedron coordinates for the full icosahedral group I_h=A5xC2 as built
    by core.groups.icosahedral_group (sympy labeling), via its odd 3-dim irrep 3au.

    Returns {'canonical','galois'}: each [120,3], element g=(a5, sign) ->
    (-1)^sign * R(a5) @ seed, with R the icosahedral rotation rep matched to the
    sympy A5 by an explicit isomorphism. Orbit = the 12 icosahedron vertices.
    """
    from core.groups import icosahedral_group
    reps, Rq, tB, gp = _quat_a5()
    G = icosahedral_group(); tA = np.array(G["table"])[:60, :60]
    _, eA = _order(tA, 0); _, eB = _order(tB, 0)

    def gens(t, e):
        a = next(x for x in range(60) if _order(t, x)[0] == 2)
        for b in range(60):
            if _order(t, b)[0] == 3 and _order(t, t[a, b])[0] == 5:
                seen = {e}; fr = [e]
                while fr:
                    nx = []
                    for s in fr:
                        for g in (a, b):
                            r = t[s, g]
                            if r not in seen: seen.add(r); nx.append(r)
                    fr = nx
                if len(seen) == 60:
                    return a, b
    aA, bA = gens(tA, eA)

    def iso():
        for aB in range(60):
            if _order(tB, aB)[0] != 2: continue
            for bB in range(60):
                if _order(tB, bB)[0] != 3 or _order(tB, tB[aB, bB])[0] != 5: continue
                seen = {eA: eB}; stack = [(eA, eB)]; ok = True
                while stack and ok:
                    sA, sB = stack.pop()
                    for gA, gB in ((aA, aB), (bA, bB)):
                        rA, rB = tA[sA, gA], tB[sB, gB]
                        if rA in seen:
                            if seen[rA] != rB: ok = False; break
                        else:
                            seen[rA] = rB; stack.append((rA, rB))
                if ok and len(seen) == 60 and all(
                        seen[tA[i, j]] == tB[seen[i], seen[j]] for i in range(60) for j in range(60)):
                    return seen
    phi = iso()
    inv = np.array([phi[i] for i in range(60)])
    seed = None
    for M in Rq:
        if np.allclose(np.linalg.matrix_power(M, 5), np.eye(3), atol=1e-5) and not np.allclose(M, np.eye(3), atol=1e-5):
            w, V = np.linalg.eig(M); seed = np.real(V[:, int(np.argmin(np.abs(w - 1)))])
            seed /= np.linalg.norm(seed); break
    out = {}
    for name, mp in (("canonical", inv), ("galois", inv[gp])):
        C = np.zeros((120, 3))
        for g in range(120):
            a5, sign = g % 60, g // 60
            C[g] = ((-1) ** sign) * (Rq[mp[a5]] @ seed)
        out[name] = C
    return out


def alternating5_quat():
    """A5 (order 60) as the rotation quotient 2I/{+-1}; 3-D icosahedral geometry.

    Elements are sign-canonicalized icosian cosets {+-q}; multiplication is
    quaternion multiplication followed by re-canonicalization. The faithful
    geometry here is the 3-D rotation rep (the icosahedron), not 4-D.
    """
    Vfull = verts_2I()

    def canon(v):
        v = np.round(v, 6)
        for c in v:                     # sign so first nonzero coord > 0
            if abs(c) > 1e-9:
                if c < 0:
                    v = -v
                break
        return v

    reps, key = [], {}
    for v in Vfull:
        cv = canon(v)
        k = tuple(np.round(cv, 6))
        if k not in key:
            key[k] = len(reps)
            reps.append(cv)
    reps = np.array(reps)               # 60 representatives
    m = len(reps)

    def lookup(v):
        k = tuple(np.round(canon(v), 6))
        if k in key:
            return key[k]
        d = np.abs(reps - canon(v)).sum(1)
        i = int(np.argmin(d))
        assert d[i] < 1e-4
        return i

    table = np.empty((m, m), dtype=np.int64)
    for i in range(m):
        for j in range(m):
            table[i, j] = lookup(qmul(reps[i], reps[j]))
    ident = lookup(np.array([1.0, 0.0, 0.0, 0.0]))
    seed = icosahedron_vertices()[0]
    coords3d = np.array([rotation_matrix(reps[i]) @ seed for i in range(m)])
    return {
        "n": m, "table": table, "name": "A5", "identity_index": int(ident),
        "normal_subgroup": None, "coset_labels": None,
        "geometry": {"irrep_dim": 3, "polytope": "icosahedron",
                     "coords4d": None, "coords3d": coords3d, "seed3d": np.array(seed)},
    }
