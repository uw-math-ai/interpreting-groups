"""Generic numeric representation theory from a Cayley table.

The icosahedral pipeline (``experiments/rep_analysis``) only needs, for a finite
group G, the irreducible characters as functions on the group elements plus their
dimensions — everything else (isotypic projection, energy spectrum, Parseval) is
already group-agnostic. This module computes that character table *numerically*
for ANY finite group given its Cayley table, so the same grokking + mech-interp
analysis runs on S5, the binary polyhedral groups 2I/2O/2T, A5, etc.

Method (Burnside--Dixon, real/complex eigenvector form):
  * the class-sum matrices M_i (structure constants of the center of C[G]) commute
    and are simultaneously diagonalized by the central primitive idempotents;
  * a generic real combination of the M_i has distinct eigenvalues, so a single
    eigendecomposition separates the irreps;
  * each eigenvector's components are proportional to conj(chi_r) over the classes,
    and the character is recovered by the weighted orthonormality
    sum_i |C_i| |chi_r(c_i)|^2 = |G| with phase fixed by chi_r(1) = d_r > 0.

Everything is validated downstream (orthogonality, sum d^2 = |G|, Parseval=1).
"""
import numpy as np

from core.groups import conjugacy_classes, inverses


def conjugacy_data(table):
    """Return (classes, class_of_element[n], class_sizes[k], identity_index)."""
    n = table.shape[0]
    classes = conjugacy_classes(table)
    class_of = np.empty(n, dtype=np.int64)
    for ci, c in enumerate(classes):
        for g in c:
            class_of[g] = ci
    sizes = np.array([len(c) for c in classes], dtype=np.int64)
    inv, ident = inverses(table)
    return classes, class_of, sizes, ident, inv


def class_sum_matrices(table, classes, class_of, inv):
    """Class-algebra structure-constant matrices M_i (k x k), which commute.

    a_{i j l} = |{x in C_i : x^{-1} z_l in C_j}| for a fixed representative z_l of
    class l. Then (M_i)[l, j] = a_{i j l}.
    """
    k = len(classes)
    M = np.zeros((k, k, k), dtype=np.float64)   # M[i][l][j]
    for l in range(k):
        z = classes[l][0]                        # representative of class l
        for i in range(k):
            cnt = np.zeros(k, dtype=np.float64)
            for x in classes[i]:
                y = int(table[inv[x], z])
                cnt[class_of[y]] += 1.0
            M[i, l, :] = cnt
    return M


def character_table(table, rng_seed=0):
    """Compute the full irreducible character table from a Cayley table.

    Returns a dict:
        classes:    list of class index-lists (sorted by (size, min) per conjugacy_classes)
        class_of:   [n] class id per element
        sizes:      [k] class sizes
        identity:   identity element index
        id_class:   class id of the identity
        chars:      [k, k] complex character table, rows=irreps, cols=classes,
                    rows sorted by (dim, then lexicographic) for determinism
        dims:       [k] integer dimensions (chars[:, id_class] rounded, real)
        names:      [k] names like '1a','1b','3a',... grouped by dimension
        char_per_element: [k, n] complex chi_r over every element
    """
    n = table.shape[0]
    classes, class_of, sizes, ident, inv = conjugacy_data(table)
    k = len(classes)
    id_class = int(class_of[ident])

    M = class_sum_matrices(table, classes, class_of, inv)

    # Generic real combination -> distinct eigenvalues -> common eigenvectors.
    rng = np.random.default_rng(rng_seed)
    for _ in range(50):
        t = rng.standard_normal(k)
        S = np.einsum("i,ilj->lj", t, M)         # sum_i t_i M_i  (k x k)
        evals, evecs = np.linalg.eig(S)
        # require distinct eigenvalues (well-separated) for a clean split
        ev = np.sort_complex(evals)
        if np.min(np.abs(np.diff(ev))) > 1e-6:
            break
    else:
        raise RuntimeError("could not separate irreps (degenerate eigenvalues)")

    chars = np.zeros((k, k), dtype=np.complex128)
    h = sizes.astype(np.float64)
    for r in range(k):
        v = evecs[:, r]                          # component per class, ∝ conj(chi_r)
        chi = np.conjugate(v)
        # weighted normalization: sum_i h_i |chi_i|^2 = |G|
        norm = np.sqrt(n / np.sum(h * np.abs(chi) ** 2))
        chi = chi * norm
        # phase so chi(identity class) is real positive (= dimension)
        phase = chi[id_class] / abs(chi[id_class])
        chi = chi / phase
        chars[r] = chi

    dims = np.rint(chars[:, id_class].real).astype(np.int64)
    # deterministic ordering: by dimension, then by character signature
    order = sorted(range(k), key=lambda r: (dims[r],
                   tuple(np.round(chars[r].real, 4)), tuple(np.round(chars[r].imag, 4))))
    chars = chars[order]
    dims = dims[order]

    # names grouped by dimension
    names = []
    seen = {}
    letters = "abcdefghijklmnop"
    for d in dims:
        seen[d] = seen.get(d, 0)
        names.append(f"{d}{letters[seen[d]]}")
        seen[d] += 1

    char_per_element = np.zeros((k, n), dtype=np.complex128)
    for r in range(k):
        char_per_element[r] = chars[r][class_of]

    return {
        "classes": classes, "class_of": class_of, "sizes": sizes,
        "identity": ident, "id_class": id_class,
        "chars": chars, "dims": dims, "names": names,
        "char_per_element": char_per_element,
    }


def char_dict(table, rng_seed=0):
    """Convenience: (char_per_element: name->[n] complex, dims: name->int).

    Drop-in for the dicts that ``experiments.rep_analysis`` consumes, but generic.
    """
    ct = character_table(table, rng_seed=rng_seed)
    cpe = {nm: ct["char_per_element"][r] for r, nm in enumerate(ct["names"])}
    dims = {nm: int(ct["dims"][r]) for r, nm in enumerate(ct["names"])}
    return cpe, dims, ct


def verify_character_table(table, ct=None, tol=1e-6):
    """Sanity checks: orthogonality, sum d^2 = |G|, dims positive integers."""
    n = table.shape[0]
    if ct is None:
        ct = character_table(table)
    chars, dims, h = ct["chars"], ct["dims"], ct["sizes"].astype(float)
    k = len(dims)
    # row orthogonality (weighted by class size): sum_i h_i chi_p(i) conj(chi_q(i)) = n delta
    G = (chars * h) @ chars.conj().T
    orth = np.allclose(G, n * np.eye(k), atol=1e-4 * n)
    sumsq = int(np.sum(dims ** 2))
    return {
        "orthogonal": bool(orth),
        "sum_d2": sumsq,
        "order": n,
        "ok": bool(orth and sumsq == n and np.all(dims >= 1)),
        "dims": sorted(int(d) for d in dims),
    }
