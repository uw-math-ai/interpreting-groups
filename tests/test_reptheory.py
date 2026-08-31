"""Pytest wrappers for the representation-theory self-tests (plan P0).

Each core module carries an exhaustive _self_test() verifying homomorphism,
unitarity/orthogonality, characters, center action, FVE recovery of planted
sectors, encoding invariances, and numpy/torch encoding agreement. CI runs
them all; `uv run pytest tests/ -q`.
"""
import pytest


def test_su2():
    from core.su2 import _self_test
    _self_test()


def test_su3():
    from core.su3 import _self_test
    _self_test()


def test_spin4():
    from core.spin4 import _self_test
    _self_test()


def test_spin5():
    from core.spin5 import _self_test
    _self_test()


def test_schur_block_dimensions():
    """Real coefficient-block dims follow (dim_R W)^2 / dim_R End_G(W)."""
    import numpy as np
    from core.su2 import haar, spin_basis
    from core.su3 import haar_su3, sector_basis
    rng = np.random.default_rng(0)
    q = haar(3072, rng)
    b = spin_basis(q, j_list=(0.5, 1.0))
    assert b[0.5].shape[1] == 4      # H type: 16/4
    assert b[1.0].shape[1] == 9      # R type: 9/1
    U = haar_su3(3072, rng)
    b3 = sector_basis(U, sectors=((1, 0), (1, 1)))
    assert b3[(1, 0)].shape[1] == 18   # C type pair: 36/2
    assert b3[(1, 1)].shape[1] == 64   # R type: 64/1


def test_carrier_search_reproduces_known_selections():
    import itertools
    from experiments.carrier_search import (cost_lin, cost_quad, group_forms,
                                            is_faithful)
    want = {"SU2": {"j1/2"}, "SU2/Z2": {"j1"}, "SU3": {"3"},
            "SU3/Z3": {"8"}, "Sp2": {"4"}, "Sp2/Z2": {"5"}}
    for name, center, reps, quotient, mods in group_forms():
        if name not in want:
            continue
        best = (float("inf"), None)
        for r in range(1, 3):
            for carrier in itertools.combinations(reps, r):
                if not is_faithful(carrier, center, quotient, mods, len(mods)):
                    continue
                c = sum(cost_quad(d, t) for _, d, _, t, _ in carrier)
                if c < best[0]:
                    best = (c, carrier)
        assert {x[0] for x in best[1]} == want[name], (name, best)
