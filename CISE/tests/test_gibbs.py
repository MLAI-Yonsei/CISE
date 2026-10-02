import math

import numpy as np
import pytest
from scipy.optimize import linprog

from cise.cci.gibbs import GibbsCCI


@pytest.mark.parametrize("n,alpha", [(5,.1), (9,.1), (10,.1), (19,.05),
                                      (20,.05), (19,.1), (30,.1), (11,.4)])
def test_constant_basis_is_finite_sample_split_conformal(n, alpha):
    scores = np.arange(1., n + 1)
    out = GibbsCCI(scores, np.ones((n, 1)), alpha).cutoff([1])
    rank = math.ceil((n + 1) * (1 - alpha) - 1e-12)
    if rank > n:
        assert out.status == "unbounded_interval" and np.isposinf(out.cutoff)
    else:
        assert out.status == "ok"
        assert out.cutoff == pytest.approx(scores[rank - 1], abs=1e-7)
    assert out.diagnostics["test_point_correction"]


def direct_augmented_primal(A, S, z, assumed, alpha):

    A = np.vstack((A, z)); S = np.append(S, assumed)
    n, d = A.shape
    obj = np.r_[np.zeros(d), np.full(n, 1-alpha), np.full(n, alpha)]
    eq = np.column_stack((A, np.eye(n), -np.eye(n)))
    out = linprog(obj, A_eq=eq, b_eq=S,
                  bounds=[(None,None)]*d+[(0,None)]*(2*n), method="highs")
    assert out.success
    return float(z @ out.x[:d]), float(out.fun)


@pytest.mark.parametrize("seed", [0,1,2,4,10])
def test_boundary_matches_direct_augmented_pinball(seed):
    rng = np.random.default_rng(seed)
    A = np.column_stack((np.ones(45), rng.normal(size=(45, 2))))
    S = np.abs(rng.normal(size=45)) * (1 + np.abs(A[:,1]))
    z = np.array([1, .3, -.25]); alpha = .15
    core = GibbsCCI(S, A, alpha)
    out = core.cutoff(z)
    assert out.status == "ok"
    for delta in [-1e-3, 1e-3]:
        assumed = out.cutoff + delta
        fitted, loss = direct_augmented_primal(A,S,z,assumed,alpha)
        check = core.augmented(z,assumed)
        assert loss == pytest.approx(check["objective"], abs=1e-7)
        if delta < 0:
            assert assumed <= fitted + 1e-7
            assert check["test_dual"] < 1-alpha-1e-6
        else:
            assert assumed > fitted + 1e-7
            assert check["test_dual"] == pytest.approx(1-alpha)


def test_correction_is_not_calibration_only_quantile_regression():
    scores = np.arange(1.,11)
    core = GibbsCCI(scores,np.ones((10,1)),.1)
    naive = linprog(-scores,A_eq=np.ones((1,10)),b_eq=[0],bounds=(-.1,.9),method="highs")
    naive_prediction = -float(naive.eqlin.marginals[0])
    assert core.cutoff([1]).cutoff == pytest.approx(10)
    assert naive_prediction < core.cutoff([1]).cutoff


def test_heteroscedastic_groups_adapt():
    group = np.repeat([0.,1.],50)
    A = np.column_stack((np.ones(100),group))
    S = np.tile(np.linspace(.1,1,50),2)*(1+4*group)
    core = GibbsCCI(S,A,.1)
    low, high = core.cutoff([1,0]), core.cutoff([1,1])
    assert low.status == high.status == "ok"
    assert high.cutoff == pytest.approx(5*low.cutoff)


def test_ties_zero_scores_rank_deficiency_and_permutation():
    S = np.repeat([0.,1.,2.],10)
    A = np.column_stack((np.ones(30), np.ones(30)*2,np.zeros(30)))
    core = GibbsCCI(S,A,.1)
    out = core.cutoff([1,2,0])
    assert out.status == "ok" and out.cutoff == pytest.approx(2)
    assert out.diagnostics["calibration_rank"] == 1
    perm = np.random.default_rng(22).permutation(30)
    assert GibbsCCI(S[perm],A[perm],.1).cutoff([1,2,0]).cutoff == out.cutoff
    assert GibbsCCI(np.zeros(30),A,.1).cutoff([1,2,0]).cutoff == 0
    assert np.isposinf(core.cutoff([1,2,1]).cutoff)


def test_extreme_out_of_support_features_abstain():
    A = np.column_stack((np.ones(30), np.linspace(-1,1,30)))
    out = GibbsCCI(np.ones(30),A,.1).cutoff([1,1e10])
    assert out.status == "unbounded_interval" and np.isposinf(out.cutoff)


def test_contexts_and_input_mutation_do_not_share_state():
    A = np.ones((30,1)); S = np.arange(30.)
    first = GibbsCCI(S,A,.1); old = first.cutoff([1]).cutoff
    S[:] *= 5; A[:] *= 2
    assert first.cutoff([1]).cutoff == old
    assert GibbsCCI(S,A,.1).cutoff([2]).cutoff == pytest.approx(5*old)


def test_signed_scores_are_not_clamped_by_scalar_core():
    out = GibbsCCI(-np.ones(30), np.ones((30,1)),.1).cutoff([1])
    assert out.status == "ok" and out.cutoff == pytest.approx(-1)


def test_invalid_inputs():
    with pytest.raises(ValueError): GibbsCCI([np.nan], [[1]])
    with pytest.raises(ValueError): GibbsCCI([1], [[1]], alpha=0)
    core=GibbsCCI(np.arange(20.),np.ones((20,1)))
    with pytest.raises(ValueError): core.cutoff([np.inf])


def test_indicator_basis_equals_groupwise_finite_sample_conformal():


    sizes = [40, 9, 4]
    groups = np.repeat(np.arange(3), sizes)
    basis = np.eye(3)[groups]
    scores = np.concatenate([np.linspace(0, k + 1, n) for k, n in enumerate(sizes)])
    core = GibbsCCI(scores, basis, .1)
    for group, n in enumerate(sizes):
        result = core.cutoff(np.eye(3)[group])
        rank = math.ceil((n + 1) * .9 - 1e-12)
        expected = np.inf if rank > n else np.sort(scores[groups == group])[rank - 1]
        assert result.cutoff == pytest.approx(expected)
        assert result.diagnostics['test_point_correction']


def test_nontrivial_basis_reparametrization_preserves_cutoffs():
    rng = np.random.default_rng(23)
    basis = np.column_stack((np.ones(80), rng.normal(size=(80, 2))))
    transform = np.array([[1., 2., -1.], [0., 3., 1.], [0., -1., 2.]])
    scores = np.abs(rng.normal(size=80)) * (2 + basis[:, 1] ** 2)
    original = GibbsCCI(scores, basis, .15)
    changed = GibbsCCI(scores, basis @ transform, .15)
    for feature in [[1., 0., 0.], [1., .5, -.4], [1., -.7, .1]]:
        expected = original.cutoff(feature)
        actual = changed.cutoff(np.array(feature) @ transform)
        assert expected.status == actual.status == 'ok'
        assert actual.cutoff == pytest.approx(expected.cutoff, abs=1e-6)
