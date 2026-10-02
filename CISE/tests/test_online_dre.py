import numpy as np
from cise.cci.online_dre import BoundedLSIF,FixedCCIBasis,ratio_basis_diagnostics
from cise.shared.structure_features import STRUCTURE_FEATURES


def rows(values):
    return [dict(id=str(i),features={k:float(x) for k in STRUCTURE_FEATURES}) for i,x in enumerate(values)]


def test_no_shift_source_empirical_distribution():
    source=rows(np.linspace(-3,3,1001))
    model=BoundedLSIF(source);log=model.fit(source)
    ratio=model.predict(source)
    assert log['direction']=='target_over_source'
    assert abs(ratio.mean()-1)<.05
    assert np.std(ratio)<.1
    assert log['source_ratio']['ESS']>950


def test_known_normal_shift_ratio_direction_and_shape():
    rng=np.random.default_rng(938)
    source=rows(rng.normal(size=3000));target=rows(rng.normal(.65,1,3000))
    model=BoundedLSIF(source);model.fit(target)
    x=np.linspace(-1.5,1.5,151)
    ratio=model.predict(rows(x));truth=np.exp(.65*x-.65**2/2)
    assert np.corrcoef(ratio,truth)[0,1]>.98
    assert ratio[-1]>ratio[0]*2
    assert np.mean(np.abs(ratio-truth))<.25
    log=ratio_basis_diagnostics(FixedCCIBasis(source).transform(rows(x)),ratio)
    assert log['augmented_rank']==log['base_rank']+1
    assert log['ratio_projection_relative_l2']>1e-3


def test_redundant_ratio_is_reported_without_noise():
    source=rows(np.linspace(-1,1,100))
    basis=FixedCCIBasis(source).transform(source)
    log=ratio_basis_diagnostics(basis,np.ones(len(source)))
    assert log['augmented_rank']==log['base_rank']
    assert log['ratio_projection_relative_l2']<1e-12


def test_dre_and_fixed_basis_ignore_labels():
    source=rows(np.linspace(-1,1,100));target=rows(np.linspace(0,2,50))
    contaminated=[dict(r,truth=1e99,prediction=-1e99,score=1e99) for r in target]
    left=BoundedLSIF(source);right=BoundedLSIF(source)
    left.fit(target);right.fit(contaminated)
    np.testing.assert_array_equal(left.coefficients,right.coefficients)
    extrapolated = left.predict(rows(np.linspace(1000, 1100, 50)))
    assert np.all((extrapolated >= 0) & (extrapolated <= 10))
    basis=FixedCCIBasis(source)
    np.testing.assert_array_equal(basis.transform(target),basis.transform(contaminated))


def test_cached_transforms_do_not_reuse_previous_context_weights(monkeypatch):
    from cise.cci import online_dre as dre
    source=rows(np.linspace(-3,3,200))
    cal=[dict(r,score=float(i)/100) for i,r in enumerate(source)]
    monkeypatch.setattr(dre,'calibration_data',lambda prop:dict(rows=cal,dre_source=source))
    dre.fixed_calibration_inputs.cache_clear()
    try:
        first=dre.CurrentWeights('wbg',rows(np.linspace(-3,-1,30)),1,'first',properties=['x'])
        saved=first.weights['x'].copy()
        second=dre.CurrentWeights('wbg',rows(np.linspace(1,3,30)),2,'second',properties=['x'])
        np.testing.assert_array_equal(first.weights['x'],saved)
        assert not np.allclose(first.weights['x'],second.weights['x'])
        np.testing.assert_allclose(second.weights['x'],second.models['x'].predict(cal))
    finally:dre.fixed_calibration_inputs.cache_clear()
