from unittest.mock import patch

import joblib
import numpy as np
import pytest
from sklearn.dummy import DummyRegressor

from cise.cci.context_calibration import ContextCalibration
from cise.cci.gibbs import CutoffResult, GibbsCCI
from cise.cci.quantile_model import ADJUSTED_SCORE, FrozenQuantileModel
from cise.cci.score_adapter import ABSOLUTE, CQR, inverse_bounds
from cise.runtime import CONFIG
from cise.shared.structure_features import STRUCTURE_FEATURES


def frozen(tmp_path,constant=50.):
    path=tmp_path/"interval";path.mkdir(exist_ok=True)
    estimator=DummyRegressor(strategy="constant",constant=constant).fit([[0]],[0])
    joblib.dump(estimator,path/"band_gap_quantile.joblib")
    return path


def fixture_data():
    features={name:0. for name in STRUCTURE_FEATURES}
    rows=[dict(id=str(i),features=features,score=float(i)-50.,raw_score=float(i),base_quantile=50.) for i in range(1,31)]
    return dict(rows=rows,dre_source=[dict(id="source",features=features)],
                score_definition=ADJUSTED_SCORE,quantile_model="band_gap_quantile.joblib"),features


def test_constant_offset_equivariance_and_negative_delta(tmp_path):
    frozen(tmp_path);data,features=fixture_data()
    with patch("cise.cci.context_calibration.EXPERIMENT",tmp_path), \
         patch("cise.cci.context_calibration.calibration_data",return_value=data), \
         patch.dict(CONFIG["cci"],quantile_offset=True,alpha_total=.05):
        ctx=ContextCalibration("wbg",["band_gap"],None,0,"ctx",False)
        actual=ctx.cutoff("band_gap",features,3.)
        expected=GibbsCCI(np.arange(1.,31),np.ones((30,1)),.05).cutoff([1])
        assert actual.cutoff==pytest.approx(expected.cutoff)
        assert actual.diagnostics["gibbs_residual_cutoff"]<0
        assert actual.diagnostics["base_quantile"]==50.
        assert ctx.cutoff_without_dre("band_gap",features,3.).cutoff==actual.cutoff


def test_offset_does_not_clamp_empty_inverse(tmp_path):
    path=frozen(tmp_path,constant=2.);data,features=fixture_data()
    model=FrozenQuantileModel(path,"band_gap",data)
    ctx=ContextCalibration.__new__(ContextCalibration)
    ctx.quantile_models={"band_gap":model}
    out=ctx._add_offset("band_gap",features,3.,CutoffResult(-3.,"ok",{}))
    assert out.cutoff==-1.
    assert out.diagnostics["gibbs_residual_cutoff"]==-3.


def test_metadata_requires_frozen_model_and_config_agreement(tmp_path):
    path=frozen(tmp_path);data,features=fixture_data()
    with pytest.raises(ValueError,match="missing"):
        FrozenQuantileModel(path,"band_gap",dict(score_definition=ADJUSTED_SCORE))
    with pytest.raises(ValueError,match="explicitly adjusted"):
        FrozenQuantileModel(path,"band_gap",dict(quantile_model="band_gap_quantile.joblib"))
    with patch("cise.cci.context_calibration.EXPERIMENT",tmp_path), \
         patch("cise.cci.context_calibration.calibration_data",return_value=data), \
         patch.dict(CONFIG["cci"],quantile_offset=False):
        with pytest.raises(ValueError,match="disagree"):
            ContextCalibration("wbg",["band_gap"],None,0,"ctx")


def test_base_quantile_clipping_is_fixed_but_prediction_is_required(tmp_path):
    path=frozen(tmp_path,constant=-2.);data,features=fixture_data()
    model=FrozenQuantileModel(path,"band_gap",data)
    assert model.predict(features,3.)==0.
    with pytest.raises(ValueError,match="raw proxy"):
        model.predict(features)


def test_zero_offset_ignores_incidental_model_file(tmp_path):
    path=frozen(tmp_path);_,features=fixture_data()
    model=FrozenQuantileModel(path,"band_gap",{})
    assert not model.enabled and model.predict(features)==0.


def test_absolute_score_preserves_physical_inverse(tmp_path):
    model=FrozenQuantileModel(tmp_path,"band_gap",
        {"score_definition":"absolute_residual_over_local_scale"})
    assert model.score_definition == ABSOLUTE
    assert not model.enabled
    assert inverse_bounds(3.,2.,.5,model.manifest()["score_definition"],
                          ["lower","upper"]) == {"lower":2.,"upper":4.}


def test_cqr_loads_two_hashed_signed_endpoint_models(tmp_path):
    path=tmp_path/'interval';path.mkdir();_,features=fixture_data()
    hashes={};names={}
    import hashlib
    for side,value in [('lower',.3),('upper',.8)]:
        name='band_gap_'+side+'_quantile.joblib';p=path/name
        joblib.dump(DummyRegressor(strategy='constant',constant=value).fit([[0]],[0]),p)
        names[side]=name;hashes[side]=hashlib.sha256(p.read_bytes()).hexdigest()
    model=FrozenQuantileModel(path,'band_gap',dict(score_definition=CQR,
        quantile_models=names,quantile_models_sha256=hashes))
    assert model.predict(features,1.)=={'lower':.3,'upper':.8}
    assert model.manifest()['quantile_model_sha256']==hashes
