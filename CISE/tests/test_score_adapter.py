import numpy as np
import pytest

from cise.cci.score_adapter import (
    ABSOLUTE_QR, UPPER_QR, LOWER_QR, CQR, calibration_score, inverse_bounds)


def test_pv_upper_signed_score_and_inverse_are_exact_on_dense_grid():
    prediction=-.4;scale=.2;base=-.25;delta=.75
    upper=inverse_bounds(prediction,scale,base+delta,UPPER_QR,['upper'])['upper']
    for truth in np.linspace(-2.,1.,3001):
        score=calibration_score(truth,prediction,scale,UPPER_QR,base)
        assert (score <= delta) == (truth <= upper)


def test_signed_negative_cutoff_is_valid_but_negative_absolute_radius_is_empty():
    assert inverse_bounds(-1.,.2,-1.,UPPER_QR,['upper']) == {'upper':-1.2}
    assert inverse_bounds(1.,.2,-1.,LOWER_QR,['lower']) == {'lower':1.2}
    assert inverse_bounds(1.,.2,-1.,ABSOLUTE_QR,['lower','upper']) is None


def test_cqr_inverse_matches_joint_score_and_rejects_crossing():
    prediction=3.;scale=2.;base={'lower':.4,'upper':1.1};delta=.2
    bounds=inverse_bounds(prediction,scale,delta,CQR,['lower','upper'],base)
    for truth in np.linspace(-3.,10.,2601):
        score=calibration_score(truth,prediction,scale,CQR,(base['lower'],base['upper']))
        assert (score <= delta+1e-12) == (bounds['lower']-1e-12 <= truth <= bounds['upper']+1e-12)
    assert inverse_bounds(0.,1.,-2.,CQR,['lower','upper'],base) is None


@pytest.mark.parametrize('definition,sides',[(UPPER_QR,['lower','upper']),(LOWER_QR,['upper'])])
def test_signed_tail_cannot_be_silently_used_for_wrong_endpoint(definition,sides):
    assert inverse_bounds(0.,1.,1.,definition,sides) is None
