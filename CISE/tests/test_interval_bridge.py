import itertools
from unittest.mock import patch

import numpy as np
import pytest

from cise.cci.gibbs import CutoffResult, GibbsCCI
from cise.cci.interval_bridge import correct
from cise.cci.score_adapter import UPPER_QR
from cise.shared.inference import score_values


class FixedContext:
    iteration = 7
    frozen_hash = "independent-test-context"

    def __init__(self, cutoff=1., status="ok"):
        self.value=cutoff; self.status=status

    def cutoff(self, prop, features, prediction=None):
        return CutoffResult(self.value,self.status,{"test_point_correction":True})

    def ratio(self, prop, features):
        return 1.


class SignedUpperContext(FixedContext):


    def cutoff(self, prop, features, prediction=None):
        definition = UPPER_QR if prop == "formation_energy" else "normalized_absolute_residual"
        value = self.value if prop == "formation_energy" else .5
        return CutoffResult(value, self.status, {
            "test_point_correction": True,
            "score_definition": definition,
        })


def run_adapter(task, raw, scales, context):
    props={p:{side:{} for side in (["lower","upper"] if task == "pv" and p!="formation_energy"
                                  else ["upper"] if p=="formation_energy" else ["lower"])} for p in raw}
    row=dict(task=task,property_values=raw,species=["Li","O"],cif_path="unused")
    with patch("cise.cci.interval_bridge.calibrations",return_value={task:{"properties":props}}), \
         patch("cise.cci.interval_bridge.scales",return_value=scales), \
         patch("cise.cci.interval_bridge.structure_feature_values",return_value={}):
        return correct(row,score_values,context)


def test_actual_gibbs_cutoff_maps_to_adaptive_interval():
    class ActualContext(FixedContext):
        def __init__(self):
            self.core=GibbsCCI(np.arange(1.,20.),np.ones((19,1)),.05)
        def cutoff(self,prop,features,prediction=None):return self.core.cutoff([1])
    out=run_adapter("wbg",dict(band_gap=4.,formation_energy=-3.),
                    dict(band_gap=.1,formation_energy=.05),ActualContext())
    assert out["conformal_bounds"]["band_gap"]["lower"]==pytest.approx(2.1)
    assert out["conformal_bounds"]["formation_energy"]["upper"]==pytest.approx(-2.05)
    assert out["raw_proxy_pass"] and not out["interval_pass"]
    assert out["interval_context"]["iteration"]==7


@pytest.mark.parametrize("task,raw",[
    ("wbg",dict(band_gap=3.,formation_energy=-2.)),
    ("sse",dict(band_gap=3.,formation_energy=-2.)),
    ("pv",dict(band_gap=1.35,formation_energy=-1.))])
def test_corner_minimum_equals_dense_interior_minimum(task,raw):
    scales={p:.2 for p in raw}
    out=run_adapter(task,raw,scales,FixedContext())
    names=list(raw)
    grids=[np.linspace(raw[p]-.2,raw[p]+.2,13) for p in names]
    dense=min(score_values(task,dict(zip(names,vals)),["Li","O"])[0]
              for vals in itertools.product(*grids))
    assert out["score"]==pytest.approx(dense)
    assert out["interval_pass"]


@pytest.mark.parametrize("cutoff,status",[(np.inf,"unbounded_interval"),
                                           (np.nan,"solver_failure"),
                                           (-1.,"ok"),(1.,"solver_failure")])
def test_failed_or_empty_cutoff_never_falls_back_to_raw(cutoff,status):
    out=run_adapter("wbg",dict(band_gap=4.,formation_energy=-3.),
                    dict(band_gap=.1,formation_energy=.1),FixedContext(cutoff,status))
    assert out["raw_proxy_pass"]
    assert out["abstained"] and not out["interval_pass"] and not out["output_eligible"]
    assert all(v is None for v in out["property_values"].values())


@pytest.mark.parametrize("scale",[0.,-1.,np.nan,np.inf])
def test_nonpositive_or_invalid_scale_abstains(scale):
    out=run_adapter("wbg",dict(band_gap=4.,formation_energy=-3.),
                    dict(band_gap=scale,formation_energy=.1),FixedContext())
    assert out["abstained"] and not out["output_eligible"]


def test_pv_signed_upper_negative_cutoff_is_finite_and_can_pass():

    context = SignedUpperContext(cutoff=-1.)
    out = run_adapter("pv", dict(band_gap=1.35, formation_energy=-1.),
                      dict(band_gap=.1, formation_energy=.2), context)
    assert not out["abstained"]
    assert out["conformal_bounds"]["formation_energy"]["upper"] == pytest.approx(-1.2)
    assert out["interval_pass"] and out["output_eligible"]


def test_pv_containment_plus_coverage_cannot_be_a_false_positive():

    out = run_adapter("pv", dict(band_gap=1.35, formation_energy=-1.),
                      dict(band_gap=.1, formation_energy=.2), SignedUpperContext(.5))
    truth = dict(band_gap=1.32, formation_energy=-1.05)
    gap = out["conformal_bounds"]["band_gap"]
    upper = out["conformal_bounds"]["formation_energy"]["upper"]
    assert out["output_eligible"]
    assert gap["lower"] <= truth["band_gap"] <= gap["upper"]
    assert truth["formation_energy"] <= upper
    assert .7 <= truth["band_gap"] <= 2. and truth["formation_energy"] <= 0.


@pytest.mark.parametrize('task,species,limits', [
    ('wbg', ['Mg', 'O'], ((0., 5.), (-3., 2.))),
    ('sse', ['Li', 'O'], ((0., 5.), (-3., 2.))),
    ('pv', ['Si', 'O'], ((0., 4.), (-3., 2.))),
])
def test_original_task_reward_has_corner_minimum(task, species, limits):

    rng = np.random.default_rng(105)
    names = ['band_gap', 'formation_energy']
    for _ in range(12):
        interval = [np.sort(rng.uniform(lo, hi, 2)) for lo, hi in limits]
        corner = min(score_values(task, dict(zip(names, point)), species)[0]
                     for point in itertools.product(*interval))

        for point in itertools.product(*(np.linspace(lo, hi, 13) for lo, hi in interval)):
            assert score_values(task, dict(zip(names, point)), species)[0] >= corner - 1e-10


@pytest.mark.parametrize('task,raw,scale,passed',[
    ('pv',dict(band_gap=1.3,formation_energy=-1),.2,True),
    ('pv',dict(band_gap=1.3,formation_energy=-1),2,False),
    ('pv',dict(band_gap=1.9,formation_energy=-1),1,False),
    ('wbg',dict(band_gap=3,formation_energy=-2),1,True),
    ('sse',dict(band_gap=3,formation_energy=-2),1,True)])
def test_task_endpoint_intersections(task,raw,scale,passed):
    out = run_adapter(task, raw, {prop: scale for prop in raw}, FixedContext(.5))
    assert out['output_eligible'] is passed
