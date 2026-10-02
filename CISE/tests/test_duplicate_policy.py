import hashlib
import json
import pickle
from pathlib import Path
from types import SimpleNamespace

import pytest

from cise.cci import runner
from cise.llema import runner as llema_runner


@pytest.fixture
def batch_case(tmp_path, monkeypatch):
    parent = tmp_path / 'parent.cif'
    parent.write_text('parent')
    monkeypatch.setattr(runner, 'validate_response', lambda value, schema=None: value)
    monkeypatch.setattr(runner, 'features', lambda path: {'volume': 1.})
    monkeypatch.setattr(runner, 'fingerprint', lambda path: path.read_text())
    def apply_edit(parent_path, proposal, task, path):
        path.write_text(proposal['fingerprint'])
        return {'formula': 'Si'}
    monkeypatch.setattr(runner, 'apply_edit', apply_edit)

    def execute(fingerprints, *, phase='admission', seen=(), external=(), task='wbg'):
        run = tmp_path / task
        for directory in ['api', 'batches', 'cifs']:
            (run / directory).mkdir(parents=True, exist_ok=True)
        buffer = {'memory': []}
        ctx = dict(iteration=1, attempt=0, messages=[],
            parents=[dict(parent_id='parent', path=str(parent), sha256=runner.sha(parent))],
            buffer_sha256=hashlib.sha256(pickle.dumps(buffer)).hexdigest())
        state = dict(buffer=buffer, next_call=1, seen=set(seen),
            static_exclusions=set(external), pending={'frozen_hash': runner.freeze_hash(ctx)})
        proposals = [dict(parent_id='parent', fingerprint=fp, substitutions=[], lattice_scale=1.)
                     for fp in fingerprints]
        monkeypatch.setattr(runner, 'request', lambda *a, **k: (json.dumps(proposals), {'call_id': 'free-test'}))
        rows = runner.batch(task, state, ctx, phase, {}, run)
        return rows, state
    return execute


def test_external_dataset_and_old_llema_overlap_is_allowed(batch_case):
    rows, state = batch_case(['old_llema', 'external_calibration'],
        external=['old_llema', 'external_calibration'])
    assert [r['status'] for r in rows] == ['admission_valid', 'admission_valid']
    assert state['seen'] == set()


def test_prior_main_candidate_in_same_task_is_rejected(batch_case):
    rows, _ = batch_case(['prior_main', 'new'], seen=['prior_main'])
    assert rows[0]['status'] == 'rejected'
    assert rows[0]['error'] == 'duplicate_geometry_species_within_task'
    assert rows[1]['status'] == 'admission_valid'


def test_repeated_geometry_within_admission_batch_is_rejected(batch_case):
    rows, _ = batch_case(['same', 'same'])
    assert rows[0]['status'] == 'admission_valid'
    assert rows[1]['status'] == 'rejected'
    assert rows[1]['error'] == 'duplicate_geometry_species_within_task'


def test_probe_repeats_and_task_seen_and_external_overlap_are_retained(batch_case):
    rows, state = batch_case(['same', 'same'], phase='probe', seen=['same'], external=['same'])
    assert [r['status'] for r in rows] == ['probe_valid', 'probe_valid']
    assert state['seen'] == {'same'}


def test_main_duplicates_are_isolated_between_tasks(batch_case):
    wbg, _ = batch_case(['wbg_prior', 'new'], task='wbg', seen=['wbg_prior'])
    pv, _ = batch_case(['wbg_prior', 'new'], task='pv', seen=[])
    assert wbg[0]['status'] == 'rejected'
    assert pv[0]['status'] == 'admission_valid'


def test_initial_seed_geometry_is_rejected_from_task_seen(batch_case):
    rows, state = batch_case(['initial_seed', 'new'], seen=['initial_seed'])
    assert rows[0]['status'] == 'rejected'
    assert rows[0]['error'] == 'duplicate_geometry_species_within_task'
    assert rows[1]['status'] == 'admission_valid'
    assert state['seen'] == {'initial_seed'}


@pytest.fixture
def initialized_search(tmp_path, monkeypatch):

    prototypes = tmp_path / 'prototypes'
    prototypes.mkdir()
    specs = [
        ('rocksalt_MgO', 'initial_seed', 'wbg', True, False),
        ('finite_failure', 'failing_seed', 'wbg', False, False),
        ('unbounded_seed', 'abstained_seed', 'wbg', False, True),
        ('diamond_Si', 'other_task_seed', 'pv', True, False),
    ]
    seed_rows = {}
    manifest = []
    for cid, fp, task, passed, abstained in specs:
        path = prototypes / (cid + '.cif')
        path.write_text(fp)
        manifest.append(dict(path=str(path), eligible_tasks=[task]))
        seed_rows[str(path)] = dict(candidate_id=cid, fingerprint=fp,
            cif_path=str(path), cif_sha256=runner.sha(path), formula=cid,
            score=1., proxy_pass=passed, interval_pass=passed, abstained=abstained,
            failed_constraints=[], conformal_bounds={}, property_values={})
    runner.save(prototypes / 'manifest.json', manifest)
    monkeypatch.setattr('cise.provenance.verify_run', lambda: None)
    monkeypatch.setattr(llema_runner, 'verify_run', lambda: None)

    def descriptor(path, **kwargs):
        path = Path(path)
        return dict(parent_id=path.stem, path=str(path), sha256=runner.sha(path), formula='Si')

    def evaluate(task, desc, *args, **kwargs):
        if desc['path'] in seed_rows:
            return dict(seed_rows[desc['path']], status='evaluated', task=task)
        path = Path(desc['path'])
        return dict(candidate_id=desc['parent_id'], fingerprint=path.read_text(),
            cif_path=str(path), cif_sha256=runner.sha(path), formula='Si', score=1.,
            proxy_pass=True, abstained=False, failed_constraints=[], property_values={},
            status='evaluated', task=task)

    def apply_edit(parent, proposal, task, path):
        path.write_text(proposal['fingerprint'])
        return descriptor(path)

    for module in [runner, llema_runner]:
        monkeypatch.setattr(module, 'ROOT', tmp_path / module.__package__ / 'runs')
        monkeypatch.setattr(module, 'PROTOTYPES', prototypes)
        monkeypatch.setattr(module, 'Server', lambda *a: SimpleNamespace(close=lambda: None))
        monkeypatch.setattr(module, 'parent_descriptor', descriptor)
        monkeypatch.setattr(module, 'evaluate', evaluate)
        monkeypatch.setattr(module, 'ExperienceBuffer', InitializationBuffer)
        monkeypatch.setattr(module, 'fingerprint', lambda path: path.read_text())
        monkeypatch.setattr(module, 'apply_edit', apply_edit)
        monkeypatch.setattr(module, 'validate_response', lambda value, *a: value)
        monkeypatch.setattr(module, 'status', lambda *a: None)
    monkeypatch.setattr(llema_runner, 'build_prompt', lambda *a: 'offline test prompt')

    def execute(method, *, task='wbg', fingerprints=None):
        module = runner if method == 'cci' else llema_runner
        cfg = dict(module.CONFIG, iterations=0 if fingerprints is None else 1)
        cfg['llema'] = dict(cfg['llema'], max_attempts=1)
        monkeypatch.setattr(module, 'CONFIG', cfg)
        if fingerprints is not None:
            assert method == 'llema'
            initial_id = 'rocksalt_MgO' if task == 'wbg' else 'diamond_Si'
            proposals = [dict(parent_id=initial_id, fingerprint=fp, substitutions=[], lattice_scale=1.)
                         for fp in fingerprints]
            monkeypatch.setattr(module, 'request', lambda *a: (json.dumps(proposals), {'call_id': 'free-test'}))
        (module.run if method == 'cci' else module.generate)(task)
        run = module.ROOT / task
        checkpoint = run / ('online_state.pkl' if method == 'cci' else 'state.pkl')
        with checkpoint.open('rb') as handle:
            state = pickle.load(handle)
        return state, run
    return execute


@pytest.mark.parametrize('method', ['cci', 'llema'])
def test_real_initialization_sees_every_task_eligible_seed(initialized_search, method):
    state, _ = initialized_search(method)

    assert state['seen'] == {'initial_seed', 'failing_seed', 'abstained_seed'}
    assert 'static_exclusions' not in state


def test_both_runners_start_with_identical_task_local_seen(initialized_search):
    cci, _ = initialized_search('cci')
    llema, _ = initialized_search('llema')
    pv, _ = initialized_search('cci', task='pv')
    assert cci['seen'] == llema['seen']
    assert pv['seen'] == {'other_task_seed'}
    assert cci['seen'].isdisjoint(pv['seen'])


@pytest.mark.parametrize('seed_fp', ['initial_seed', 'failing_seed', 'abstained_seed'])
def test_both_runners_reject_return_to_any_initial_seed(initialized_search, batch_case, seed_fp):
    cci_state, _ = initialized_search('cci')
    cci_rows, _ = batch_case([seed_fp, 'new'], seen=cci_state['seen'])
    llema_state, run = initialized_search('llema', fingerprints=[seed_fp, 'new'])
    result = runner.load(run / 'iterations/0001.json')
    llema_rows = sorted(result['candidates'], key=lambda row: row['candidate_ordinal'])
    assert cci_rows[0]['status'] == llema_rows[0]['status'] == 'rejected'
    assert cci_rows[0]['error'] in llema_rows[0]['errors'][0]
    assert cci_rows[1]['status'] == 'admission_valid'
    assert llema_rows[1]['status'] == 'evaluated'
    assert llema_state['seen'] == cci_state['seen'] | {'new'}


def test_llema_rejects_within_batch_repeat_and_preserves_seed_seen(initialized_search):
    state, run = initialized_search('llema', fingerprints=['new', 'new'])
    rows = sorted(runner.load(run / 'iterations/0001.json')['candidates'],
                  key=lambda row: row['candidate_ordinal'])
    assert [row['status'] for row in rows] == ['evaluated', 'rejected']
    assert 'duplicate_geometry_species_within_task' in rows[1]['errors'][0]
    assert state['seen'] == {'initial_seed', 'failing_seed', 'abstained_seed', 'new'}


def test_initial_seed_repeats_remain_valid_probes(initialized_search, batch_case):
    state, _ = initialized_search('cci')
    rows, after = batch_case(['initial_seed', 'initial_seed'], phase='probe', seen=state['seen'])
    assert [row['status'] for row in rows] == ['probe_valid', 'probe_valid']
    assert after['seen'] == state['seen']


class InitializationBuffer:
    def __init__(self, **kwargs):
        self.rows = []
    def register(self, island, row, score, iteration):
        self.rows.append(row)
    def get_prompt(self, **kwargs):
        return SimpleNamespace(island_id=0)
    def get_unique_examples_for_evolution(self, *args, **kwargs):
        return dict(success=[], failure=[])
