import copy
import hashlib
import pickle
from types import SimpleNamespace

import pytest

from cise.cci import runner
from cise.runtime import load, save


class RecordingBuffer:
    def __init__(self, *args, **kwargs):
        self.registered = []

    def register(self, island, row, scores, iteration=0):
        self.registered.append((island, row['candidate_id'], scores['total']))

    def get_prompt(self, iteration=0):
        return SimpleNamespace(island_id=0)

    def get_unique_examples_for_evolution(self, island, max_examples=20):
        return {'success': [], 'failure': []}


class NoModelServer:
    def __init__(self, *args):
        pass

    def close(self):
        pass


def seed(candidate_id, abstained=False):
    return dict(candidate_id=candidate_id, abstained=abstained,
                cif_path=candidate_id + '.cif', cif_sha256='sha-' + candidate_id,
                fingerprint='fp-' + candidate_id, formula=candidate_id,
                property_values={'band_gap': None if abstained else 2.,
                                 'formation_energy': None if abstained else -2.},
                conformal_bounds={'band_gap': {'lower': None if abstained else 2.}},
                proxy_pass=False, output_selected=False, proxy_hit=True,
                failed_constraints=['unbounded_interval' if abstained else 'band_gap'],
                score=-10. if abstained else 0.)


@pytest.fixture
def isolated_runner(tmp_path, monkeypatch):
    config = copy.deepcopy(runner.CONFIG)
    config['iterations'] = 1
    config['cci']['probe_min_valid'] = 2
    monkeypatch.setattr(runner, 'CONFIG', config)
    monkeypatch.setattr(runner, 'ROOT', tmp_path / 'runs')
    monkeypatch.setattr(runner, 'PROTOTYPES', tmp_path / 'prototypes')
    monkeypatch.setattr(runner, 'Server', NoModelServer)
    monkeypatch.setattr('cise.provenance.verify_run', lambda: None)

    def no_request(*args, **kwargs):
        pytest.fail('An offline archive/seed test attempted an LLM request')

    monkeypatch.setattr(runner, 'request', no_request)
    monkeypatch.setattr(runner, 'make_calibration', lambda *args: SimpleNamespace(logs={}))
    return tmp_path


@pytest.mark.parametrize('abstentions', [(True, False), (True, True)])
def test_saved_evaluations_exclude_abstentions_but_preserve_scientific_records(
        isolated_runner, monkeypatch, abstentions):

    run = runner.ROOT / 'wbg'
    for name in ['batches', 'dre', 'iterations']:
        (run / name).mkdir(parents=True, exist_ok=True)
    buffer = RecordingBuffer()
    ctx = dict(task='wbg', iteration=1, island=0, parents=[], attempt=0,
               buffer_sha256=hashlib.sha256(pickle.dumps(buffer)).hexdigest())
    frozen = runner.freeze_hash(ctx)
    rows = []
    for ordinal, abstained in enumerate(abstentions):
        row = seed('candidate' + str(ordinal), abstained)
        row.update(task='wbg', phase='admission', status='evaluated', iteration=1,
                   candidate_ordinal=ordinal, call_sequence=2, frozen_hash=frozen)
        rows.append(row)
    save(run / 'batches/0002.json', dict(task='wbg', phase='admission', rows=rows))
    state = dict(iteration=1, completed_iterations=0, next_call=2, buffer=buffer,
                 seen={'initial-seed'}, pending=dict(context=ctx, frozen_hash=frozen,
                 probes=[dict(candidate_id='p0'), dict(candidate_id='p1')],
                 attempt=0, attempt_records=[], admission_batches=0))
    checkpoint = run / 'online_state.pkl'
    runner.persist_state(checkpoint, state)

    runner.run('wbg')
    with checkpoint.open('rb') as handle:
        completed = pickle.load(handle)
    expected = ['candidate' + str(i) for i, abstained in enumerate(abstentions)
                if not abstained]
    assert [item[1] for item in completed['buffer'].registered] == expected

    assert all(item[2] == 0. for item in completed['buffer'].registered)
    assert completed['seen'] == {'initial-seed', 'fp-candidate0', 'fp-candidate1'}
    assert completed['completed_iterations'] == 1
    assert completed['next_call'] == 3
    assert load(run / 'iterations/0001.json')['candidates'] == rows
    assert load(run / 'batches/0002.json')['rows'] == rows
    status = load(run / 'status.json')
    assert status['evaluated'] == 2
    assert status['abstentions'] == sum(abstentions)
    assert status['unique_outputs'] == 0
    assert status['status'] == 'GENERATION_COMPLETED'
    runner.run('wbg')
    with checkpoint.open('rb') as handle:
        resumed = pickle.load(handle)
    assert resumed['buffer'].registered == completed['buffer'].registered
    assert resumed['seen'] == completed['seen']


def prepare_seeds(monkeypatch, rows):
    save(runner.PROTOTYPES / 'manifest.json',
         [dict(path=row['cif_path'], eligible_tasks=['wbg']) for row in rows])
    by_path = {row['cif_path']: row for row in rows}
    monkeypatch.setattr(runner, 'parent_descriptor', lambda path, task: {'path': path})
    monkeypatch.setattr(runner, 'evaluate',
                        lambda task, desc, *args: copy.deepcopy(by_path[desc['path']]))
    monkeypatch.setattr(runner, 'ExperienceBuffer', RecordingBuffer)


@pytest.mark.parametrize('preferred_abstained', [False, True])
def test_seed_predictions_preserved_and_only_usable_seed_pool_is_adopted(
        isolated_runner, monkeypatch, preferred_abstained):
    rows = [seed('rocksalt_MgO', preferred_abstained), seed('fallback'), seed('bad', True)]
    prepare_seeds(monkeypatch, rows)
    captured = {}

    def capture(task, state, *args):
        captured.update(state)

    monkeypatch.setattr(runner, 'evolve_iterations', capture)
    runner.run('wbg')
    run = runner.ROOT / 'wbg'
    recorded = load(run / 'seed_predictions.json')
    assert [r['candidate_id'] for r in recorded] == [r['candidate_id'] for r in rows]
    assert all(r['coverage_role'] == 'initialization_excluded' for r in recorded)
    assert all(not r['output_selected'] for r in recorded)
    assert [r['candidate_id'] for r in captured['seeds']] == [
        r['candidate_id'] for r in rows if not r['abstained']]
    expected_base = 'fallback' if preferred_abstained else 'rocksalt_MgO'
    assert captured['buffer'].registered == [(i, expected_base, 0.) for i in range(5)]
    assert captured['seen'] == {r['fingerprint'] for r in rows}


def test_all_abstained_seeds_fail_before_generation(isolated_runner, monkeypatch):
    rows = [seed('rocksalt_MgO', True), seed('bad', True)]
    prepare_seeds(monkeypatch, rows)

    def no_evolution(*args):
        pytest.fail('No usable seeds must stop before evolution')

    monkeypatch.setattr(runner, 'evolve_iterations', no_evolution)
    with pytest.raises(RuntimeError, match='CCI_NO_USABLE_SEEDS'):
        runner.run('wbg')
    run = runner.ROOT / 'wbg'
    assert len(load(run / 'seed_predictions.json')) == 2
    assert not (run / 'online_state.pkl').exists()
    assert load(run / 'status.json')['status'] == 'GENERATION_BLOCKED'


def test_context_defensively_excludes_abstained_seed_parents(isolated_runner, monkeypatch):
    visited = []

    def descriptor(path, task):
        visited.append(path)
        return dict(parent_id=path, path=path)

    monkeypatch.setattr(runner, 'parent_descriptor', descriptor)
    monkeypatch.setattr(runner, 'build_prompt', lambda *args: 'offline prompt')
    monkeypatch.setattr(runner, 'annotate_prompt', lambda prompt, task: prompt)
    monkeypatch.setattr(runner, 'sha', lambda path: 'offline seal')
    state = dict(buffer=RecordingBuffer(), seeds=[seed('bad', True), seed('usable')],
                 seen={'fp-bad', 'fp-usable'})
    ctx = runner.context('wbg', state, 1)
    assert visited == ['usable.cif']
    assert len(ctx['parents']) == 1
    assert ctx['parents'][0]['property_values']['band_gap'] == 2.
