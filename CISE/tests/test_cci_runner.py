from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock,patch
import copy
import hashlib
import pickle
from cise.runtime import save, load
import pytest
from cise.cci import runner


def exercise(state, fail=None, invalid_phase=None):
    cfg={'iterations':100,'cci':{'probe_min_valid':8,'admission_min_valid':2,'max_attempts':3}}
    def batch(task,state,ctx,phase,*args):
        call=state['next_call']
        if call==fail:raise RuntimeError('INTERRUPTED_BEFORE_CALL')
        return [dict(candidate_id=f'{call}_{i}',fingerprint=f'{call}_{i}',cif_sha256=f'{call}_{i}',phase=phase,
            status='rejected' if phase==invalid_phase else 'probe_valid' if phase=='probe' else 'admission_valid',score=1, candidate_ordinal=i,error='invalid_structure') for i in range(2)]
    with patch.object(runner,'CONFIG',cfg),patch.object(runner,'context',return_value={'island':0,'prompt':'fixture'}),\
         patch.object(runner,'batch',side_effect=batch),patch.object(runner,'make_calibration',return_value=SimpleNamespace(logs={})),\
         patch.object(runner,'evaluate_admissions',side_effect=lambda task,rows,*args:[dict(r,status='evaluated') for r in rows]),\
         patch.object(runner,'persist_state'),patch.object(runner,'save'),patch.object(runner,'status'):
        runner.evolve_iterations('wbg',state,Path('unused'),[],Path('unused'))


def test_100_iterations_two_slots_and_resume_without_probe_pollution():
    state=dict(iteration=0,completed_iterations=0,next_call=1,seen=set(),buffer=Mock())
    with pytest.raises(RuntimeError,match='INTERRUPTED'):exercise(state,64)
    assert state['completed_iterations']==12 and state['iteration']==13
    assert len(state['pending']['probes'])==6
    assert len(state['seen'])==24
    exercise(state)
    assert state['completed_iterations']==100 and state['next_call']-1==500
    assert state['buffer'].register.call_count==200
    assert len(state['seen'])==200
    assert 'pending' not in state


def test_probe_consume_does_not_touch_seen():
    state=dict(next_call=1,seen={'main'})
    runner.consume(state,[dict(phase='probe',status='probe_valid',fingerprint='probe')])
    assert state==dict(next_call=2,seen={'main'})


@pytest.mark.parametrize('phase,expected_calls,error',[
    ('probe',12,'CCI_PROBE_ATTEMPTS_EXHAUSTED'),
    ('admission',15,'CCI_ADMISSION_ATTEMPTS_EXHAUSTED'),
])
def test_invalid_generation_stops_without_committing_iteration(phase,expected_calls,error):
    state=dict(iteration=0,completed_iterations=0,next_call=1,seen=set(),buffer=Mock())
    with pytest.raises(RuntimeError,match=error):
        exercise(state,invalid_phase=phase)
    assert state['next_call']-1==expected_calls
    assert state['completed_iterations']==0
    assert state['seen']==set()
    state['buffer'].register.assert_not_called()
    assert 'pending' in state

    with pytest.raises(RuntimeError,match=error):
        exercise(state,invalid_phase=phase)
    assert state['next_call']-1==expected_calls


class CheckpointBuffer:
    fail_on_second = False
    def __init__(self):
        self.registered = []
    def register(self, island, row, scores, iteration):
        self.registered.append(row['candidate_id'])
        if self.fail_on_second and len(self.registered) == 2:
            raise RuntimeError('injected_memory_commit_failure')


class NoModelServer:
    def __init__(self, *args):
        pass
    def close(self):
        pass


@pytest.fixture
def resume_case(tmp_path, monkeypatch):
    task_root = tmp_path / 'runs'
    run = task_root / 'wbg'
    for name in ['iterations', 'cifs', 'prediction_cache', 'api', 'batches', 'dre']:
        (run / name).mkdir(parents=True, exist_ok=True)
    config = copy.deepcopy(runner.CONFIG)
    config['iterations'] = 1
    config['cci']['probe_min_valid'] = 2
    monkeypatch.setattr(runner, 'ROOT', task_root)
    monkeypatch.setattr(runner, 'CONFIG', config)
    monkeypatch.setattr(runner, 'Server', NoModelServer)
    monkeypatch.setattr('cise.provenance.verify_run', lambda: None)
    monkeypatch.setattr(runner, 'make_calibration', lambda *a: SimpleNamespace(logs={}))
    def no_request(*args, **kwargs):
        pytest.fail('Resume issued another API request despite a durable admission batch')
    monkeypatch.setattr(runner, 'request', no_request)
    buffer = CheckpointBuffer()
    ctx = dict(task='wbg', iteration=1, island=0, parents=[], attempt=0,
               buffer_sha256=hashlib.sha256(pickle.dumps(buffer)).hexdigest())
    frozen = runner.freeze_hash(ctx)
    state = dict(iteration=1, completed_iterations=0, next_call=2, buffer=buffer,
                 seen=set(), pending=dict(context=ctx, frozen_hash=frozen,
                 probes=[dict(candidate_id='p0'), dict(candidate_id='p1')],
                 attempt=0, attempt_records=[], best=None))
    checkpoint = run / 'online_state.pkl'
    runner.persist_state(checkpoint, state)
    rows = [dict(candidate_id=f'candidate{i}', task='wbg', phase='admission',
            iteration=1, call_sequence=2, candidate_ordinal=i,
            frozen_hash=frozen, status='admission_valid', output_selected=False,
            fingerprint=f'fp{i}', cif_sha256=f'hash{i}') for i in range(2)]
    save(run / 'batches/0002.json', dict(task='wbg', phase='admission', rows=rows))
    return run, checkpoint


def complete_saved_rows(run, rows):
    records = [dict(r, status='evaluated', score=0., proxy_pass=False) for r in rows]
    batch = load(run / 'batches/0002.json')
    batch['rows'] = records
    save(run / 'batches/0002.json', batch)
    return records


def test_python_evaluation_failure_resumes_saved_batch_without_api(resume_case, monkeypatch):
    run, checkpoint = resume_case
    observed = []
    def failing_evaluation(task, rows, calibration, servers, where):
        batch = load(where / 'batches/0002.json')
        batch['rows'][0].update(status='evaluated', score=0., proxy_pass=False)
        save(where / 'batches/0002.json', batch)
        raise RuntimeError('injected_second_candidate_evaluation_failure')
    monkeypatch.setattr(runner, 'evaluate_admissions', failing_evaluation)
    with pytest.raises(RuntimeError, match='second_candidate'):
        runner.run('wbg')
    with checkpoint.open('rb') as handle:
        restored = pickle.load(handle)
    assert restored['next_call'] == 2
    assert restored['completed_iterations'] == 0
    def success(task, rows, calibration, servers, where):
        observed.extend(r['status'] for r in rows)
        return complete_saved_rows(where, rows)
    monkeypatch.setattr(runner, 'evaluate_admissions', success)
    runner.run('wbg')
    assert observed == ['evaluated', 'admission_valid']
    with checkpoint.open('rb') as handle:
        completed = pickle.load(handle)
    assert completed['completed_iterations'] == 1
    assert completed['buffer'].registered == ['candidate0', 'candidate1']
    assert completed['next_call'] == 3


def test_memory_commit_failure_does_not_persist_partial_archive(resume_case, monkeypatch):
    run, checkpoint = resume_case
    monkeypatch.setattr(runner, 'evaluate_admissions', lambda t, r, c, s, w: complete_saved_rows(w, r))
    monkeypatch.setattr(CheckpointBuffer, 'fail_on_second', True)
    with pytest.raises(RuntimeError, match='memory_commit'):
        runner.run('wbg')
    with checkpoint.open('rb') as handle:
        restored = pickle.load(handle)
    assert restored['buffer'].registered == []
    assert restored['seen'] == set()
    assert restored['completed_iterations'] == 0
    monkeypatch.setattr(CheckpointBuffer, 'fail_on_second', False)
    runner.run('wbg')
    with checkpoint.open('rb') as handle:
        completed = pickle.load(handle)
    assert completed['buffer'].registered == ['candidate0', 'candidate1']
    assert completed['seen'] == {'fp0', 'fp1'}
    assert completed['completed_iterations'] == 1
