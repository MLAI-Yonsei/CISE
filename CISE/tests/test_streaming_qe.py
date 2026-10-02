import copy

from pymatgen.core import Lattice, Structure
from cise.validation import pipeline as qe, streaming


def test_changed_batch_revisited_after_inference(tmp_path, monkeypatch):
    monkeypatch.setattr(streaming, 'RUNS', tmp_path)
    monkeypatch.setattr(streaming, 'CONFIG', dict(tasks=['wbg']))
    p = tmp_path/'wbg/batches/0001.json'; p.parent.mkdir(parents=True)
    qe.save(p, dict(rows=[dict(status='admission_valid')]))
    seen = {}
    assert streaming.changed_rows(seen)[0]['status'] == 'admission_valid'
    assert streaming.changed_rows(seen) == []
    qe.save(p, dict(rows=[dict(status='evaluated', output_selected=True)]))
    assert streaming.changed_rows(seen)[0]['output_selected']


def test_selection_dedup_highk_upgrade_and_final_reuse(tmp_path, monkeypatch):
    monkeypatch.setattr(qe, 'ROOT', tmp_path)
    cfg = copy.deepcopy(qe.CONFIG); cfg['qe']['pseudo_dir'] = str(tmp_path)
    monkeypatch.setattr(streaming, 'CONFIG', cfg)
    cif = tmp_path/'MgO.cif'
    Structure(Lattice.cubic(4), ['Mg', 'O'], [[0,0,0],[.5,.5,.5]]).to(filename=cif)
    for el in ['mg','o']: (tmp_path/f'{el}.pbesol.UPF').write_text('fixture')
    row = dict(status='evaluated', phase='admission', interval_pass=True, output_selected=True,
               task='wbg', cif_path=str(cif), cif_sha256=qe.sha(cif))
    jobs = streaming.extend_manifest([], [dict(row, output_selected=False), dict(row, status='probe_valid')])
    assert jobs == []
    streaming.extend_manifest(jobs, [row, dict(row, task='sse')])
    assert len(jobs) == 1
    qe.save(qe.summary(jobs[0], 'streaming'), dict(status='completed'))
    qe.save(tmp_path/'streaming_manifest.json', jobs)
    final = dict(jobs[0], index=99)
    assert qe.summary(final, 'final') == qe.summary(jobs[0], 'streaming')
    assert '0099_' in str(qe.summary(dict(final, task_id='highk'), 'final'))
    streaming.extend_manifest(jobs, [dict(row, task='highk'), row])
    assert len(jobs) == 2 and jobs[1]['task_id'] == 'highk'
    qe.save(qe.summary(jobs[1], 'streaming'), dict(status='completed'))
    qe.save(tmp_path/'streaming_manifest.json', jobs)
    assert qe.summary(dict(final, task_id='highk'), 'final') == qe.summary(jobs[1], 'streaming')


def test_streaming_refills_and_drains_without_exceeding_slots(tmp_path, monkeypatch):
    monkeypatch.setattr(qe, 'ROOT', tmp_path)
    (tmp_path/'logs').mkdir()
    cfg = copy.deepcopy(qe.CONFIG); cfg['qe'].update(slots=2, min_disk_gib=0, min_memory_gib=0)
    monkeypatch.setattr(qe, 'CONFIG', cfg)
    monkeypatch.setattr(qe, 'resources', lambda: dict(free_disk_GiB=100, available_memory_GiB=100, external_qe=[]))
    monkeypatch.setattr(qe.time, 'sleep', lambda _: None)
    cif = tmp_path/'fixture.cif'; cif.write_text('fake geometry')
    jobs = [dict(index=i, label=f'j{i}', jid=f'j{i}', audit_cif=str(cif), cif_sha256=qe.sha(cif), task_id='wbg') for i in range(1,5)]
    tick = [0]; active = set(); launches = []; exits = []
    class Proc:
        def __init__(self, cmd, **kwargs):
            self.pid = int(cmd[cmd.index('--row-index')+1]); self.start = tick[0]
            launches.append((self.pid, tick[0])); active.add(self.pid)
            assert len(active) <= 2
        def poll(self):
            if tick[0]-self.start < (4 if self.pid == 1 else 1): return None
            active.remove(self.pid); exits.append((self.pid, tick[0]))
            qe.save(qe.summary(jobs[self.pid-1], 'streaming'), dict(status='completed'))
            return 0
    monkeypatch.setattr(qe.subprocess, 'Popen', Proc)
    def refresh():
        tick[0] += 1
        return jobs[:min(tick[0]+1,4)], tick[0] < 8
    qe.run_jobs([], 'streaming', refresh=refresh)
    assert len(launches) == 4 and not active
    assert dict(launches)[3] < dict(exits)[1]
    rec = qe.load(tmp_path/'streaming_status.json')
    assert rec['completed'] == 4 and rec['status'] == 'EXECUTION_TERMINAL'
