import copy
import json

import pytest
from pymatgen.core import Lattice, Structure
from cise import reporting
from cise.validation import pipeline


def test_missing_formation_reference_preserves_verified_gap(tmp_path, monkeypatch):


    cif = tmp_path / 'MgO.cif'
    Structure(Lattice.cubic(4.2), ['Mg', 'O'], [[0, 0, 0], [.5, .5, .5]]).to(filename=cif)
    structure = Structure.from_file(cif)
    pseudo_dir = tmp_path / 'pseudos'
    pseudo_dir.mkdir()
    pseudos = {}
    for el in ['Mg', 'O']:
        p = pseudo_dir / (el.lower() + '.pbesol.UPF')
        p.write_text('fixture pseudopotential ' + el)
        pseudos[el] = str(p)
    reference = tmp_path / 'refs.json'
    reference.write_text(json.dumps({'Mg': {'total_elemental_Ry': -2.}}))
    config = copy.deepcopy(pipeline.CONFIG)
    config['qe'].update(reference_file=str(reference), pseudo_dir=str(pseudo_dir))
    monkeypatch.setattr(pipeline, 'CONFIG', config)
    folder = tmp_path / 'profile'
    folder.mkdir()
    summary = folder / 'summary.json'
    params = dict(xc='PBEsol', ecutwfc_Ry=45, ecutrho_Ry=250,
                  occupations='Gaussian 0.01 Ry', magnetism={'spin_polarized': False},
                  valence_electrons=4, nbands=4)
    record = dict(status='completed', scf_converged=True,
        cif_sha256=pipeline.sha(cif), energy_eV_atom=-10 * pipeline.RY / 2,
        parameters=params, pseudopotentials=pseudos,
        pseudopotential_sha256={k: pipeline.sha(v) for k, v in pseudos.items()})
    summary.write_text(json.dumps(record))
    (folder / 'scf.out').write_text('! total energy = -10.0 Ry\nconvergence has been achieved\nJOB DONE.\n')
    lattice = '\n'.join(' '.join(str(float(x)) for x in row) for row in structure.lattice.matrix)
    sites = '\n'.join(str(site.specie) + ' ' + ' '.join(str(float(x)) for x in site.frac_coords) for site in structure)
    (folder / 'scf.in').write_text('CELL_PARAMETERS angstrom\n' + lattice + '\nATOMIC_POSITIONS crystal\n' + sites + '\n')
    (folder / 'nscf_fixed.out').write_text('number of k points= 2\n bands (ev):\n -4.0 -1.0 2.0 5.0\n\n bands (ev):\n -3.0 -2.0 3.0 5.0\n\nJOB DONE.\n')
    monkeypatch.setattr(pipeline, 'summary', lambda *a: summary)
    job = dict(jid='fixture', label='fixture', cif_sha256=pipeline.sha(cif), audit_cif=str(cif), task_id='wbg')
    result = pipeline.read_result(job, 'final')
    assert result['gap_eV'] == pytest.approx(3.)
    assert result['formation_eV_atom'] is None
    assert result['geometry_verified'] and result['energy_verified']
    assert result['gap_status'] == 'STRICT_COMPLETED'
    assert any('reference' in str(v).lower() for k, v in result.items() if 'reason' in k or 'status' in k)


def test_all_unavailable_qe_jobs_are_terminal_without_launch(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, 'ROOT', tmp_path)
    job = dict(index=1, label='unavailable', jid='fixture', cif_sha256='fixture')
    pipeline.save(tmp_path / 'validation_unavailable.json', {
        'unavailable': dict(reason='MISSING_PSEUDOPOTENTIAL', elements=['Nd'])})
    monkeypatch.setattr(pipeline.subprocess, 'Popen', lambda *a, **k: pytest.fail('unavailable job launched'))
    pipeline.run_jobs([job], 'final')
    rec = pipeline.load(pipeline.summary(job, 'final'))
    assert rec['status'] == 'VALIDATION_UNAVAILABLE'
    assert rec['scf_converged'] is False
    assert pipeline.load(tmp_path / 'final_status.json')['status'] == 'EXECUTION_TERMINAL'


def test_concurrent_qe_profile_resume_executes_original_once(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from types import SimpleNamespace
    import threading
    import time
    from cise.validation import worker
    args = SimpleNamespace(row_json=json.dumps({'label': 'same-profile', 'jid': 'same-profile'}),
                           row_index=1, output_dir=str(tmp_path))
    gate = threading.Barrier(2)
    calls = []
    cleanup_finished = threading.Event()
    def original(_):
        calls.append(1)
        folder = tmp_path / 'profiles/0001_same-profile'

        (folder / 'summary.json').write_text(json.dumps({'status': 'completed'}))
        time.sleep(.03)
        cleanup_finished.set()
        return 0
    monkeypatch.setattr(worker, 'single_original', original)
    def invoke(_):
        gate.wait()
        code = worker.single(args)
        return code, cleanup_finished.is_set()
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(invoke, range(2)))
    assert calls == [1]
    assert results == [(0, True), (0, True)]


def test_fixed_gap_parses_adjacent_negative_fixed_width_eigenvalues():
    from cise.validation.native.fixed_occupation import fixed_gap
    text = """bands (ev):
 -113.3326-113.3325 -2.0000 1.2500

bands (ev):
 -113.4000-113.3000 -1.5000 1.5000
"""
    result = fixed_gap(text, electrons=6)
    assert result['valid_kpoint_blocks'] == 2
    assert result['vbm_eV'] == pytest.approx(-1.5)
    assert result['cbm_eV'] == pytest.approx(1.25)
    assert result['band_gap_eV'] == pytest.approx(2.75)


def test_missing_dft_and_categorical_constraints():
    assert reporting.task_label('wbg',dict(species=['Li','O']),dict(gap_eV=None,formation_eV_atom=-2)) is None
    assert reporting.task_label('wbg',dict(species=['Li','O']),dict(gap_eV=1,formation_eV_atom=None)) is False
    assert reporting.task_label('sse',dict(species=['Si']),dict(gap_eV=4,formation_eV_atom=-2)) is False
    assert reporting.task_label('pv',dict(species=['Pb']),dict(gap_eV=1,formation_eV_atom=-2)) is False
    assert reporting.task_label('pv',dict(species=['Si']),dict(gap_eV=1,formation_eV_atom=-2)) is True
