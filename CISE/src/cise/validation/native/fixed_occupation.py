"""Fixed-occupation band gaps and electronic dielectric validation."""

import json
import math
import os
from pathlib import Path
import re

import numpy as np
from . import qe_runner as base

original_run=base.run_qe

def fixed_gap(text,electrons):
    occupied=int(round(electrons/2))
    blocks=[]; current=None
    for line in text.splitlines():
        if 'bands (ev)' in line.lower():
            current=[]; blocks.append(current); continue
        if current is None: continue
        stripped=line.strip()
        if stripped and re.fullmatch(r'[\s\d.+EeDd-]+',stripped):


            numbers=re.findall(r'[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[EeDd][-+]?\d+)?',stripped)
            current.extend(float(x.replace('D','E').replace('d','e')) for x in numbers)
        elif current:
            current=None
    good=[x for x in blocks if len(x)>occupied]
    if not good or len(good)!=len(blocks): return {'band_gap_eV':None,'method':'fixed_eigenvalue_parse_failed'}
    vbm=max(x[occupied-1] for x in good); cbm=min(x[occupied] for x in good)
    return {'band_gap_eV':max(0.,cbm-vbm),'vbm_eV':vbm,'cbm_eV':cbm,
            'method':'fixed_occupation_electron_count_edges','valid_kpoint_blocks':len(good)}

def fixed_input(text):
    lines=[]
    for line in text.splitlines():
        if re.match(r'\s*(smearing|degauss)\s*=',line): continue
        if re.match(r'\s*occupations\s*=',line): line="  occupations = 'fixed',"
        lines.append(line)
    return '\n'.join(lines)+'\n'

def tensor_from_output(text):
    marker='Dielectric constant in cartesian axis'
    if marker not in text: return None
    after=text.rsplit(marker,1)[1]
    rows=[]
    for line in after.splitlines()[1:12]:
        numbers=re.findall(r'[-+]?\d+\.\d+(?:[EeDd][-+]?\d+)?',line)
        if len(numbers)==3:
            rows.append([float(x.replace('D','E').replace('d','e')) for x in numbers])
        if len(rows)==3: break
    if len(rows)!=3 or not np.isfinite(rows).all(): return None
    tensor=np.array(rows)
    if not np.allclose(tensor,tensor.T,atol=.05): return None
    return {'tensor':rows,'xx':float(tensor[0,0]),'trace_mean':float(np.trace(tensor)/3),
            'eigenvalues':np.linalg.eigvalsh((tensor+tensor.T)/2).tolist(),
            'quantity':'clamped-ion electronic macroscopic dielectric tensor; ionic contribution not included'}

def run(args,row,run_dir):
    record=original_run(args,row,run_dir)
    if args.skip_bands or not record.get('scf_converged'): return record
    record['smearing_band_gap']=record.get('band_gap')
    params=record['parameters']
    electrons=float(params['valence_electrons'])
    eligible=not params['magnetism']['spin_polarized'] and abs(electrons/2-round(electrons/2))<1e-8
    record['gap_validation_status']='UNRESOLVED_SPIN_OR_ODD_ELECTRONS' if not eligible else 'PENDING_FIXED_OCCUPATION_NSCF'
    env={**os.environ,'OMP_NUM_THREADS':str(args.omp_threads),'CUDA_VISIBLE_DEVICES':''}
    if eligible:
        nscf=run_dir/'nscf.in'
        if not nscf.exists():
            record['gap_validation_status']='MISSING_NSCF_INPUT'
            return record
        (run_dir/'nscf_fixed.in').write_text(fixed_input(nscf.read_text()))
        outcome=base.run_to_file([record['pw_x'],'-in','nscf_fixed.in'],run_dir,run_dir/'nscf_fixed.out',args.timeout,env)
        text=(run_dir/'nscf_fixed.out').read_text(errors='replace')
        gap=fixed_gap(text,electrons)
        record['fixed_nscf_run']=outcome
        record['fixed_occupation_band_gap']=gap
        if outcome['returncode']==0 and 'JOB DONE.' in text and gap.get('band_gap_eV') is not None:
            record['gap_validation_status']='FIXED_OCCUPATION_NSCF_COMPLETED'
            record['band_gap']=gap

            if record.get('energy_eV_atom') is not None: record['status']='completed'
        else:
            record['gap_validation_status']='FIXED_OCCUPATION_NSCF_FAILED_OR_UNRESOLVED'
    if row.get('task_id')!='highk': return record
    gap=record.get('band_gap') or {}
    if record['gap_validation_status']!='FIXED_OCCUPATION_NSCF_COMPLETED':
        record['dielectric_status']='NOT_RUN_GAP_UNRESOLVED'; return record
    if not 2.5<=float(gap.get('band_gap_eV',-1))<=6.5:
        record['dielectric_status']='NOT_NEEDED_AND_GAP_CONDITION_FAILED'; return record

    (run_dir/'dielectric_scf.in').write_text(fixed_input((run_dir/'scf.in').read_text()))
    scf=base.run_to_file([record['pw_x'],'-in','dielectric_scf.in'],run_dir,run_dir/'dielectric_scf.out',args.timeout,env)
    scftext=(run_dir/'dielectric_scf.out').read_text(errors='replace')
    record['dielectric_scf_run']=scf
    if scf['returncode']!=0 or not base.parse_scf_converged(scftext):
        record['dielectric_status']='FIXED_SCF_FAILED'; return record
    prefix=base.safe_label(record['label'])
    outdir=run_dir/'qe_tmp'
    ph=Path(os.environ['CISE_PH']) if os.environ.get('CISE_PH') and os.environ['CISE_PH']!='ph.x' else Path(record['pw_x']).with_name('ph.x')
    inp=("Electronic dielectric validation\n&inputph\n"
         f" prefix='{prefix}',\n outdir='{outdir}',\n"
         " epsil=.true.,\n trans=.false.,\n ldisp=.false.,\n tr2_ph=1.0d-14,\n fildyn='dielectric.dyn',\n/\n0.0 0.0 0.0\n")
    (run_dir/'dielectric_ph.in').write_text(inp)
    outcome=base.run_to_file([str(ph),'-in','dielectric_ph.in'],run_dir,run_dir/'dielectric_ph.out',args.timeout,env)
    text=(run_dir/'dielectric_ph.out').read_text(errors='replace')
    tensor=tensor_from_output(text)
    record['dielectric_ph_run']=outcome
    record['dielectric']=tensor
    record['dielectric_status']='ELECTRONIC_DFPT_COMPLETED' if outcome['returncode']==0 and 'JOB DONE.' in text and tensor else 'DFPT_FAILED_OR_UNPARSED'
    record['highk_full_label']='UNRESOLVED_DIELECTRIC_SCALAR_AND_IONIC_CONVENTION'
    return record
