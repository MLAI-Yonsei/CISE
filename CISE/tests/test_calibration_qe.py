import json
from unittest.mock import patch
import pytest
from pymatgen.core import Lattice,Structure
from cise.calibration import fit_manifest,FLOORS
from cise.shared.structure_features import STRUCTURE_FEATURES
from cise.validation.gap_parser import parse_gap

@pytest.mark.parametrize('mode',['absolute','single','endpoints'])
def test_import_preserves_declared_quantile_models(tmp_path,mode):
    import hashlib
    from cise.assets import import_calibration
    source=tmp_path/'source';folder=source/'interval';folder.mkdir(parents=True)
    for prop in FLOORS:
        (folder/f'{prop}_scale.joblib').write_bytes(b'scale')
        metadata={}
        if mode=='single':
            name=f'{prop}_quantile.joblib';(folder/name).write_bytes(b'quantile')
            metadata=dict(quantile_model=name,quantile_model_sha256=hashlib.sha256(b'quantile').hexdigest())
        if mode=='endpoints':
            names={side:f'{prop}_{side}.joblib' for side in ['lower','upper']}
            for name in names.values():(folder/name).write_bytes(b'quantile')
            metadata=dict(quantile_models=names,quantile_models_sha256={side:hashlib.sha256(b'quantile').hexdigest() for side in names})
        (folder/f'{prop}_residuals.json').write_text(json.dumps(metadata))
    for task in ['wbg','sse','pv','highk']:(folder/f'{task}.json').write_text('{}')
    (folder/'training_manifest.json').write_text('{}')
    (folder/'fixed_basis.json').write_text('{}')
    (folder/'band_gap_fixed_basis.json').write_text('{}')
    destination=tmp_path/'imported';import_calibration(source,destination)
    for path in folder.iterdir():assert (destination/path.name).read_bytes()==path.read_bytes()

    assert not (destination/'data_exclusion_locked.json').exists()
    if mode!='absolute':
        meta=json.loads((folder/'band_gap_residuals.json').read_text())
        if mode=='single':meta['quantile_model']='../escape.joblib'
        else:meta['quantile_models']['lower']='../escape.joblib'
        (folder/'band_gap_residuals.json').write_text(json.dumps(meta))
        with pytest.raises(ValueError,match='local filename'):import_calibration(source,tmp_path/'bad')
        assert not (tmp_path/'bad').exists()

@pytest.mark.parametrize('field,value,message',[
    ('id',123,'nonempty string id'),
    ('group',[],'nonempty string group'),
    ('features',[], 'Missing structure features'),
    ('features',{k:float('inf') for k in STRUCTURE_FEATURES},'numeric or missing'),
    ('features',{k:'1.0' for k in STRUCTURE_FEATURES},'numeric or missing'),
])
def test_calibration_rejects_invalid_inputs_before_writing(tmp_path,field,value,message):
    row=dict(id='sample',property='band_gap',group='sample',split='fit',prediction=2.,truth=2.1,
             features={k:0. for k in STRUCTURE_FEATURES})
    row[field]=value
    source=tmp_path/'manifest.jsonl';source.write_text(json.dumps(row)+'\n')
    with pytest.raises(ValueError,match=message):fit_manifest(source,tmp_path/'cal')
    assert not (tmp_path/'cal').exists()

def test_calibration_rejects_repeated_cif_across_splits_with_features(tmp_path):
    cif=tmp_path/'same.cif';cif.write_text('same structure bytes')
    rows=[dict(id=split,property='band_gap',group=split,split=split,cif_path=cif.name,
               prediction=2.,truth=2.1,features={k:0. for k in STRUCTURE_FEATURES})
          for split in ['fit','calibration']]
    source=tmp_path/'manifest.jsonl';source.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    with pytest.raises(ValueError,match='Repeated CIF across splits'):fit_manifest(source,tmp_path/'cal')
    assert not (tmp_path/'cal').exists()

def test_calibration_checks_cif_hash_even_with_supplied_features(tmp_path):
    cif=tmp_path/'same.cif';cif.write_text('same structure bytes')
    row=dict(id='sample',property='band_gap',group='sample',split='fit',cif_path=cif.name,
             cif_sha256='incorrect',prediction=2.,truth=2.1,features={k:0. for k in STRUCTURE_FEATURES})
    source=tmp_path/'manifest.jsonl';source.write_text(json.dumps(row)+'\n')
    with pytest.raises(ValueError,match='CIF hash mismatch'):fit_manifest(source,tmp_path/'cal')
    assert not (tmp_path/'cal').exists()

def test_calibration_roles_and_group_leakage(tmp_path):
    rows=[]
    for prop in FLOORS:
        for split,n in [('fit',32),('dre_source',32),('calibration',20)]:
            for i in range(n):
                rows.append(dict(id=f'{split}{i}',property=prop,group=f'{split}{i}',split=split,
                    prediction=2.,truth=2.1,features={k:float(i) for k in STRUCTURE_FEATURES}))
    source=tmp_path/'manifest.jsonl';source.write_text(''.join(json.dumps(x)+'\n' for x in rows))
    fit_manifest(source,tmp_path/'cal')
    data=json.loads((tmp_path/'cal/band_gap_residuals.json').read_text())
    assert len(data['rows'])==20 and len(data['dre_source'])==32
    assert abs(data['rows'][0]['score']-1)<1e-5
    rows[32]['group']=rows[0]['group'];source.write_text(''.join(json.dumps(x)+'\n' for x in rows))
    with pytest.raises(ValueError,match='Composition leakage'):fit_manifest(source,tmp_path/'bad')
    assert not (tmp_path/'bad').exists()

def test_strict_gap_requires_converged_all_kpoints(tmp_path):
    p=tmp_path/'nscf.out'
    text='number of k points= 2\n bands (ev):\n -3.0 -1.0 2.0 4.0\n\n bands (ev):\n -2.0 -0.5 1.5 4.0\n\nJOB DONE.\n'
    p.write_text(text);assert parse_gap(p,4,1,4)['band_gap_eV']==2.
    p.write_text(text+'eigenvalues not converged\n')
    with pytest.raises(AssertionError):parse_gap(p,4,1,4)
    p.write_text(text)
    with pytest.raises(AssertionError,match='nonintegral'):parse_gap(p,3,1,4)

def test_qetb_fixed_geometry_input(tmp_path):
    from cise.validation import worker
    struct=Structure(Lattice.cubic(4.2),['Mg','O'],[[0,0,0],[.5,.5,.5]])
    pseudo={x:str(tmp_path/(x.lower()+'.pbesol.UPF')) for x in ['Mg','O']}
    with patch.object(worker,'original_parameters',return_value={'valence_electrons':8.}):
        params=worker.parameters(struct,{},pseudo,20,30)
    assert params['xc']=='PBEsol' and params['ecutwfc_Ry']==45 and params['ecutrho_Ry']==250
    assert all(2<=v<=14 and v%2==0 for v in params['scf_kgrid'])
    text=worker.qe_input(struct,pseudo,'fixture',tmp_path/'scratch','scf',params['scf_kgrid'],45,250,12,0,params['magnetism'])
    assert "input_dft = 'PBEsol'" in text and "smearing = 'gaussian'" in text
    assert 'ecfixed = 44.5' in text and 'degauss = 0.01' in text
    assert 'nspin = 2' not in text
    with patch.object(worker,'run_original',return_value={}):
        result=worker.run(None,{},tmp_path)
    assert 'PBEsol' in result['interpretation']
