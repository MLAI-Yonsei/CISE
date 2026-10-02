import json
import os
import subprocess
import sys
import pytest

def cli(*args):
    env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    env.pop('OPENROUTER_API_KEY',None);env.pop('CISE_CONFIG',None)
    return subprocess.run([sys.executable,'-m','cise.cli',*map(str,args)],text=True,capture_output=True,env=env,timeout=90)

@pytest.mark.parametrize('method,expected',[('llema',4),('cci',4)])
def test_offline_cli_complete_resume_and_no_fake_dft(tmp_path,method,expected):
    cfg=tmp_path/'config.json';out=tmp_path/'output'
    cfg.write_text(json.dumps(dict(method=method,output_dir='output',tasks=['wbg','sse','pv','highk'],iterations=2,demo=True,parallel_tasks=2)))
    for command in ['doctor','init','run','run','report']:
        p=cli(command,'--config',cfg);assert p.returncode==0,p.stdout+p.stderr
    metrics=json.loads((out/'results/main_results.json').read_text())[0]
    assert metrics['generated_candidates']==expected
    assert metrics['completed_iterations']==2 and metrics['DFT_TP']==metrics['DFT_FP']==0
    assert metrics['DFT_status']=='NOT_RUN'
    assert metrics['DFT_unresolved']==metrics['outputs']
    assert 'DEMO' in (out/'REPORT.md').read_text()
    assert metrics['api_calls']==(2 if method=='llema' else 10)
    assert not (out/'budget.json').exists()
    assert not (out/'model_preflight.json').exists()
    d=json.loads(cfg.read_text());d['iterations']=3;cfg.write_text(json.dumps(d))
    p=cli('run','--config',cfg);assert p.returncode!=0 and 'CONFIG_CHANGED' in p.stderr

def test_wheel_only_configuration_command(tmp_path):
    path=tmp_path/'portable.json'
    p=cli('configure','--method','cci','--demo','--output',path)
    assert p.returncode==0,p.stderr
    cfg=json.loads(path.read_text());assert cfg['demo'] and cfg['iterations']==2 and cfg['method']=='cci'
    assert cli('configure','--method','cci','--demo','--output',path).returncode!=0


def test_unknown_config_rejected(tmp_path):
    from cise.config import load_config
    p=tmp_path/'bad.json';p.write_text(json.dumps(dict(api_key='not-a-real-key')))
    with pytest.raises(ValueError,match='Unknown'):load_config(p)

def test_asset_drift_blocks_before_run(tmp_path):
    cfg=tmp_path/'config.json';cfg.write_text(json.dumps(dict(method='cci',output_dir='out',demo=True,tasks=['wbg'],iterations=1)))
    p=cli('init','--config',cfg);assert p.returncode==0,p.stderr
    asset=tmp_path/'out/interval/wbg.json';asset.write_text('{}')
    p=cli('run','--config',cfg)
    assert p.returncode!=0 and 'ASSET_CHANGED' in p.stderr


def test_quantile_offset_demo_runs_actual_gibbs_path(tmp_path):
    cfg=tmp_path/'qr.json';out=tmp_path/'qr_out'
    cfg.write_text(json.dumps(dict(method='cci',output_dir='qr_out',tasks=['wbg'],iterations=1,demo=True,
                                 cci=dict(quantile_offset=True))))
    for command in ['init','run']:
        p=cli(command,'--config',cfg);assert p.returncode==0,p.stdout+p.stderr
    row=json.loads((out/'runs/B/wbg/iterations/0001.json').read_text())['candidates'][0]
    assert row['cci_solver']['band_gap']['test_point_correction']
    assert row['cci_solver']['band_gap']['base_quantile']==0
    assert row['cci_solver']['band_gap']['score_definition']=='normalized_absolute_residual_minus_frozen_quantile'
    assert row['cci_solver']['band_gap']['quantile_model_sha256']
