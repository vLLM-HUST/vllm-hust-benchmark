import hashlib
import importlib.util
import json
from pathlib import Path
import pytest

spec=importlib.util.spec_from_file_location('offline_pipeline',Path(__file__).parents[1]/'scripts/run_offline_throttle_pipeline.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def config(tmp_path,arrival='0',cap=0):
    workload=tmp_path/'workload';workload.mkdir();csv='session_id,arrival_time\na,'+arrival+'\n';(workload/'sessions.csv').write_text(csv);(workload/'manifest.json').write_text('{}')
    ride=tmp_path/'ride/scripts';ride.mkdir(parents=True);(ride/'deployment_payload.py').write_text('# test payload helper')
    deployment=tmp_path/'deployment.json';deployment.write_text(json.dumps({'run_id':'old','remote_root':'/workspace/home/old','operator_root':'/tmp/old'}))
    template=tmp_path/'window.json';template.write_text(json.dumps(dict(arrival_mode='offline-burst',concurrency=cap,client_concurrency_limit=None,expected_sessions_sha256=hashlib.sha256(csv.encode()).hexdigest(),programs=1,points=[dict(scheduling_enabled=True),dict(scheduling_enabled=False)])))
    return dict(deployment_plan=deployment,window_template=template,workload=workload,output=tmp_path/'results',ride_root=ride.parent)

def test_prepare_rebinds_all_run_paths_and_freezes_sources(tmp_path):
    c=config(tmp_path);d=m.prepare(c,dict(run_id='batch-new',seed=17,order='OFF_ON'))
    plan=m.read(d/'plan.json');window=m.read(d/'window-plan.json')
    assert plan['remote_root']=='/workspace/home/batch-new'
    assert plan['operator_root']=='/tmp/batch-new-operator'
    assert window['output']==plan['remote_root']+'/window' and window['seed']==17
    assert [p['scheduling_enabled'] for p in window['points']]==[False,True]
    assert (d/'payload/experiments/workload/arrival_process.py').exists()
    with pytest.raises(ValueError,match='Run exists'):m.prepare(c,dict(run_id='batch-new',seed=17,order='OFF_ON'))

@pytest.mark.parametrize('arrival,cap',[('1',0),('0',32)])
def test_bad_offline_contract_rejected_before_directory_creation(tmp_path,arrival,cap):
    c=config(tmp_path,arrival,cap)
    with pytest.raises(ValueError):m.prepare(c,dict(run_id='bad',seed=11,order='ON_OFF'))
    assert not (c['output']/'bad').exists()

def test_launch_rejects_modified_frozen_source_before_ssh(tmp_path,monkeypatch):
    c=config(tmp_path);d=m.prepare(c,dict(run_id='batch-new',seed=17,order='OFF_ON'))
    (d/'payload/experiments/workload/replay_sessions.py').write_text('changed')
    monkeypatch.setattr(m,'command',lambda *a,**k:pytest.fail('No remote action expected'))
    with pytest.raises(ValueError,match='Frozen payload changed'):m.launch('unused',d)

def test_resume_spec_cannot_relabel_an_existing_seed(tmp_path):
    c=config(tmp_path);c['runs']=[dict(run_id='batch-new',seed=17,order='OFF_ON')];c['ssh_host']='unused'
    m.prepare(c,c['runs'][0]);c['runs'][0]['seed']=23
    with pytest.raises(ValueError,match='Resume spec differs'):m.run(c,'finish')

def test_launch_uses_ride_payload_and_submission_intent(tmp_path,monkeypatch):
    c=config(tmp_path);d=m.prepare(c,dict(run_id='batch-new',seed=17,order='OFF_ON'))
    calls=[]
    monkeypatch.setattr(m,'command',lambda argv,**kwargs:calls.append(argv))
    def probe(host,code):
        compile(code,'generated-remote','exec')
        assert 'from deployment_payload import prepare_payload' in code
        assert 'flock' in code and 'Another window is active' in code
        assert 'batch-new--offline-window-v1' in code
        return {'job_id':'batch-new--offline-window-v1'}
    monkeypatch.setattr(m,'remote',probe)
    assert m.launch('operator',d)['job_id']=='batch-new--offline-window-v1'
    assert len(calls)==2 and m.read(d/'submission.json')['run_id']=='batch-new'
    with pytest.raises(ValueError,match='Submission intent'):m.launch('operator',d)
