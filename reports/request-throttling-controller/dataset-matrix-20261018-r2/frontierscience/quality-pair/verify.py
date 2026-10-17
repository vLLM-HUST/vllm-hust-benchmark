import hashlib,json,pathlib,subprocess,sys
root=pathlib.Path(__file__).parent
for line in (root/'SHA256SUMS').read_text().splitlines():
 h,n=line.split('  ',1);assert pathlib.Path(n).name==n;assert hashlib.sha256((root/n).read_bytes()).hexdigest()==h
r=json.loads((root/'report.json').read_text());assert r['status']=='QUALITY_PAIR_COMPLETE_PERFORMANCE_PENDING' and r['performance'] is None
for arm in ['native','controller']:
 p=root.parent/(arm+'-quality-checkpoint');subprocess.run([sys.executable,str(p/'verify.py')],check=True,capture_output=True);c=json.loads((p/'checkpoint.json').read_text());assert c['summary']==r['arms'][arm]['summary'];assert c['tasks_sha256']==r['tasks_sha256'];assert c['client_code_sha256']==r['client_code_sha256']
for t in ['olympiad','research']:
 n=r['arms']['native']['summary']['tracks'][t]['success_rate'];c=r['arms']['controller']['summary']['tracks'][t]['success_rate'];assert r['comparison'][t]=={'native':n,'controller':c,'difference_percentage_points':100*(c-n),'retained_native_score_fraction':c/n}
assert all(r['pair_input_check'].values())
print(json.dumps({'quality_pair':'PASS','performance':'PENDING','private_input_comparison':'receipt only; public verifier checks source/task hashes and report consistency'}))
