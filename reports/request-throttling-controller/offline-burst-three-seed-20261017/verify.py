import hashlib,json,pathlib,subprocess,sys
from summarize_throttle_windows import summarize
root=pathlib.Path(__file__).resolve().parent
for line in (root/'SHA256SUMS').read_text().splitlines():
 digest,name=line.split('  ',1);assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest,name
paths=[root.parent/('offline-burst-20261017-r'+str(n)) for n in [15,17,18]];repeat=[root.parent/'offline-burst-20261017-r16']
for path in paths+repeat:subprocess.run([sys.executable,str(path/'verify.py')],check=True)
actual=summarize(paths,repeat);expected=json.loads((root/'report.json').read_text());assert actual=={k:v for k,v in expected.items() if k not in ['runtime_profile','runtime_profile_sha256']}
profile=root.parent/'offline-burst-protocol-20261017/runtime-profile.json';assert hashlib.sha256(profile.read_bytes()).hexdigest()==expected['runtime_profile_sha256'];assert json.loads(profile.read_text())==expected['runtime_profile']
print(json.dumps(dict(integrity='PASS',aggregate=actual['aggregate'],finite_makespan_claim=False)))
