"""Verify every archived byte before extracting; never execute archive members."""
import argparse,base64,hashlib,io,json,tarfile
from pathlib import Path
def sha(b):return hashlib.sha256(b).hexdigest()
def verify(root,extract=None):
 m=json.loads((root/'manifest.json').read_text());parts=[]
 for part in m['archive']['parts']:
  b=(root/part['path']).read_bytes();assert sha(b)==part['encoded_sha256']
  data=base64.b64decode(b);assert len(data)==part['bytes'] and sha(data)==part['sha256'];parts.append(data)
 payload=b''.join(parts);assert len(payload)==m['archive']['bytes'] and sha(payload)==m['archive']['sha256']
 expected={r['path']:r for r in m['members']};assert len(expected)==len(m['members']);seen=set()
 with tarfile.open(fileobj=io.BytesIO(payload),mode='r:gz') as tar:
  for member in tar:
   p=Path(member.name);assert member.isfile() and not p.is_absolute() and '..' not in p.parts
   assert member.name in expected and member.name not in seen
   b=tar.extractfile(member).read();spec=expected[member.name];assert len(b)==spec['bytes'] and sha(b)==spec['sha256'];seen.add(member.name)
   if extract is not None:
    target=extract/p;target.parent.mkdir(parents=True,exist_ok=True);assert not target.exists();target.write_bytes(b)
 assert seen==set(expected)
 for row in m['derived_files']:assert sha((root/row['path']).read_bytes())==row['sha256']
 return {'verified_members':len(seen),'archive_sha256':m['archive']['sha256']}
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--extract',type=Path);args=p.parse_args()
 if args.extract is not None:assert not args.extract.exists()
 print(json.dumps(verify(Path(__file__).resolve().parent,args.extract)))
