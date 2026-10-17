import pathlib,json,hashlib
r=pathlib.Path(__file__).parent
for line in (r/'SHA256SUMS').read_text().splitlines():
 h,n=line.split('  ',1);assert pathlib.Path(n).name==n;assert hashlib.sha256((r/n).read_bytes()).hexdigest()==h
m=json.loads((r/'single-matrix-merge-receipt.json').read_text());o=json.loads((r/'single-matrix-online-receipt.json').read_text());c=json.loads((r/'single-matrix-live-check.json').read_text());assert m['pr']==o['pr']==443 and m['head']==o['head'];assert o['merge']==m['merged_sha']==o['pages']['commit'];assert o['pages']['status']=='built' and o['status']=='PUBLISHED_SINGLE_MATRIX_VERIFIED';assert c=={'tables':1,'rows':6,'separateSections':0,'controller':True};assert len(m['checks'])==4 and all(x['conclusion']=='success' for x in m['checks']);print(json.dumps({'single_matrix_publication':'PASS','merge':o['merge']}))
