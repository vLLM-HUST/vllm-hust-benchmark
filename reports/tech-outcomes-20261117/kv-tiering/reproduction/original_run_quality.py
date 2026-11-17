import argparse,asyncio,json,time,random,hashlib,traceback,math
from pathlib import Path
import aiohttp

def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def values(text):
 out={}
 for l in text.splitlines():
  if l and not l.startswith('#'):
   key,val,*_=l.split();out[key]=float(val)
 return out
async def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--label',required=True);ap.add_argument('--plan',required=True);args=ap.parse_args();root=Path(args.root);out=root/(args.label+'-client');out.mkdir(exist_ok=False);(out/'requests').mkdir();plan=json.loads(Path(args.plan).read_text());dump(out/'plan.json',plan)
 records=[]
 async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=1800,sock_connect=10)) as session:
  async def metrics(name):
   async with session.get('http://127.0.0.1:18236/metrics') as r:r.raise_for_status();text=await r.text()
   (out/name).write_text(text);return values(text)
  for phase in plan['phases']:
   name=phase['name'];before=await metrics(name+'-before.prom');start=time.monotonic();sem=asyncio.Semaphore(phase.get('concurrency',1));phase_records=[]
   async def request(index,item):
    async with sem:
     rng=random.Random(plan['seed']+item['prefix_id']);prefix=[rng.randrange(1000,20000) for _ in range(plan['prefix_len'])];srng=random.Random(plan['seed']+100000+item.get('suffix_id',item['prefix_id']));prompt=prefix+[srng.randrange(20001,24000) for _ in range(plan['suffix_len'])]
     if 'prompt_file' in item:prompt=json.loads(Path(item['prompt_file']).read_text())
     data={'model':plan.get('model','qwen3.5-35b-a3b'),'prompt':prompt,'max_tokens':phase.get('output_len',plan['output_len']),'temperature':0,'ignore_eos':True,'stream':True,'stream_options':{'include_usage':True}}
     key=f'{name}-{index:04d}';dump(out/'requests'/(key+'-input.json'),data);t=time.monotonic();first=None;raw=bytearray();usage=None;finish=None;done=0;text='';chunks=[]
     async with session.post('http://127.0.0.1:18236/v1/completions',json=data) as r:
      status=r.status
      async for line in r.content:
       raw.extend(line)
       if not line.startswith(b'data:'):continue
       body=line[5:].strip()
       if body==b'[DONE]':done+=1;continue
       event=json.loads(body)
       if event.get('error'):raise RuntimeError(str(event['error']))
       if event.get('usage'):usage=event['usage']
       for c in event.get('choices',[]):
        if c.get('text'):
         now=time.monotonic();chunks.append(now-t)
         if first is None:first=now
         text+=c['text']
        finish=c.get('finish_reason') or finish
     elapsed=time.monotonic()-t;(out/'requests'/(key+'.sse')).write_bytes(raw)
     valid=status==200 and done==1 and finish=='length' and usage is not None and usage['completion_tokens']==data['max_tokens'] and usage['total_tokens']==usage['prompt_tokens']+usage['completion_tokens']
     rec={'key':key,'phase':name,**item,'http_status':status,'ttft_s':None if first is None else first-t,'wall_s':elapsed,'tpot_s':None if first is None else (elapsed-(first-t))/max(1,usage['completion_tokens']-1),'usage':usage,'finish_reason':finish,'done':done,'valid':valid,'input_sha256':hashlib.sha256(json.dumps(prompt,separators=(',',':')).encode()).hexdigest(),'output_sha256':hashlib.sha256(text.encode()).hexdigest(),'output_text':text,'chunk_times_s':chunks}
     if 'expected' in item:rec['answer_correct']=item['expected'].casefold() in text.casefold()
     dump(out/'requests'/(key+'-result.json'),rec)
     with (out/'requests.jsonl').open('a') as f:f.write(json.dumps(rec,ensure_ascii=False)+'\n')
     phase_records.append(rec);records.append(rec);print(json.dumps({k:v for k,v in rec.items() if k not in ['output_text','chunk_times_s']}),flush=True)
     if not valid:raise RuntimeError('invalid completion '+key)
   await asyncio.gather(*(request(i,item) for i,item in enumerate(phase['requests'])))
   wall=time.monotonic()-start
   await asyncio.sleep(1)
   after=await metrics(name+'-after.prom')
   delta={k:after[k]-before.get(k,0) for k in after if k.endswith('}') and any(t in k.split('{')[0] for t in ['_total','_sum','_count'])}
   summary={'name':name,'wall_s':wall,'completed':len(phase_records),'output_tokens':sum(x['usage']['completion_tokens'] for x in phase_records),'output_tok_s':sum(x['usage']['completion_tokens'] for x in phase_records)/wall,'metrics_delta':delta,'metrics_after':{k:v for k,v in after.items() if any(t in k for t in ['kv_tiering','kv_offload','prefix_cache','preempt','kv_cache_usage'])}}
   dump(out/(name+'-summary.json'),summary);print('PHASE',name,json.dumps(summary),flush=True)
 dump(out/'complete.json',{'completed':True,'requests':len(records)})
try:asyncio.run(main())
except BaseException:
 traceback.print_exc();raise
