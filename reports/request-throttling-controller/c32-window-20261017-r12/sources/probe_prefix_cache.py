#!/usr/bin/env python3
"""Bounded real-model prefix-cache verification, before performance exploration."""
import argparse
import json
import re
import time
from pathlib import Path
from urllib.request import Request, urlopen


def call(base, path, body=None, timeout=60):
    request=Request(base+path, data=None if body is None else json.dumps(body).encode(),
                    headers={'Content-Type':'application/json'})
    with urlopen(request,timeout=timeout) as response:
        raw=response.read().decode()
        return json.loads(raw) if 'json' in response.headers.get('Content-Type','') else raw


def counters(raw):
    result={}
    for line in raw.splitlines():
        if line.startswith('#') or not line.startswith('vllm:'):continue
        name=line.split('{',1)[0].split()[0]
        if name in ('vllm:prefix_cache_hits_total','vllm:prefix_cache_queries_total',
                    'vllm:request_prefill_kv_computed_tokens_sum','vllm:num_preemptions_total'):
            result[name]=result.get(name,0)+float(line.rsplit(' ',1)[1])
        if name=='vllm:cache_config_info':result['cache_config']=dict(re.findall(r'(\w+)="([^"]*)"',line))
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--backend',required=True);p.add_argument('--model',required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    base=a.backend.rstrip('/');report={'status':'RUNNING','evidence_type':'real-model cache mechanism probe','calls':[]}
    def save():(a.output/'probe.json').write_text(json.dumps(report,indent=2)+'\n')
    save();start=time.monotonic()
    try:
        # Keep compilation/warmup separate from the repeated-prefix measurement.
        for _ in range(2):call(base,'/v1/chat/completions',{'model':a.model,'messages':[{'role':'user','content':'Say hello.'}],'max_tokens':8,'ignore_eos':True,'temperature':0,'chat_template_kwargs':{'enable_thinking':False}})
        call(base,'/reset_prefix_cache',{})
        message=[{'role':'user','content':'Cache probe unique sequence. '+(' Observation.'*4096)}]
        tokenized=call(base,'/tokenize',{'model':a.model,'messages':message,'add_generation_prompt':True,'chat_template_kwargs':{'enable_thinking':False}})
        report['prompt_tokens']=tokenized['count'];report['before']=counters(call(base,'/metrics'))
        for repeat in range(2):
            t=time.monotonic();reply=call(base,'/v1/chat/completions',{'model':a.model,'messages':message,'max_tokens':16,'ignore_eos':True,'temperature':0,'seed':11,'chat_template_kwargs':{'enable_thinking':False}})
            report['calls'].append({'repeat':repeat,'elapsed_seconds':time.monotonic()-t,'usage':reply.get('usage'),'finish_reason':reply['choices'][0]['finish_reason']});save()
            time.sleep(1)
        report['after']=counters(call(base,'/metrics'))
        report['prefix_hit_tokens_delta']=report['after'].get('vllm:prefix_cache_hits_total',0)-report['before'].get('vllm:prefix_cache_hits_total',0)
        report['status']='PASS' if report['prefix_hit_tokens_delta']>0 else 'NO_CACHE_REUSE'
    except Exception as error:
        report.update(status='FAIL',error=str(error));raise
    finally:
        report['total_seconds_including_warmup']=time.monotonic()-start;save();print(json.dumps(report),flush=True)

if __name__=='__main__':main()
