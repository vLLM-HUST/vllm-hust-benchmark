#!/usr/bin/env python3
"""Reconstruct cache accounting from live metadata and installed vLLM source.

No model is instantiated and no inference is sent. Results are source-derived
capacity bounds, cross-checked against the actual engine's reported blocks and
startup capacity; they are not hardware performance measurements.
"""
import argparse
import copy
import dataclasses
import hashlib
import importlib
import inspect
import json
import math
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from urllib.request import urlopen
from probe_prefix_cache import counters


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,required=True)
    args=parser.parse_args();plan=json.loads(args.plan.read_text())
    output=Path(plan['output']);output.mkdir(parents=True,exist_ok=False)
    with urlopen(plan['backend']+'/metrics',timeout=10) as response:
        metrics=counters(response.read().decode())
    cache=metrics['cache_config']
    assert cache['enable_prefix_caching']=='True'
    assert cache['mamba_cache_mode']=='align'
    source=Path(plan['model_config']);model=json.loads(source.read_text());text=model.get('text_config',model)
    import torch
    from vllm.v1.kv_cache_interface import FullAttentionSpec,MambaSpec
    from vllm.v1.core.kv_cache_utils import get_kv_cache_groups,get_kv_cache_config_from_groups,get_max_concurrency_for_kv_cache_config
    from vllm.v1.core.block_pool import BlockPool
    module=importlib.import_module('vllm.model_executor.models.qwen3_5')
    model_class=next(cls for cls in vars(module).values() if isinstance(cls,type) and cls.__module__==module.__name__ and hasattr(cls,'get_mamba_state_shape_from_config'))
    block=int(cache['block_size']);tp=plan['tensor_parallel_size']
    cfg=SimpleNamespace(model_config=SimpleNamespace(dtype=torch.bfloat16,hf_text_config=SimpleNamespace(**text),max_model_len=plan['max_model_len']),
        parallel_config=SimpleNamespace(tensor_parallel_size=tp,decode_context_parallel_size=1,prefill_context_parallel_size=1),
        cache_config=SimpleNamespace(mamba_cache_dtype=cache['mamba_cache_dtype'],mamba_ssm_cache_dtype=cache['mamba_ssm_cache_dtype'],mamba_cache_mode=cache['mamba_cache_mode'],num_gpu_blocks_override=None),
        scheduler_config=SimpleNamespace(disable_hybrid_kv_cache_manager=False),speculative_config=None)
    shapes=model_class.get_mamba_state_shape_from_config(cfg)
    dtypes=model_class.get_mamba_state_dtype_from_config(cfg)
    full=FullAttentionSpec(block_size=block,num_kv_heads=max(1,text['num_key_value_heads']//tp),head_size=text['head_dim'],dtype=torch.bfloat16)
    mamba=MambaSpec(block_size=block,shapes=shapes,dtypes=dtypes,page_size_padded=int(cache['mamba_page_size_padded']),mamba_cache_mode=cache['mamba_cache_mode'])
    # Mirror the installed Ascend runner's padding step (model_runner_v1.py):
    # attention pages are padded to the actual Mamba page before grouping.
    if full.page_size_bytes < mamba.page_size_bytes:
        full=dataclasses.replace(full,page_size_padded=mamba.page_size_bytes)
    types=text['layer_types'];assert len(types)==text['num_hidden_layers']
    assert set(types)<= {'full_attention','linear_attention'}
    specs={f'layer-{i}-{kind}':full if kind=='full_attention' else mamba for i,kind in enumerate(types)}
    groups=get_kv_cache_groups(cfg,specs)
    resolved=get_kv_cache_config_from_groups(cfg,groups,int(cache['kv_cache_memory_bytes']))
    expected_blocks=int(cache['num_gpu_blocks']);assert resolved.num_blocks==expected_blocks,(resolved.num_blocks,expected_blocks)
    concurrency=get_max_concurrency_for_kv_cache_config(cfg,resolved)
    startup_tokens=int(concurrency*cfg.model_config.max_model_len)
    assert startup_tokens==plan['expected_startup_tokens'],(startup_tokens,plan['expected_startup_tokens'])
    kwargs=dict(num_gpu_blocks=expected_blocks,enable_caching=True,hash_block_size=block)
    kwargs={k:v for k,v in kwargs.items() if k in inspect.signature(BlockPool).parameters}
    pool=BlockPool(**kwargs);usable=pool.get_num_free_blocks()
    full_groups=[g for g in groups if isinstance(g.kv_cache_spec,FullAttentionSpec)]
    assert len(full_groups)==1,'Only one full-attention group is supported by this accounting proposal'
    fixed_blocks=sum(g.kv_cache_spec.max_memory_usage_bytes(cfg)//g.kv_cache_spec.page_size_bytes for g in groups if isinstance(g.kv_cache_spec,MambaSpec))
    details=[]
    for length in plan['context_lengths']:
        local=copy.copy(cfg);local.model_config=copy.copy(cfg.model_config);local.model_config.max_model_len=length
        costs=[math.ceil(g.kv_cache_spec.max_memory_usage_bytes(local)/g.kv_cache_spec.page_size_bytes) for g in groups]
        required=sum(costs);details.append(dict(context_tokens=length,group_blocks=costs,blocks_per_program_upper_bound=required,whole_programs_within_usable_pool=usable//required))
    files=[inspect.getfile(fn) for fn in [get_kv_cache_groups,get_kv_cache_config_from_groups,get_max_concurrency_for_kv_cache_config,BlockPool,model_class]]
    import vllm_ascend
    files.append(str(Path(vllm_ascend.__file__).parent/'worker/model_runner_v1.py'))
    result=dict(status='PASS',evidence_type='source-derived-capacity-model',scope='No inference or model allocation; derived layout validated by engine block count and startup estimate, not direct worker-layout introspection',
        model_config_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),model_layer_types=dict(Counter(types)),
        resolved_groups=[dict(type=type(g.kv_cache_spec).__name__,layers=len(g.layer_names),block_size=g.kv_cache_spec.block_size,page_size_bytes=g.kv_cache_spec.page_size_bytes) for g in groups],
        engine_num_blocks=expected_blocks,reconstructed_num_blocks=resolved.num_blocks,usable_blocks_after_source_block_pool_initialization=usable,
        reported_startup_tokens=startup_tokens,source_startup_concurrency=concurrency,raw_block_tokens=expected_blocks*block,
        per_program_mamba_blocks_upper_bound=fixed_blocks,full_attention_rounding_reserve_tokens=block-1,
        proposed_usable_token_budget=usable*block,proposed_per_program_buffer_tokens=fixed_blocks*block+block-1,
        context_costs=details,source_sha256={str(p):hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sorted(set(files))})
    (output/'audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)

if __name__=='__main__':main()
