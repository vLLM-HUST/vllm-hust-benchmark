"""Run the requested checks in an isolated directory; preserve each command and exit."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
BASE = Path('/root/diffspec-experiments')
PYTHON = str(BASE / 'environments/diffspec-moe-venv/bin/python')
CORE = BASE / 'source-snapshots/diffspec-runtime-core-kvgroup-20261029'
ASCEND = BASE / 'source-snapshots/diffspec-runtime-ascend-moe-20261030'
OPP = '/vllm-workspace/vllm-ascend/vllm_ascend/_cann_ops_custom/vendors/custom_transformer'
ENV = os.environ.copy()
ENV.update(
    PYTHONPATH=':'.join(['/usr/local/Ascend/cann-9.1.0/python/site-packages',
                         str(CORE), str(ASCEND), str(ROOT / 'source'), str(ROOT)]),
    VLLM_PLUGINS='ascend,diffspec', ASCEND_RT_VISIBLE_DEVICES='0,1,2,3',
    OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', TOKENIZERS_PARALLELISM='false',
    PYTHONDONTWRITEBYTECODE='1', VLLM_WORKER_MULTIPROC_METHOD='spawn',
    VLLM_CACHE_ROOT=str(ROOT / 'cache'), ASCEND_CUSTOM_OPP_PATH=OPP,
    LD_PRELOAD=OPP + '/op_api/lib/libcust_opapi.so',
)

stage = sys.argv[1]
if stage == 'preflight':
    code = '''import json,inspect,torch,torch_npu,vllm,vllm_ascend,diffspec
from diffspec.runtime import DiffSpecDraftCache
data={"torch":torch.__version__,"torch_npu":torch_npu.__version__,
"vllm":vllm.__file__,"ascend":vllm_ascend.__file__,"diffspec":diffspec.__file__,
"runtime":inspect.getsourcefile(DiffSpecDraftCache),"devices":[]}
for i in range(torch.npu.device_count()):
 p=torch.npu.get_device_properties(i)
 data["devices"].append({"runtime_index":i,"name":p.name,"uuid":str(p.uuid)})
print(json.dumps(data,indent=2))
'''
    command = [PYTHON, '-c', code]
elif stage == 'cpu':
    command = [PYTHON, '-m', 'pytest', str(ROOT / 'source/tests/test_runtime.py'),
               '-q', '-k', 'linear_proposer_retrieves_only_committed_queries or '
               'linear_compact_root_excludes_rejected_target_suffix or '
               'tp_retrieval_selects_same_context_from_all_target_heads']
elif stage == 'model':
    command = [PYTHON, str(ROOT / 'model_smoke.py')]
else:
    raise ValueError(stage)

with (ROOT / f'{stage}-command.json').open('w') as file:
    json.dump({'command': command, 'environment': {key: ENV[key] for key in
               ['PYTHONPATH', 'VLLM_PLUGINS', 'ASCEND_RT_VISIBLE_DEVICES',
                'VLLM_CACHE_ROOT', 'ASCEND_CUSTOM_OPP_PATH', 'LD_PRELOAD']}}, file, indent=2)
started = time.time()
with (ROOT / f'{stage}.log').open('w') as log:
    process = subprocess.Popen(command, cwd=ROOT, env=ENV, stdout=log,
                               stderr=subprocess.STDOUT)
    (ROOT / f'{stage}-process.json').write_text(json.dumps({'pid': process.pid}))
    result = process.wait()
(ROOT / f'{stage}-exit.json').write_text(json.dumps({'exit_code': result,
    'started_epoch': started, 'finished_epoch': time.time()}, indent=2))
print(f'{stage}: exit={result}', flush=True)
raise SystemExit(result)
