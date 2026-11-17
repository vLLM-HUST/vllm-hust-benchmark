"""Read-only source/model/runtime identities; do not initialize an NPU."""
import datetime
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path('/root/CGCL/kv-tiering-performance-recheck-20261116')
PREVIOUS = Path('/root/CGCL/kv-tiering-working-set-20261116')
PLUGIN = PREVIOUS / 'revision-v2/plugin'


def main():
    output = ROOT / (sys.argv[1] + '-identity.json')
    if output.exists():
        raise FileExistsError(output)
    config = json.loads((ROOT / 'swe32-measurements-v1/r1-native.argv.json').read_text())
    sys.path[:0] = config['env']['PYTHONPATH'].split(':')
    expected = json.loads((PREVIOUS / 'revision-v2/local-source-manifest.json').read_text())
    plugin_hashes = {name: hashlib.sha256((PLUGIN / name).read_bytes()).hexdigest()
        for name in expected['sha256']}
    assert plugin_hashes == expected['sha256']
    old = json.loads((PREVIOUS / 'runtime-git-identity.json').read_text())
    repos = {}
    env = dict(os.environ, GIT_OPTIONAL_LOCKS='0')
    for label in ('core', 'ascend'):
        prior = old[label]
        head = subprocess.check_output(['git', '-C', prior['root'], 'rev-parse', 'HEAD'], text=True, env=env).strip()
        status = subprocess.check_output(['git', '-C', prior['root'], 'status', '--short'], text=True, env=env)
        assert head == prior['commit'] and status == prior['status']
        repos[label] = {'root': prior['root'], 'head': head, 'status': status}
    repos['plugin'] = {'root': str(PLUGIN),
        'head': subprocess.check_output(['git', '-C', str(PLUGIN), 'rev-parse', 'HEAD'], text=True, env=env).strip(),
        'status': subprocess.check_output(['git', '-C', str(PLUGIN), 'status', '--short'], text=True, env=env),
        'staged_diff': subprocess.check_output(['git', '-C', str(PLUGIN), 'diff', '--cached', '--stat'], text=True, env=env)}
    assert not repos['plugin']['staged_diff']
    model = Path('/models/Qwen3.5-35B-A3B')
    files = {}
    for name in ('config.json', 'generation_config.json', 'tokenizer.json',
        'tokenizer_config.json', 'chat_template.jinja', 'model.safetensors.index.json'):
        path = model / name
        if path.exists():
            files[name] = {'bytes': path.stat().st_size, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    weights = {p.name: {'bytes': p.stat().st_size, 'mtime_ns': p.stat().st_mtime_ns}
        for p in sorted(model.glob('*.safetensors'))}
    result = {'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'python': sys.version, 'executable': sys.executable,
        'imports': {name: importlib.util.find_spec(name).origin for name in
            ('vllm_hust_kv_tiering', 'vllm', 'vllm_ascend', 'torch', 'torch_npu', 'acl')},
        'distributions': {name: importlib.metadata.version(name) for name in
            ('vllm', 'vllm-ascend', 'torch', 'torch-npu', 'transformers', 'tokenizers', 'numpy')},
        'repositories': repos, 'plugin_sha256': plugin_hashes, 'model_files': files,
        'weight_stat_only_not_full_byte_hash': weights,
        'client_sha256': hashlib.sha256(Path('/root/CGCL/kv-tiering-recovery-validation-20261112/run_quality.py').read_bytes()).hexdigest(),
        'supervisor_sha256': hashlib.sha256(Path('/root/CGCL/kv-tiering-recovery-validation-20261112/serve_supervisor.py').read_bytes()).hexdigest()}
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'output': str(output), 'plugin_files_verified': len(plugin_hashes),
        'core_ascend_unchanged': True, 'weight_shards': len(weights)}))


if __name__ == '__main__':
    main()
