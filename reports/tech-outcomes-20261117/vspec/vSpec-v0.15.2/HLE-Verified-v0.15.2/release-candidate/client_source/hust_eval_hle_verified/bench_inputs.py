"""Import a sealed accuracy artifact into a NEW, independently frozen bench input.

This is not load_plan/resume: old accuracy code and its hash guard stay unchanged.
The complete source seal is checked before importing immutable task data.
"""
import base64
import hashlib
import json
import shutil
from pathlib import Path

from .hashing import sha256_file
from .seal import seal_check


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def write(path, value):
    with Path(path).open('x', encoding='utf8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def image_info(url):
    head, data = url.split(',', 1)
    if not head.startswith('data:image/') or not head.endswith(';base64'):
        raise ValueError('bench images must be embedded image data URLs')
    raw = base64.b64decode(data, validate=True)
    if not raw:
        raise ValueError('empty image')
    return {'mime': head[5:-7], 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


def materialize_sealed(accuracy_run, output, modality, sample=None):
    if modality not in ('text', 'image'):
        raise ValueError('modality must be text or image')
    source, out = Path(accuracy_run).resolve(), Path(output).resolve()
    if source == out or source in out.parents:
        raise ValueError('bench export must be outside the sealed accuracy run')
    if out.exists():
        raise FileExistsError(out)
    if not (source/'MANIFEST.json').is_file():
        raise ValueError('a final accuracy manifest is required')
    check = seal_check(source)
    if not check['ok'] or check['table_matches_manifest'] is not True:
        raise ValueError('source accuracy seal failed verification')
    c = read(source/'plan/contract.json')
    if sha256_file(source/'plan/tasks.json') != c['tasks_sha256']:
        raise ValueError('source tasks do not match the frozen contract')
    if sha256_file(source/'plan/contract.json') != (source/'plan/contract.sha256').read_text().strip():
        raise ValueError('source contract hash mismatch')
    tasks = read(source/'plan/tasks.json')
    eligible = [t for t in tasks if bool(t['image']) == (modality == 'image')]
    count = len(eligible)
    if sample is not None:
        if type(sample) is not int or not 0 < sample <= count:
            raise ValueError('sample exceeds eligible rows; no repetition')
        eligible = [eligible[i*count//sample] for i in range(sample)]
    if not eligible or len({t['case_id'] for t in eligible}) != len(eligible):
        raise ValueError('empty selection or duplicate case IDs')
    out.mkdir(parents=True)
    refs = {}
    for rel in ('MANIFEST.json', 'EVIDENCE_BUNDLE.json', 'plan/contract.json', 'source.json'):
        src = source/rel
        dst = out/'source'/rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        refs[rel] = sha256_file(dst)
    write(out/'source/seal-check.json', check)
    cases = []
    with (out/'prompts.jsonl').open('x', encoding='utf8') as f:
        for t in eligible:
            entry = {'case_id': t['case_id'], 'track': t['track'], 'row_index': t['row_index'],
                     'modality': modality, 'text_sha256': hashlib.sha256(t['problem'].encode()).hexdigest()}
            if modality == 'image':
                entry['image'] = image_info(t['image'])
                row = {'content': [{'type': 'text', 'text': t['problem']},
                                   {'type': 'image_url', 'image_url': {'url': t['image']}}]}
            else:
                row = {'prompt': t['problem']}
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
            cases.append(entry)
    manifest = {'schema': 1, 'kind': 'independent_bench_export_from_sealed_accuracy',
                'source_accuracy_run': str(source), 'source_sha256': refs,
                'source_tasks_sha256': c['tasks_sha256'], 'modality': modality,
                'eligible_count': count, 'selection': 'all' if sample is None else 'floor(i*N/sample), frozen task order',
                'cases': cases, 'sha256': sha256_file(out/'prompts.jsonl'),
                'note': 'Original question plus original embedded image bytes where present; no answers or official solver system prompt. Not an accuracy request replay.'}
    write(out/'prompts.jsonl.manifest.json', manifest)
    return manifest


def validate_input(dataset, modality, manifest=None):
    rows = []
    with Path(dataset).open(encoding='utf8') as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        raise ValueError('empty dataset')
    items = []
    for row in rows:
        if modality == 'text':
            if set(row) != {'prompt'} or not isinstance(row['prompt'], str):
                raise ValueError('text input must contain only prompt')
            text = row['prompt']
            image = None
        elif modality == 'image':
            parts = row.get('content', [])
            if set(row) != {'content'} or len(parts) != 2 or parts[0].get('type') != 'text' or parts[1].get('type') != 'image_url':
                raise ValueError('image input requires original text and embedded image')
            text = parts[0]['text']
            image = image_info(parts[1]['image_url']['url'])
        else:
            raise ValueError('unknown modality')
        items.append((hashlib.sha256(text.encode()).hexdigest(), image))
    if manifest:
        m = read(manifest)
        if m['sha256'] != sha256_file(dataset) or m['modality'] != modality or len(m['cases']) != len(rows):
            raise ValueError('input/manifest mismatch')
        if len({c['case_id'] for c in m['cases']}) != len(rows):
            raise ValueError('duplicate case IDs')
        for case, (text_hash, image) in zip(m['cases'], items):
            if case['text_sha256'] != text_hash or case.get('image') != image:
                raise ValueError('text or decoded image hash mismatch')
    elif modality == 'image':
        raise ValueError('image input requires a provenance manifest')
    return len(rows)
