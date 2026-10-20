"""Explicit engine-capacity evidence at the pinned scheduler's input boundary.

Hybrid caches cannot infer context capacity from block_size*num_gpu_blocks.
The vLLM startup concurrency estimate is configuration dependent, rather than
an exact bound for every request length. No scheduling policy is replaced.
"""
import json
import hashlib
import importlib
from pathlib import Path


class CapacityConfig:
    def __init__(self, original, tokens):
        self.original = original
        self.total_tokens_capacity = tokens

    def __getattr__(self, name):
        return getattr(self.original, name)


def apply_capacity_evidence(router, evidence_path):
    evidence = json.loads(Path(evidence_path).read_text())
    if evidence.get('kind') not in ('vllm-startup-hybrid-concurrency-estimate', 'vllm-startup-whole-request-capacity', 'vllm-source-derived-hybrid-reservation'):
        raise ValueError('Unsupported capacity evidence kind')
    tokens = evidence.get('tokens')
    if type(tokens) is not int or tokens <= 0:
        raise ValueError('Capacity tokens must be a positive integer')
    if type(evidence.get('max_model_len')) is not int or evidence['max_model_len'] <= 0:
        raise ValueError('Capacity evidence requires configured max_model_len')
    if evidence['kind'] == 'vllm-startup-whole-request-capacity':
        startup = evidence.get('startup_tokens')
        if type(startup) is not int or startup < evidence['max_model_len']:
            raise ValueError('Startup estimate cannot hold one configured full request')
        conservative = (startup // evidence['max_model_len']) * evidence['max_model_len']
        if tokens != conservative:
            raise ValueError('Whole-request budget must floor the startup concurrency estimate')
    digest = evidence.get('source_sha256', '')
    if len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
        raise ValueError('Capacity evidence requires source SHA256')
    reservation = None
    if evidence['kind'] == 'vllm-source-derived-hybrid-reservation':
        audit_path = Path(evidence_path).parent / evidence['audit_file']
        audit_bytes = audit_path.read_bytes()
        if hashlib.sha256(audit_bytes).hexdigest() != evidence['source_sha256']:
            raise ValueError('Hybrid audit SHA256 mismatch')
        audit = json.loads(audit_bytes)
        if audit.get('status') != 'PASS' or audit.get('evidence_type') != 'source-derived-capacity-model':
            raise ValueError('Hybrid audit must pass the source-derived contract')
        groups = audit['resolved_groups']
        block_size = evidence['block_size']
        if (len(groups) != 4 or [g['type'] for g in groups] != ['MambaSpec'] * 3 + ['FullAttentionSpec']
                or any(g['block_size'] != block_size for g in groups)
                or audit['engine_num_blocks'] != evidence['num_gpu_blocks']
                or audit['reconstructed_num_blocks'] != evidence['num_gpu_blocks']
                or audit['usable_blocks_after_source_block_pool_initialization'] != evidence['num_gpu_blocks'] - 1
                or audit['per_program_mamba_blocks_upper_bound'] != 6):
            raise ValueError('Unsupported hybrid layout; a fresh source audit is required')
        reservation = 6 * block_size + block_size - 1
        if (tokens != (evidence['num_gpu_blocks'] - 1) * block_size
                or tokens != audit['proposed_usable_token_budget']
                or reservation != audit['proposed_per_program_buffer_tokens']
                or evidence.get('per_program_buffer_tokens') != reservation):
            raise ValueError('Hybrid reservation must follow audited group costs')
    applied = []
    # Validate every backend before changing any input.
    for backend in router.backends.values():
        original = backend.cache_config
        if original is None:
            raise ValueError('Engine cache configuration is unavailable')
        if (original.block_size != evidence.get('block_size') or
                original.num_gpu_blocks != evidence.get('num_gpu_blocks')):
            raise ValueError('Capacity evidence does not match live cache configuration')
        if tokens > original.total_tokens_capacity:
            raise ValueError('Evidence cannot enlarge raw block-derived capacity')
        applied.append((backend, original))
    if not applied:
        raise ValueError('No backends available for capacity evidence')
    if reservation is not None:
        # The pinned router imports this existing parameter at admission/resume.
        # Set once before readiness in an isolated proxy process; no policy rewrite.
        state = importlib.import_module('ThunderAgent.backend.state')
        if state.BUFFER_PER_PROGRAM != 100:
            raise ValueError('Official per-program buffer was already modified')
        state.BUFFER_PER_PROGRAM = reservation
    for backend, original in applied:
        backend.metrics_client.cache_config = CapacityConfig(original, tokens)
    return {**evidence, 'raw_block_derived_tokens': applied[0][1].total_tokens_capacity,
            'scope': ('Source-derived per-program upper reservation; approximate token estimation and cache sharing remain validation boundaries' if reservation is not None else 'Explicit startup estimate; request-length dependent, not an exact universal hard limit')}
