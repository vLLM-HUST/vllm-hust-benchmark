import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from thunderagent_controller_mod.upstream_capacity import apply_capacity_evidence


class CapacityEvidenceTests(unittest.TestCase):
    def test_checked_evidence_changes_capacity_and_retains_block_identity(self):
        raw = SimpleNamespace(block_size=2048, num_gpu_blocks=25, total_tokens_capacity=51200)
        backend = SimpleNamespace(cache_config=raw, metrics_client=SimpleNamespace(cache_config=raw))
        router = SimpleNamespace(backends={'engine': backend})
        evidence = dict(kind='vllm-startup-hybrid-concurrency-estimate', tokens=29257,
                        max_model_len=16384, block_size=2048, num_gpu_blocks=25, source_sha256='a'*64)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'evidence.json';path.write_text(json.dumps(evidence))
            identity = apply_capacity_evidence(router, path)
            self.assertEqual(identity['raw_block_derived_tokens'], 51200)
            self.assertEqual(backend.metrics_client.cache_config.total_tokens_capacity, 29257)
            self.assertEqual(backend.metrics_client.cache_config.num_gpu_blocks, 25)
            self.assertEqual(raw.total_tokens_capacity, 51200)

    def test_mismatched_live_configuration_is_rejected_without_changes(self):
        raw = SimpleNamespace(block_size=2048, num_gpu_blocks=26, total_tokens_capacity=53248)
        backend = SimpleNamespace(cache_config=raw, metrics_client=SimpleNamespace(cache_config=raw))
        evidence = dict(kind='vllm-startup-hybrid-concurrency-estimate', tokens=29257,
                        max_model_len=16384, block_size=2048, num_gpu_blocks=25, source_sha256='a'*64)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'evidence.json';path.write_text(json.dumps(evidence))
            with self.assertRaisesRegex(ValueError,'does not match'):
                apply_capacity_evidence(SimpleNamespace(backends={'engine':backend}),path)
            self.assertIs(backend.metrics_client.cache_config,raw)

    def test_whole_request_budget_requires_explicit_floor_not_arbitrary_throttle(self):
        raw = SimpleNamespace(block_size=2048, num_gpu_blocks=25, total_tokens_capacity=51200)
        backend = SimpleNamespace(cache_config=raw, metrics_client=SimpleNamespace(cache_config=raw))
        evidence = dict(kind='vllm-startup-whole-request-capacity', tokens=16384, startup_tokens=29257,
                        max_model_len=16384, block_size=2048, num_gpu_blocks=25, source_sha256='a'*64)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'evidence.json';path.write_text(json.dumps(evidence))
            apply_capacity_evidence(SimpleNamespace(backends={'engine':backend}),path)
            self.assertEqual(backend.metrics_client.cache_config.total_tokens_capacity,16384)
            evidence['tokens']=8192;path.write_text(json.dumps(evidence))
            with self.assertRaisesRegex(ValueError,'must floor'):
                apply_capacity_evidence(SimpleNamespace(backends={'engine':backend}),path)

    def test_hybrid_reservation_uses_official_capacity_checks(self):
        import hashlib
        from ThunderAgent.backend import state
        from ThunderAgent.program.state import Program, ProgramStatus
        raw = SimpleNamespace(block_size=2048,num_gpu_blocks=25,total_tokens_capacity=51200)
        backend = state.BackendState('engine',metrics_client=SimpleNamespace(cache_config=raw))
        audit = dict(status='PASS',evidence_type='source-derived-capacity-model',
            resolved_groups=[dict(type=kind,block_size=2048) for kind in ['MambaSpec']*3+['FullAttentionSpec']],
            engine_num_blocks=25,reconstructed_num_blocks=25,
            usable_blocks_after_source_block_pool_initialization=24,
            per_program_mamba_blocks_upper_bound=6,proposed_usable_token_budget=49152,
            proposed_per_program_buffer_tokens=14335)
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            audit_bytes = json.dumps(audit).encode()
            (directory/'audit.json').write_bytes(audit_bytes)
            evidence = dict(kind='vllm-source-derived-hybrid-reservation',tokens=49152,
                max_model_len=16384,block_size=2048,num_gpu_blocks=25,
                source_sha256=hashlib.sha256(audit_bytes).hexdigest(),audit_file='audit.json',
                per_program_buffer_tokens=14335)
            (directory/'evidence.json').write_text(json.dumps(evidence))
            previous = state.BUFFER_PER_PROGRAM
            try:
                apply_capacity_evidence(SimpleNamespace(backends={'engine':backend}),directory/'evidence.json')
                self.assertEqual(state.BUFFER_PER_PROGRAM,14335)
                backend.register_program('long',Program('long',total_tokens=13242,status=ProgramStatus.REASONING))
                self.assertFalse(backend.has_capacity(extra_tokens=13242,extra_count=1))
                self.assertTrue(backend.has_capacity(extra_tokens=4096,extra_count=1))
                self.assertEqual(backend.remaining_capacity(),49152-13242-14335)
                self.assertEqual(backend.capacity_overflow(),0)
            finally:
                state.BUFFER_PER_PROGRAM = previous
