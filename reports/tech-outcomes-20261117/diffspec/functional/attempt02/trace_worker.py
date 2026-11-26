"""Observation-only worker extension for the requested real-request regression."""
import hashlib
import inspect
import json
from pathlib import Path


class TraceWorker:
    def install_trace(self):
        import torch
        import diffspec.kernels as kernels
        import diffspec.runtime as runtime

        cache = self.model_runner.drafter.diffspec_cache
        target = Path(__file__).resolve().parent / 'artifacts/traces'
        target.mkdir(parents=True, exist_ok=True)
        path = target / f'rank-{self.rank}.jsonl'
        self._trace_events = 0
        current = {}
        original_scores = kernels.paged_chunk_attention_scores
        original_accept = cache.accept_target_queries

        def scores(query, keys, lse, table, lengths, **kwargs):
            event = current.get('event')
            if event is not None:
                event['kernel_sequence_lengths'] = lengths.detach().cpu().tolist()
                event['kernel_batch_rows'] = int(table.shape[0])
                indices = torch.tensor(event['committed_query_indices'],
                                       device=cache._target_query.device)
                expected = cache._target_query.index_select(0, indices)
                actual_cpu, expected_cpu = query.detach().cpu(), expected.detach().cpu()
                event['kernel_queries_match_last_committed'] = torch.equal(actual_cpu, expected_cpu)
                if not event['kernel_queries_match_last_committed']:
                    event['query_comparison_diagnostic'] = {
                        'actual_nan': int(torch.isnan(actual_cpu).sum()),
                        'expected_nan': int(torch.isnan(expected_cpu).sum()),
                        'equal_including_nan': bool(torch.allclose(actual_cpu, expected_cpu,
                            rtol=0, atol=0, equal_nan=True)),
                    }
            return original_scores(query, keys, lse, table, lengths, **kwargs)

        def accept(count_tensor, *, is_prefill):
            count_before = cache.metrics.retrieval_count
            active = list(cache._active_slots)
            saved = None
            original_metadata_rows = len(cache._target_query_ends)
            if (not is_prefill and len(active) == 2 and original_metadata_rows == 2
                    and cache._target_query is not None
                    and (cache._retrieval_cycle + 1) % cache.controller.retrieval_interval == 0):
                names = ['_target_query', '_target_lse', '_target_block_tables',
                         '_target_query_ends', '_target_seq_lens']
                saved = {name: getattr(cache, name) for name in names}
                end = saved['_target_query_ends'][-1]
                query = saved['_target_query'][:end]
                lse = cache._target_lse_by_token()[:end]
                cache._target_query = torch.cat([query, torch.zeros_like(query[:1])])
                cache._target_lse = torch.cat([lse, torch.zeros_like(lse[:1])])
                table = saved['_target_block_tables'][:2]
                cache._target_block_tables = torch.cat([table, torch.zeros_like(table[:1])])
                cache._target_query_ends = saved['_target_query_ends'][:2] + [end + 1]
                cache._target_seq_lens = saved['_target_seq_lens'][:2] + [1]
            counts = count_tensor.detach().reshape(-1).cpu().tolist()[:len(active)]
            ends = list(cache._target_query_ends)
            sequence_lengths = list(cache._target_seq_lens)
            starts = [0, *ends[:max(0, len(active) - 1)]]
            query_lengths = [end - start for end, start in zip(ends[:len(active)], starts)]
            rows = [i for i, count in enumerate(counts) if count > 0]
            committed = [min(max(count, 1), size) for count, size in zip(counts, query_lengths)]
            event = {
                'rank': self.rank, 'active_slots': active,
                'request_slots': dict(cache._request_slots), 'is_prefill': bool(is_prefill),
                'valid_counts': counts, 'attention_query_ends': ends,
                'attention_sequence_lengths': sequence_lengths,
                'query_lengths': query_lengths, 'real_request_count': len(active),
                'metadata_rows': len(ends),
                'metadata_rows_before_fixture': original_metadata_rows,
                'padding_rows_added': int(saved is not None),
                'padding_source': 'controlled_extra_row_on_real_target_snapshot'
                    if saved is not None else 'native_metadata',
                'expected_retrieval_sequence_lengths': [
                    sequence_lengths[i] - query_lengths[i] + committed[i] for i in rows],
                'committed_query_indices': [starts[i] + committed[i] - 1 for i in rows],
                'has_rejected_suffix': any(counts[i] < query_lengths[i] for i in rows),
            }
            current['event'] = event
            try:
                result = original_accept(count_tensor, is_prefill=is_prefill)
            finally:
                current.clear()
                if saved is not None:
                    for name, value in saved.items():
                        setattr(cache, name, value)
            if cache.metrics.retrieval_count > count_before:
                event['retrieval_cycle'] = cache._retrieval_cycle
                selected = []
                for row, limit in zip(rows, event['expected_retrieval_sequence_lengths']):
                    indices = cache._selected_token_indices[active[row]].detach().cpu().tolist()
                    selected.append({
                        'row': row, 'slot': active[row], 'committed_prefix_end': limit,
                        'selected_chunks': sorted({i // cache.settings.chunk_size for i in indices}),
                        'selected_token_count': len(indices),
                        'selected_token_max': max(indices, default=-1),
                        'selected_tokens_sha256': hashlib.sha256(
                            json.dumps(indices, separators=(',', ':')).encode()).hexdigest(),
                        'rejected_suffix_excluded': all(i < limit for i in indices),
                    })
                event['selected'] = selected
                event['kernel_boundaries_match'] = event.get('kernel_sequence_lengths') == event[
                    'expected_retrieval_sequence_lengths']
                event['dummy_excluded'] = (
                    len(ends) == len(active) + 1
                    and event.get('kernel_batch_rows') == len(rows)
                    and len(selected) == len(rows))
                with path.open('a') as output:
                    output.write(json.dumps(event) + '\n')
                self._trace_events += 1
            return result

        kernels.paged_chunk_attention_scores = scores
        cache.accept_target_queries = accept
        runtime_source = Path(inspect.getsourcefile(runtime.DiffSpecDraftCache))
        self._trace_identity = {
            'rank': self.rank, 'device_uuid': str(torch.npu.get_device_properties().uuid),
            'runtime_source': str(runtime_source),
            'runtime_sha256': hashlib.sha256(runtime_source.read_bytes()).hexdigest(),
            'proposer': type(self.model_runner.drafter).__name__,
            'trace_path': str(path), 'production_source_modified': False,
            'real_query_values_modified': False,
            'padding_fixture': 'one extra metadata/query/LSE/table row on two real requests; restored after retrieval',
        }
        return self._trace_identity

    def trace_status(self):
        cache = self.model_runner.drafter.diffspec_cache
        return {**self._trace_identity, 'events': self._trace_events,
                'retrieval_status': cache.retrieval_status()}
