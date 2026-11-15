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
                event['kernel_queries_match_last_committed'] = torch.equal(
                    query.detach().cpu(), expected.detach().cpu())
            return original_scores(query, keys, lse, table, lengths, **kwargs)

        def accept(count_tensor, *, is_prefill):
            count_before = cache.metrics.retrieval_count
            active = list(cache._active_slots)
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
            'trace_path': str(path), 'input_or_output_mutations': False,
        }
        return self._trace_identity

    def trace_status(self):
        cache = self.model_runner.drafter.diffspec_cache
        return {**self._trace_identity, 'events': self._trace_events,
                'retrieval_status': cache.retrieval_status()}
