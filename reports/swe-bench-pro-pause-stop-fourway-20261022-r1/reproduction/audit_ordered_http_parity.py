"""CPU-only controlled event comparison, never a hardware performance replay.

Feed identical ordered first-round inputs through official and product HTTP.
Hold each fixture engine return separately, then return/release in input order
among admitted programs. No scheduler timer or real model/network is started.
"""
import argparse
from contextvars import ContextVar
import asyncio
import hashlib
import importlib.util
import json
from pathlib import Path

import httpx
from ThunderAgent.backend import state as backend_state
from ThunderAgent.backend.vllm_metrics import VLLMCacheConfig
from thunderagent_controller_mod.upstream_adapter import create_application


def reference_factory():
    path = Path(__file__).resolve().parents[1] / 'experiments/workload/upstream_reference_service.py'
    spec = importlib.util.spec_from_file_location('ordered_official_reference', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.create_reference_application


async def ordered_trace(factory, inputs, *, capacity, buffer_tokens):
    names = [item['task'] for item in inputs]
    if not names or len(set(names)) != len(names):
        raise ValueError('Require nonempty unique program identities')
    if capacity <= 0 or buffer_tokens < 0:
        raise ValueError('Invalid capacity/buffer')
    saved_buffer = backend_state.BUFFER_PER_PROGRAM
    backend_state.BUFFER_PER_PROGRAM = buffer_tokens
    try:
        app = factory('http://engine.invalid')
    except BaseException:
        backend_state.BUFFER_PER_PROGRAM = saved_buffer
        raise
    router = app.state.upstream_router
    backend = router.backends['http://engine.invalid']
    backend.metrics_client.cache_config = VLLMCacheConfig(block_size=1, num_gpu_blocks=capacity)
    backend.metrics_client.healthy = True
    entered = {name: asyncio.Event() for name in names}
    delivered = {name: asyncio.Event() for name in names}
    gates = {name: asyncio.Event() for name in names}
    forwarded = {}
    tasks = {}
    current_program = ContextVar('ordered_fixture_program')
    original_before = router.update_program_before_request

    async def observed_before(program_id, state, payload):
        current_program.set(program_id)
        entered[program_id].set()
        return await original_before(program_id, state, payload)

    router.update_program_before_request = observed_before

    async def engine(request):
        # Observe coroutine identity without changing the forwarded model payload.
        name = current_program.get()
        digest = hashlib.sha256(request.content).hexdigest()
        forwarded[name] = digest
        delivered[name].set()
        await gates[name].wait()
        return httpx.Response(200, json={
            'choices': [{'message': {'role': 'assistant', 'content': 'fixture'}}],
            'usage': {'prompt_tokens': 40, 'completion_tokens': 2, 'total_tokens': 42}})

    await router.client.aclose()
    router.client = httpx.AsyncClient(transport=httpx.MockTransport(engine))
    client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://fixture')

    def snapshot():
        return {'active': sorted(backend._programs),
                'waiting': sorted(router.global_waiting_queue),
                'remaining_capacity': backend.remaining_capacity(),
                'future_paused_tokens': backend.future_paused_tokens,
                'programs': {name: {'status': p.status.value, 'tokens': p.total_tokens,
                                    'step': p.step_count, 'backend': p.backend_url}
                             for name, p in sorted(router.programs.items())}}

    try:
        for item in inputs:
            name = item['task']
            payload = {**item['rounds'][0], 'program_id': name}
            tasks[name] = asyncio.create_task(client.post('/v1/chat/completions', json=payload))
            await asyncio.wait_for(entered[name].wait(), 5)
        trace = [snapshot()]
        completion_order = []
        remaining = set(names)
        while remaining:
            # Yield for resumed request coroutines to enter fixture transport.
            for _ in range(100):
                ready = [name for name in names if name in remaining and delivered[name].is_set()]
                if ready:
                    break
                await asyncio.sleep(0)
            if not ready:
                errors = [str(tasks[name].exception()) for name in remaining if tasks[name].done()]
                raise RuntimeError('No admitted fixture request; errors=' + repr(errors))
            name = ready[0]
            gates[name].set()
            response = await asyncio.wait_for(tasks[name], 5)
            if response.status_code != 200:
                raise RuntimeError('Fixture request failed: ' + str(response.status_code))
            released = await client.post('/programs/release', json={'program_id': name})
            if not released.json()['released']:
                raise RuntimeError('Program not released')
            await router._greedy_resume()
            completion_order.append(name)
            remaining.remove(name)
            trace.append(snapshot())
        return {'trace': trace, 'completion_order': completion_order,
                'forwarded_payload_sha256': forwarded,
                'loaded_source_sha256': app.state.runtime_identity['loaded_sources']}
    finally:
        for task in tasks.values():
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks.values(), return_exceptions=True)
        await client.aclose()
        await router.client.aclose()
        backend_state.BUFFER_PER_PROGRAM = saved_buffer


async def compare(inputs, capacity=1654784, buffer_tokens=14335):
    official = await ordered_trace(reference_factory(), inputs, capacity=capacity, buffer_tokens=buffer_tokens)
    product = await ordered_trace(create_application, inputs, capacity=capacity, buffer_tokens=buffer_tokens)
    return {'equal': official == product, 'official': official, 'product': product,
            'scope': 'CPU controlled ordered entry/fixture returns/release; no periodic timer, '
                     'no real engine, no historical server arrival reconstruction or performance claim',
            'fixture_usage': {'prompt_tokens': 40, 'completion_tokens': 2},
            'periodic_scheduler_started': False, 'new_model_requests': 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--input-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    raw = args.input.read_bytes()
    if hashlib.sha256(raw).hexdigest() != args.input_sha256:
        raise ValueError('Input checksum mismatch')
    inputs = json.loads(raw)
    results = {}
    for label, ordered in [('input_order', inputs), ('reversed_input_order', list(reversed(inputs)))]:
        results[label] = asyncio.run(compare(ordered))
    result = {'input_sha256': args.input_sha256, 'comparisons': results,
              'audit_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'capacity_tokens': 1654784, 'buffer_tokens_per_program': 14335}
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps({label: {'equal': value['equal'],
                             'initial_active': len(value['official']['trace'][0]['active']),
                             'completed': len(value['official']['completion_order'])}
                      for label, value in results.items()}))
    return 0 if all(value['equal'] for value in results.values()) else 1


if __name__ == '__main__':
    raise SystemExit(main())
