"""Opt-in experiment observer, installed symmetrically outside product code.

Records method boundaries, not new scheduling decisions. Bounded in-memory
recording adds overhead and is for diagnostics only. Export is explicit and
exclusive; events are not durable until export. Never log payloads or errors'
text. No services, scheduler tasks, model calls or network are started here.
"""
from collections import deque
from contextvars import ContextVar
from functools import wraps
import hashlib
import inspect
import json
from pathlib import Path
import time
import uuid


class ServerEvents:
    def __init__(self, router, service_id, max_events=100000):
        if not isinstance(service_id, str) or not service_id:
            raise ValueError('service_id must be nonempty')
        if type(max_events) is not int or max_events <= 0:
            raise ValueError('max_events must be a positive integer')
        self.instance_id = str(uuid.uuid4())
        self.source_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        self.router = router
        self.service_id = service_id
        self.max_events = max_events
        self.events = deque()
        self.sequence = 0
        self.dropped = 0
        self.snapshot_errors = 0
        self.context = ContextVar('experiment_server_event_context', default={})

    def record(self, event, **values):
        self.sequence += 1
        if len(self.events) >= self.max_events:
            self.dropped += 1
            return
        row = {'event': event, 'sequence': self.sequence, 'service_id': self.service_id,
               'epoch': time.time(), 'monotonic_ns': time.monotonic_ns(),
               **self.context.get(), **values}
        try:
            row['router'] = {'char_to_token_ratio': self.router.char_to_token_ratio,
                'waiting_count': len(self.router.global_waiting_queue),
                'backends': [{
                    'index': index, 'active_count': backend.active_program_count,
                    'active_program_tokens': backend.active_program_tokens,
                    'remaining_capacity_tokens': (None if backend.cache_config is None
                                                   else backend.remaining_capacity()),
                    'capacity_tokens': (None if backend.cache_config is None
                                        else backend.cache_config.total_tokens_capacity),
                    'shared_tokens': backend.shared_tokens,
                    'future_paused_tokens': backend.future_paused_tokens,
                } for index, backend in enumerate(self.router.backends.values())]}
        except Exception as error:
            self.snapshot_errors += 1
            row['snapshot_error_type'] = type(error).__name__
        self.events.append(row)

    def summary(self):
        return {'service_id': self.service_id, 'instance_id': self.instance_id,
                'observer_source_sha256': self.source_sha256,
                'total_seen': self.sequence,
                'recorded': len(self.events), 'dropped': self.dropped,
                'snapshot_errors': self.snapshot_errors, 'max_events': self.max_events,
                'scope': 'Boundary observations with overhead, not atomic engine/cache events',
                'durability': 'In memory until explicit export'}

    def read(self, after_sequence=0):
        if type(after_sequence) is not int or after_sequence < 0:
            raise ValueError('after_sequence must be a nonnegative integer')
        return {'summary': self.summary(),
                'events': [row for row in self.events if row['sequence'] > after_sequence],
                'clock': {'epoch': time.time(), 'monotonic_ns': time.monotonic_ns()}}

    def export(self, path):
        # Do not hide I/O failure or overwrite any earlier diagnostic evidence.
        with Path(path).open('x') as stream:
            for row in self.events:
                stream.write(json.dumps(row, allow_nan=False) + '\n')
            stream.write(json.dumps({'observer_summary': self.summary()}) + '\n')


def install_server_events(app, router, *, service_id, max_events=100000):
    """Observe original bound methods exactly once; no default product hookup."""
    if getattr(router, '_experiment_server_events', None) is not None:
        raise ValueError('Server event observer already installed')
    if app.middleware_stack is not None:
        raise ValueError('Install observer before serving; no live hotpatch')
    recorder = ServerEvents(router, service_id, max_events)

    def details(method, args, kwargs):
        if method == '_resume_program':
            state = args[0] if args else kwargs.get('state')
            program_id = getattr(state, 'program_id', None)
        elif method in ['update_program_before_request', 'update_program_after_request',
                        'release_program', '_mark_program_for_pause', '_pause_program',
                        '_add_to_global_waiting_queue_sync']:
            program_id = args[0] if args else kwargs.get('program_id')
            try:
                state = router.programs.get(program_id)
            except (TypeError, AttributeError):
                return {}
        else:
            program_id = None
            state = None
        return {} if program_id is None else {
            'program_id': program_id,
            'step': None if state is None else state.step_count,
            'program_total_tokens': None if state is None else state.total_tokens,
            'program_status': None if state is None else state.status.value,
            'program_state': None if state is None else state.state.value,
            'marked_for_pause': None if state is None else state.marked_for_pause}

    def outcome(method, result):
        if method == 'update_program_before_request':
            return {'admitted': bool(result)}
        if method == 'release_program':
            return {'released': bool(result)}
        if method == '_select_backend_for_new_program':
            return {'selected_backend_index': next((i for i, url in enumerate(router.backends)
                                                     if url == result), None)}
        return {}

    def observe(target, method, label):
        original = getattr(target, method)
        if inspect.iscoroutinefunction(original):
            @wraps(original)
            async def asynchronous(*args, **kwargs):
                token = recorder.context.set({**recorder.context.get(),
                    'parent_operation_id': recorder.context.get().get('operation_id'),
                    'operation_id': label + ':' + str(recorder.sequence + 1)})
                recorder.record(label + '_start', **details(method, args, kwargs))
                try:
                    result = await original(*args, **kwargs)
                except BaseException as error:
                    recorder.record(label + '_exception', error_type=type(error).__name__,
                                    **details(method, args, kwargs))
                    raise
                else:
                    recorder.record(label + '_end', **details(method, args, kwargs),
                                    **outcome(method, result))
                    return result
                finally:
                    recorder.context.reset(token)
            setattr(target, method, asynchronous)
        else:
            @wraps(original)
            def synchronous(*args, **kwargs):
                recorder.record(label + '_start', **details(method, args, kwargs))
                try:
                    result = original(*args, **kwargs)
                except BaseException as error:
                    recorder.record(label + '_exception', error_type=type(error).__name__,
                                    **details(method, args, kwargs))
                    raise
                recorder.record(label + '_end', **details(method, args, kwargs),
                                    **outcome(method, result))
                return result
            setattr(target, method, synchronous)

    for method, label in [
        ('update_program_before_request', 'admission'),
        ('update_program_after_request', 'backend_body_received'),
        ('_scheduled_check', 'scheduled_check'), ('_greedy_resume', 'greedy_resume'),
        ('_resume_program', 'resume'), ('_mark_program_for_pause', 'mark'),
        ('_pause_program', 'pause'), ('release_program', 'release'),
        ('_select_backend_for_new_program', 'new_program_selection'),
        ('_add_to_global_waiting_queue_sync', 'queue_entry'),
    ]:
        observe(router, method, label)
    for index, backend in enumerate(router.backends.values()):
        observe(backend, 'fetch_metrics', f'metrics_fetch_backend_{index}')

    class Boundary:
        def __init__(self, app):
            self.app = app

        async def __call__(self, scope, receive, send):
            if scope['type'] != 'http' or scope['path'] != '/v1/chat/completions':
                return await self.app(scope, receive, send)
            headers = dict(scope.get('headers', []))
            token = recorder.context.set({
                'request_id': headers.get(b'x-delivery-request-id', b'').decode(errors='replace')
                              or 'server-' + str(recorder.sequence + 1),
                'header_program_id': headers.get(b'x-session-id', b'').decode(errors='replace')})

            async def traced_send(message):
                if message['type'] == 'http.response.start':
                    recorder.record('response_headers_send', status=message['status'])
                elif message['type'] == 'http.response.body':
                    recorder.record('response_body_send', bytes=len(message.get('body', b'')),
                                    more_body=message.get('more_body', False))
                await send(message)

            try:
                recorder.record('request_received')
                return await self.app(scope, receive, traced_send)
            except BaseException as error:
                recorder.record('request_exception', error_type=type(error).__name__)
                raise
            finally:
                recorder.record('request_finished')
                recorder.context.reset(token)

    app.add_middleware(Boundary)

    @app.get('/experiment-server-events/clock')
    async def clock():
        return {'epoch': time.time(), 'monotonic_ns': time.monotonic_ns(),
                'event_sequence': recorder.sequence, 'service_id': service_id,
                'instance_id': recorder.instance_id}

    @app.get('/experiment-server-events')
    async def events(after_sequence: int = 0):
        from fastapi import HTTPException
        if after_sequence < 0:
            raise HTTPException(status_code=400, detail='Negative event cursor')
        return recorder.read(after_sequence)

    router._experiment_server_events = recorder
    app.state.experiment_server_events = recorder
    app.state.runtime_identity['server_event_observer'] = {
        'source_sha256': recorder.source_sha256, 'service_id': service_id,
        'max_events': max_events, 'scope': 'Opt-in experiment boundaries; not performance evidence'}
    return recorder


def create_observed_application(group, backend, *, service_id, max_events=100000,
                                capacity_evidence=None, resume_timeout_seconds=None,
                                transport_timeout_seconds=None):
    """Experiment entrypoint: independent reference stays outside product HTTP."""
    options = dict(capacity_evidence=capacity_evidence,
                   resume_timeout_seconds=resume_timeout_seconds,
                   transport_timeout_seconds=transport_timeout_seconds)
    if group == 'thunderagent_original':
        from experiments.workload.upstream_reference_service import create_reference_application
        app = create_reference_application(backend, **options)
    elif group in ['native', 'controller_original']:
        from thunderagent_controller_mod.upstream_adapter import create_application
        app = create_application(backend, scheduling_enabled=group != 'native',
                                 transport_mode='no-keepalive', **options)
    else:
        raise ValueError('Unknown experiment group')
    install_server_events(app, app.state.upstream_router,
                          service_id=service_id, max_events=max_events)
    app.state.runtime_identity['experiment_group'] = group
    return app


def main():
    import argparse
    import uvicorn
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--group', choices=['native', 'thunderagent_original', 'controller_original'], required=True)
    parser.add_argument('--backend', required=True)
    parser.add_argument('--service-id', required=True)
    parser.add_argument('--port', type=int, required=True)
    parser.add_argument('--capacity-evidence', type=Path, required=True)
    parser.add_argument('--resume-timeout-seconds', type=float, required=True)
    parser.add_argument('--transport-timeout-seconds', type=float, required=True)
    parser.add_argument('--max-events', type=int, default=100000)
    args = parser.parse_args()
    app = create_observed_application(args.group, args.backend, service_id=args.service_id,
        max_events=args.max_events, capacity_evidence=args.capacity_evidence,
        resume_timeout_seconds=args.resume_timeout_seconds,
        transport_timeout_seconds=args.transport_timeout_seconds)
    uvicorn.run(app, host='0.0.0.0', port=args.port)


if __name__ == '__main__':
    main()
