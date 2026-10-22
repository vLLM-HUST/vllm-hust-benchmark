"""Explicit pause-selection correction around the unchanged pinned upstream loop.

Only the synchronous pause loop may count marked REASONING tokens as future
release. Admission, resume and diagnostics continue to see actual capacity.
"""
import ast
from contextvars import ContextVar
import functools
import hashlib
import inspect
import json
import textwrap
import time


POLICY = 'projected-future-release-v1'


def install_pause_projection(router):
    from ThunderAgent.scheduler.router import MultiBackendRouter

    if hasattr(router, '_pause_projection_identity'):
        raise ValueError('Pause projection already installed')
    original_pause = router._pause_until_safe
    source = textwrap.dedent(inspect.getsource(original_pause))
    if any(isinstance(node, (ast.Await, ast.Yield, ast.YieldFrom))
           for node in ast.walk(ast.parse(source))):
        raise ValueError('Pause loop now yields; fresh concurrency audit required')
    if getattr(original_pause, '__func__', None) is not MultiBackendRouter._pause_until_safe:
        raise ValueError('Expected the original pinned upstream pause loop')
    scope = ContextVar('upstream_pause_projection_scope', default=None)
    state = {'policy': POLICY, 'decisions': 0, 'projected_capacity_reads': 0,
             'admission_uses_future_release': False,
             'original_pause_source_sha256': hashlib.sha256(source.encode()).hexdigest()}
    for backend in router.backends.values():
        original_remaining = backend.remaining_capacity

        def bind_capacity(backend, original_remaining):
            @functools.wraps(original_remaining)
            def remaining():
                actual = original_remaining()
                if scope.get() is backend:
                    state['projected_capacity_reads'] += 1
                    return actual + max(0, backend.future_paused_tokens)
                return actual
            return remaining

        backend.remaining_capacity = bind_capacity(backend, original_remaining)

    @functools.wraps(original_pause)
    async def pause_until_safe(backend):
        before = {'remaining_capacity': backend.remaining_capacity(),
                  'future_paused_tokens': backend.future_paused_tokens,
                  'active_program_count': backend.active_program_count}
        token = scope.set(backend)
        try:
            # The exact pinned loop selects candidates and performs all state
            # transitions. Its checked source contains no await/yield points.
            await original_pause(backend)
        finally:
            scope.reset(token)
            state['decisions'] += 1
            event = {'epoch': time.time(), 'policy': POLICY, 'backend': backend.url,
                     'before': before, 'after': {
                         'remaining_capacity': backend.remaining_capacity(),
                         'future_paused_tokens': backend.future_paused_tokens,
                         'active_program_count': backend.active_program_count}}
            print('PAUSE_CAPACITY_DECISION ' + json.dumps(event), flush=True)

    router._pause_until_safe = pause_until_safe
    router._pause_projection_identity = state
    return state
