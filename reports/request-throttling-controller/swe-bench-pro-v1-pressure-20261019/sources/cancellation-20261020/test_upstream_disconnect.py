"""HTTP disconnect must stop waiting and in-flight upstream operations."""
import asyncio
from types import SimpleNamespace

import pytest

from thunderagent_controller_mod.upstream_disconnect import forward_until_disconnect


class Request:
    url = SimpleNamespace(path='/v1/chat/completions')

    def __init__(self):
        self.disconnected = False

    async def is_disconnected(self):
        return self.disconnected


def test_success_and_error_are_preserved():
    async def check():
        request = Request()
        sentinel = object()
        async def success():
            return sentinel
        assert await forward_until_disconnect(request, success()) is sentinel
        async def failed():
            raise ValueError('upstream error')
        with pytest.raises(ValueError, match='upstream error'):
            await forward_until_disconnect(request, failed())
        assert len(asyncio.all_tasks()) == 1
    asyncio.run(check())


def test_disconnect_cancels_and_reaps_downstream():
    async def check():
        request = Request()
        entered, stopped = asyncio.Event(), asyncio.Event()
        async def upstream():
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                stopped.set()
        task = asyncio.create_task(forward_until_disconnect(request, upstream(), poll_interval=0.001))
        await entered.wait()
        request.disconnected = True
        response = await asyncio.wait_for(task, 1)
        assert response.status_code == 499
        assert stopped.is_set()
        assert len(asyncio.all_tasks()) == 1
    asyncio.run(check())


def test_server_cancellation_reaps_downstream_and_watcher():
    async def check():
        entered, stopped = asyncio.Event(), asyncio.Event()
        async def upstream():
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                stopped.set()
        task = asyncio.create_task(forward_until_disconnect(Request(), upstream()))
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert stopped.is_set()
        assert len(asyncio.all_tasks()) == 1
    asyncio.run(check())


def test_observer_failure_is_not_reported_as_client_disconnect():
    async def check():
        class FailedRequest(Request):
            async def is_disconnected(self):
                raise RuntimeError('observation failed')
        stopped = asyncio.Event()
        async def upstream():
            try:
                await asyncio.Event().wait()
            finally:
                stopped.set()
        with pytest.raises(RuntimeError, match='observation failed'):
            await forward_until_disconnect(FailedRequest(), upstream())
        assert stopped.is_set()
        assert len(asyncio.all_tasks()) == 1
    asyncio.run(check())


def test_official_paused_wait_is_cancelled_before_release(monkeypatch):
    """Release must not wake a disconnected request into backend forwarding."""
    import json
    from starlette.requests import Request as HTTPRequest
    from ThunderAgent.program import ProgramState
    from thunderagent_controller_mod.upstream_adapter import create_application

    async def check():
        app = create_application('http://example.invalid:8000')
        router = app.state.upstream_router
        program = router.get_or_create_program('paused-client')
        program.step_count = 1
        program.state = ProgramState.PAUSED
        program.waiting_event = asyncio.Event()
        router._add_to_global_waiting_queue_sync('paused-client', program, backend=None)
        calls = []
        async def forbidden_forward(*args, **kwargs):
            calls.append(args)
            raise AssertionError('Disconnected queued request reached backend')
        monkeypatch.setattr(router, 'proxy_request', forbidden_forward)
        disconnected = asyncio.Event()
        sent_body = False
        async def receive():
            nonlocal sent_body
            if not sent_body:
                sent_body = True
                return {'type': 'http.request', 'body': json.dumps({'program_id': 'paused-client', 'messages': [], 'stream': False}).encode()}
            await disconnected.wait()
            return {'type': 'http.disconnect'}
        request = HTTPRequest({'type': 'http', 'method': 'POST', 'path': '/v1/chat/completions', 'headers': []}, receive)
        route = next(r for r in app.routes if getattr(r, 'path', None) == '/v1/chat/completions')
        try:
            task = asyncio.create_task(route.endpoint(request))
            for _ in range(100):
                if program.step_count == 2:
                    break
                await asyncio.sleep(0.001)
            assert program.step_count == 2 and not task.done()
            disconnected.set()
            assert (await asyncio.wait_for(task, 1)).status_code == 499
            assert await router.release_program('paused-client')
            await asyncio.sleep(0)
            assert not calls and not router.programs and not router.global_waiting_queue
        finally:
            await router.client.aclose()
    asyncio.run(check())
