"""Thin lifecycle binding to pinned upstream; no scheduling policy replacement.

Install the pinned official package or put its checkout on PYTHONPATH.
The legacy controller remains a separate implementation until CLI cutover is
explicitly tested. This module does not claim hardware equivalence.
"""
import argparse
import hashlib
import inspect
import json
from contextlib import asynccontextmanager
from pathlib import Path

PIN = '7ddc8610270e56d3b109eed8796b3a4360fc67c9'

def create_application(backend, *, scheduler_interval=5.0, metrics_interval=5.0, scheduling_enabled=True, transport_mode=None, capacity_evidence=None):
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse
    from ThunderAgent import Config, MultiBackendRouter, register_routes, set_config
    manifest_path = Path(__file__).with_name('upstream_manifest.json')
    manifest = json.loads(manifest_path.read_text())
    import ThunderAgent
    source_root = Path(ThunderAgent.__file__).parent
    loaded = []
    for relative, expected in manifest['files'].items():
        path = source_root / relative
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError(f'Pinned upstream source mismatch: {relative}')
        loaded.append({'file':str(path),'sha256':actual})
    config = Config(backends=[backend], router_mode='tr' if scheduling_enabled else 'default', backend_type='vllm',
                    metrics_enabled=True, metrics_interval=metrics_interval,
                    scheduler_interval=scheduler_interval, acting_token_weight=1.0)
    set_config(config)
    router = MultiBackendRouter(config.backends, scheduling_enabled=scheduling_enabled,
                                scheduler_interval=scheduler_interval, backend_type='vllm',
                                acting_token_weight=1.0, use_acting_token_decay=False)
    identity = {'implementation':'pinned-official-thin-adapter','upstream_commit':PIN,
                'router_class':type(router).__module__+'.'+type(router).__qualname__,
                'router_file':inspect.getfile(type(router)), 'backends':list(router.backends),
                'scheduler_interval_seconds':scheduler_interval,'metrics_interval_seconds':metrics_interval,
                'scheduling_enabled':scheduling_enabled,'acting_token_weight':1.0,'use_acting_token_decay':False,'loaded_sources':loaded}
    @asynccontextmanager
    async def lifespan(app):
        from thunderagent_controller_mod.upstream_transport import configure_transport
        transport = await configure_transport(router, transport_mode)
        import json
        print(json.dumps({'effective_router_transport': transport}), flush=True)
        identity['transport']=transport
        await router.start()
        try:
            if capacity_evidence is not None:
                from thunderagent_controller_mod.upstream_capacity import apply_capacity_evidence
                identity['capacity_evidence'] = apply_capacity_evidence(router, capacity_evidence)
            assert list(router.backends)==[backend]
            print(json.dumps({'runtime_identity':identity}),flush=True)
            yield
        finally:
            await router.stop()
    app=FastAPI(lifespan=lifespan)
    register_routes(app,router,config)
    # Wrap only the HTTP boundary. Identified programs use the exact upstream
    # endpoint; ordinary requests create no Program/Event/queue entries.
    completion_route = next(route for route in app.routes
                            if getattr(route, 'path', None) == '/v1/chat/completions')
    official_completion = completion_route.endpoint
    app.routes.remove(completion_route)
    @app.post('/v1/chat/completions')
    async def completion(request: Request):
        try:
            payload = await request.json()
        except Exception:
            return JSONResponse({'detail':'Invalid JSON'}, status_code=400)
        if not isinstance(payload, dict):
            return JSONResponse({'detail':'JSON object required'}, status_code=400)
        extra = payload.get('extra_body')
        identified = ('program_id' in payload or
                      isinstance(extra, dict) and 'program_id' in extra or
                      bool(request.headers.get('x-session-id')))
        async def forward():
            if identified:
                return await official_completion(request)
            # Preserve upstream weight-sync gate even for the ordinary bypass.
            await router._weight_sync_event.wait()
            return await router.proxy_request(router.select_backend_for_new_program_default(), payload)
        if payload.get('stream'):
            return await forward()
        from thunderagent_controller_mod.upstream_disconnect import forward_until_disconnect
        return await forward_until_disconnect(request, forward())
    identity['non_streaming_chat_disconnect_policy'] = 'cancel downstream forwarding on caller disconnect; engine abort requires verification'
    identity['ordinary_request_policy'] = 'untracked upstream forwarding; no Program entries'
    @app.post('/v1/completions')
    async def text_completion(request: Request):
        """Preserve token-ID prompts used by the existing dataset clients.

        Text completions have no Agent program lifecycle. Forward them using
        the pinned upstream transport without creating scheduling entries.
        Never convert frozen token IDs back into a chat conversation.
        """
        from ThunderAgent.scheduler.vllm_request_processor import (
            forward_non_streaming_request, forward_streaming_request,
        )
        try:
            payload = await request.json()
        except Exception:
            return JSONResponse({'detail': 'Invalid JSON'}, status_code=400)
        if not isinstance(payload, dict):
            return JSONResponse({'detail': 'JSON object required'}, status_code=400)
        if 'program_id' in payload or request.headers.get('x-session-id'):
            return JSONResponse({'detail': 'Program lifecycle requires chat completions'}, status_code=400)
        await router._weight_sync_event.wait()
        backend = router.select_backend_for_new_program_default()
        url = backend.completions_url.rsplit('/v1/', 1)[0] + '/v1/completions'
        forward = forward_streaming_request if payload.get('stream') else forward_non_streaming_request
        return await forward(router.client, url, payload)
    identity['text_completion_policy'] = 'untracked upstream transport; frozen prompt preserved'
    app.state.upstream_router=router
    from thunderagent_controller_mod.upstream_diagnostics import register_scheduler_diagnostics
    register_scheduler_diagnostics(app, router)
    app.state.runtime_identity=identity
    @app.get('/migration-identity')
    async def migration_identity():
        return identity
    return app

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--backend',required=True)
    parser.add_argument('--host',default='0.0.0.0')
    parser.add_argument('--port',type=int,default=9000)
    args=parser.parse_args()
    import uvicorn
    uvicorn.run(create_application(args.backend),host=args.host,port=args.port)

if __name__=='__main__':main()
