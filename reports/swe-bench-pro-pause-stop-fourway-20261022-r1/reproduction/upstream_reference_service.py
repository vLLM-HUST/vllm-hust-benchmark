"""Independent official HTTP/router reference for adapter acceptance.

Uses register_routes directly, never create_application or the product's HTTP
wrapper. Shared transport, explicit timeout and hardware capacity adaptations
are declared separately; this is original scheduling, not all-default settings.
"""
import argparse
from contextlib import asynccontextmanager
import hashlib
import inspect
import json
from pathlib import Path


def create_reference_application(backend, *, capacity_evidence=None,
                                 resume_timeout_seconds=None,
                                 transport_timeout_seconds=None):
    import ThunderAgent
    from ThunderAgent import Config, MultiBackendRouter, register_routes, set_config
    from fastapi import FastAPI
    from thunderagent_controller_mod.upstream_adapter import PIN
    from thunderagent_controller_mod.upstream_capacity import apply_capacity_evidence
    from thunderagent_controller_mod.upstream_diagnostics import register_scheduler_diagnostics
    from thunderagent_controller_mod.upstream_resume_timeout import install_resume_timeout
    from thunderagent_controller_mod.upstream_transport import configure_transport

    manifest_path = Path(__file__).resolve().parents[2] / 'src/thunderagent_controller_mod/upstream_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    root = Path(ThunderAgent.__file__).parent
    loaded = []
    for name, expected in manifest['files'].items():
        path = root / name
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError('Pinned upstream source mismatch: ' + name)
        loaded.append({'file': str(path), 'sha256': actual})
    config = Config(backends=[backend], router_mode='tr', backend_type='vllm',
                    metrics_enabled=True, metrics_interval=5.0,
                    scheduler_interval=5.0, acting_token_weight=1.0)
    set_config(config)
    router = MultiBackendRouter(config.backends, scheduling_enabled=True,
                                scheduler_interval=5.0, backend_type='vllm',
                                acting_token_weight=1.0, use_acting_token_decay=False)
    identity = {'implementation': 'independent-official-route-reference',
                'upstream_commit': PIN, 'loaded_sources': loaded,
                'router_class': type(router).__module__ + '.' + type(router).__qualname__,
                'router_file': inspect.getfile(type(router)),
                'scheduling_enabled': True, 'scheduler_interval_seconds': 5.0,
                'metrics_interval_seconds': 5.0, 'acting_token_weight': 1.0,
                'use_acting_token_decay': False,
                'http_entrypoint': 'ThunderAgent.app.register_routes',
                'product_http_wrapper': False}
    if resume_timeout_seconds is not None:
        identity['resume_timeout_adaptation'] = install_resume_timeout(router, resume_timeout_seconds)

    @asynccontextmanager
    async def lifespan(app):
        identity['transport'] = await configure_transport(
            router, 'no-keepalive', transport_timeout_seconds)
        await router.start()
        try:
            if capacity_evidence is not None:
                identity['capacity_evidence'] = apply_capacity_evidence(router, capacity_evidence)
            print('REFERENCE_RUNTIME_IDENTITY ' + json.dumps(identity), flush=True)
            yield
        finally:
            await router.stop()

    app = FastAPI(lifespan=lifespan)
    register_routes(app, router, config)
    register_scheduler_diagnostics(app, router)
    app.state.upstream_router = router
    app.state.runtime_identity = identity

    @app.get('/migration-identity')
    async def migration_identity():
        return identity

    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', required=True)
    parser.add_argument('--port', type=int, required=True)
    parser.add_argument('--capacity-evidence', type=Path)
    parser.add_argument('--resume-timeout-seconds', type=float)
    parser.add_argument('--transport-timeout-seconds', type=float)
    args = parser.parse_args()
    import uvicorn
    app = create_reference_application(
        args.backend, capacity_evidence=args.capacity_evidence,
        resume_timeout_seconds=args.resume_timeout_seconds,
        transport_timeout_seconds=args.transport_timeout_seconds)
    uvicorn.run(app, host='0.0.0.0', port=args.port)


if __name__ == '__main__':
    main()
