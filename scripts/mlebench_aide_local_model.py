"""Bind pinned AIDE to one local OpenAI-compatible model, fail closed."""

from urllib.parse import urlsplit


def bind_local_openai(backend, *, model: str, api_base: str) -> None:
    endpoint = urlsplit(api_base)
    if (
        endpoint.scheme != "http"
        or endpoint.hostname != "127.0.0.1"
        or endpoint.port is None
        or endpoint.path not in {"/v1", "/v1/"}
        or endpoint.username is not None
        or endpoint.password is not None
        or endpoint.query
        or endpoint.fragment
    ):
        raise ValueError("AIDE local endpoint must be explicit IPv4 loopback /v1")
    if not model or model != model.strip():
        raise ValueError("an exact served model ID is required")

    import httpx
    import openai

    original = backend.backend_openai
    if (
        original._client is not None
        or backend.provider_to_query_func["openai"] is not original.query
    ):
        raise RuntimeError("bind before AIDE initializes the original OpenAI backend")
    client = openai.OpenAI(
        api_key="local-file-queue",
        base_url=api_base,
        max_retries=0,
        http_client=httpx.Client(trust_env=False, follow_redirects=False),
    )

    def only_bound_model(requested):
        if requested != model:
            raise ValueError(
                "AIDE requested a model outside the declared local binding"
            )
        return "openai"

    def require_bound_client():
        if original._client is not client:
            raise RuntimeError("AIDE local client binding was replaced")

    original._client = client
    original._setup_openai_client = require_bound_client
    backend.determine_provider = only_bound_model
