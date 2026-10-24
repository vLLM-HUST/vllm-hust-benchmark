from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import Mapping
from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING, Any

from vllm_kv_materialization.live_control import (
    HEADER_HOME_RANK,
    HEADER_PRIMARY_ANCHOR,
    HEADER_QUEUE_PRESSURE,
    HEADER_REQUEST_ID,
    HEADER_REUSE_CONFIDENCE,
    HEADER_SECONDARY_ANCHORS,
    HEADER_SHARED_PREFIX_TOKENS,
    HEADER_TURN_INDEX,
    HEADER_WORKLOAD_CASE,
    HEADER_WORKLOAD_FAMILY,
    bind_request_headers,
    build_runtime_control_extra_args,
    compute_runtime_control,
    record_observation,
    reset_request_headers,
)

if TYPE_CHECKING:
    from vllm.plugins.request_processing import RequestProcessingContext

logger = logging.getLogger(__name__)

EXTENSION_ID = "org.vllm-hust.kv-materialization-arrival-control"
PLUGIN_NAME = "kv_materialization"
REQUIRED_REQUEST_PROCESSING_API = "1.0"
REQUIRED_KV_MATERIALIZATION_API = "1.0"
_ENABLED_BUNDLES_ENV = "VLLMHUST_EXT_ENABLED_BUNDLES"
_VLLM_PLUGINS_ENV = "VLLM_PLUGINS"
_RUNTIME_EVENT_LOG_ENV = "VLLM_KV_MATERIALIZATION_RUNTIME_EVENT_LOG_PATH"
_REQUEST_HEADERS = (
    HEADER_REQUEST_ID,
    HEADER_PRIMARY_ANCHOR,
    HEADER_SECONDARY_ANCHORS,
    HEADER_SHARED_PREFIX_TOKENS,
    HEADER_REUSE_CONFIDENCE,
    HEADER_QUEUE_PRESSURE,
    HEADER_WORKLOAD_CASE,
    HEADER_WORKLOAD_FAMILY,
    HEADER_TURN_INDEX,
    HEADER_HOME_RANK,
)

_REGISTERED = False


def _legacy_vllm_023_available() -> bool:
    """Return whether the host is the supported deployed compatibility line."""

    try:
        from vllm import __version__ as host_version
    except (ImportError, AttributeError):
        return False
    return str(host_version).split("+", maxsplit=1)[0].startswith("0.23.")


def _package_version() -> str:
    try:
        return version("vllm-kv-materialization")
    except PackageNotFoundError:
        return "source-tree"


def observe_runtime_event(payload: Mapping[str, object]) -> None:
    """Persist an engine-owned receipt only when explicitly configured."""

    path = os.getenv(_RUNTIME_EVENT_LOG_ENV)
    if not path:
        return
    record = {"timestamp_s": time.time(), **payload}
    encoded = (json.dumps(record, sort_keys=True) + "\n").encode()
    fd = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)
    try:
        os.write(fd, encoded)
    finally:
        os.close(fd)


def _activation_requested() -> bool:
    """Require explicit Manager or direct-vLLM activation intent."""

    raw = os.getenv(_ENABLED_BUNDLES_ENV)
    if raw is not None:
        enabled = {item.strip() for item in raw.split(",") if item.strip()}
        return EXTENSION_ID in enabled

    allowed_plugins = {
        item.strip()
        for item in os.getenv(_VLLM_PLUGINS_ENV, "").split(",")
        if item.strip()
    }
    return PLUGIN_NAME in allowed_plugins


def process_request(
    context: RequestProcessingContext,
) -> Mapping[str, Any]:
    """Create engine metadata through the public request-processing hook."""

    token = bind_request_headers(context.headers)
    try:
        observation, plan = compute_runtime_control(
            context.request_id,
            context.prompt_tokens,
            context.max_tokens,
        )
        record_observation(observation)
    finally:
        reset_request_headers(token)
    return build_runtime_control_extra_args(plan)


def register_plugin() -> None:
    """Register the MOD through the versioned vLLM-HUST host seam."""

    global _REGISTERED
    if _REGISTERED or not _activation_requested():
        return

    host_api = "native-1.0"
    try:
        from vllm.plugins.request_processing import (
            REQUEST_PROCESSING_HOOK_API_VERSION,
            register_request_processor,
        )
        from vllm.v1.core.kv_materialization import (
            KV_MATERIALIZATION_RUNTIME_CONTROL_API_VERSION,
            register_kv_materialization_runtime_observer,
        )
    except ImportError as error:
        if not _legacy_vllm_023_available():
            raise RuntimeError(
                "KV materialization requires the versioned vLLM-HUST "
                "request-processing and KV runtime hooks; the installed host is "
                "unsupported"
            ) from error
        from vllm_kv_materialization.legacy_vllm_023 import (
            register_legacy_vllm_023_adapter,
        )

        register_legacy_vllm_023_adapter()
        host_api = "legacy-vllm-0.23-adapter"
        os.environ["VLLM_KV_MATERIALIZATION_PLUGIN_LOADED"] = "1"
        os.environ["VLLM_KV_MATERIALIZATION_PLUGIN_MODE"] = os.getenv(
            "VLLM_KV_MATERIALIZATION_PLUGIN_MODE",
            "prefix_cache_runtime_control",
        )
        _REGISTERED = True
        logger.info(
            "Registered extension=%s distribution_version=%s host_api=%s",
            EXTENSION_ID,
            _package_version(),
            host_api,
        )
        print(
            f"KV_MATERIALIZATION_PLUGIN_REGISTERED host_api={host_api}",
            flush=True,
        )
        return

    if REQUEST_PROCESSING_HOOK_API_VERSION != REQUIRED_REQUEST_PROCESSING_API:
        raise RuntimeError(
            "Unsupported vLLM-HUST request-processing hook API: "
            f"expected {REQUIRED_REQUEST_PROCESSING_API}, got "
            f"{REQUEST_PROCESSING_HOOK_API_VERSION}"
        )

    if (
        KV_MATERIALIZATION_RUNTIME_CONTROL_API_VERSION
        != REQUIRED_KV_MATERIALIZATION_API
    ):
        raise RuntimeError(
            "Unsupported vLLM-HUST KV materialization API: "
            f"expected {REQUIRED_KV_MATERIALIZATION_API}, got "
            f"{KV_MATERIALIZATION_RUNTIME_CONTROL_API_VERSION}"
        )

    register_request_processor(
        PLUGIN_NAME,
        process_request,
        header_names=_REQUEST_HEADERS,
    )
    register_kv_materialization_runtime_observer(
        PLUGIN_NAME,
        observe_runtime_event,
    )
    os.environ["VLLM_KV_MATERIALIZATION_PLUGIN_LOADED"] = "1"
    os.environ["VLLM_KV_MATERIALIZATION_PLUGIN_MODE"] = os.getenv(
        "VLLM_KV_MATERIALIZATION_PLUGIN_MODE",
        "prefix_cache_runtime_control",
    )
    _REGISTERED = True
    logger.info(
        "Registered extension=%s distribution_version=%s request_api=%s "
        "kv_materialization_api=%s",
        EXTENSION_ID,
        _package_version(),
        REQUEST_PROCESSING_HOOK_API_VERSION,
        KV_MATERIALIZATION_RUNTIME_CONTROL_API_VERSION,
    )
    print(
        f"KV_MATERIALIZATION_PLUGIN_REGISTERED host_api={host_api}",
        flush=True,
    )
