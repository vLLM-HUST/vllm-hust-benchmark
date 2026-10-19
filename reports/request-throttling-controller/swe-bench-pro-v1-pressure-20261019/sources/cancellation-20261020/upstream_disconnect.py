"""Cancel non-streaming upstream forwarding when its HTTP caller disconnects.

This is a transport boundary; pinned scheduling and payloads are unchanged.
"""
import asyncio
import logging
from contextlib import suppress

from fastapi.responses import Response

logger = logging.getLogger(__name__)


async def forward_until_disconnect(request, operation, *, poll_interval=0.05):
    """Await forwarding or disconnect, and always reap both child tasks.

    Cancelling httpx's in-flight POST closes its downstream connection. The
    engine's disconnect handling remains responsible for aborting generation;
    real-engine verification is required before claiming that abort succeeded.
    """
    async def disconnected():
        while not await request.is_disconnected():
            await asyncio.sleep(poll_interval)

    forwarding = asyncio.create_task(operation)
    watching = asyncio.create_task(disconnected())
    try:
        await asyncio.wait((forwarding, watching), return_when=asyncio.FIRST_COMPLETED)
        if forwarding.done():
            return await forwarding
        # Propagate a failed disconnect observer rather than silently cancelling
        # a live request because observation itself failed.
        await watching
        forwarding.cancel()
        with suppress(asyncio.CancelledError):
            await forwarding
        logger.info('CLIENT_DISCONNECTED downstream_forward_cancelled path=%s', request.url.path)
        return Response(status_code=499)
    finally:
        for task in (forwarding, watching):
            if not task.done():
                task.cancel()
        await asyncio.gather(forwarding, watching, return_exceptions=True)
