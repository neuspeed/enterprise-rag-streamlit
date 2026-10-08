"""Optional job-result callback.

The previous implementation used blocking ``requests`` inside the event loop,
wrapped in ``@backoff.on_exception`` with no ``max_tries``. A single webhook
endpoint returning 429 therefore retried forever and froze every concurrent
job in the process.

Delivery is now asynchronous and bounded: at most ``WEBHOOK_MAX_RETRIES``
attempts, exponential backoff with a ceiling, and a total time budget. Callbacks
are advisory — a job's result is already durable by the time this runs — so a
failed callback is logged, never raised.
"""

from __future__ import annotations

import asyncio
from typing import Any

import aiohttp
import structlog

from know_everything_ai.settings import Settings

log = structlog.get_logger("webhook_sender")

RETRYABLE_STATUSES = frozenset({408, 425, 429, 500, 502, 503, 504})


class WebhookSender:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    async def send(
        self,
        url: str,
        payload: dict[str, Any] | str,
        *,
        max_attempts: int | None = None,
    ) -> bool:
        """POST ``payload`` to ``url``. Returns whether delivery succeeded."""
        if not url:
            return False

        body = payload if isinstance(payload, str) else None
        attempts = max_attempts or self.settings.WEBHOOK_MAX_RETRIES
        timeout = aiohttp.ClientTimeout(total=self.settings.WEBHOOK_TIMEOUT)

        async with aiohttp.ClientSession(
            timeout=timeout,
            headers={"Content-Type": "application/json", "Accept": "*/*"},
        ) as session:
            for attempt in range(1, attempts + 1):
                try:
                    async with session.post(url, data=body, json=None if body else payload) as response:
                        if response.status < 400:
                            log.info(
                                "webhook_delivered",
                                url=url,
                                status=response.status,
                                attempt=attempt,
                            )
                            return True

                        retryable = response.status in RETRYABLE_STATUSES
                        log.warning(
                            "webhook_rejected",
                            url=url,
                            status=response.status,
                            attempt=attempt,
                            retryable=retryable,
                        )
                        if not retryable:
                            return False

                except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                    log.warning(
                        "webhook_failed",
                        url=url,
                        attempt=attempt,
                        error=str(exc),
                    )

                if attempt < attempts:
                    await asyncio.sleep(min(2 ** (attempt - 1), 30))

        log.error("webhook_exhausted", url=url, attempts=attempts)
        return False

    async def send_payload(self, url: str, result) -> bool:
        """Send a ``ReturnPayload``-shaped result model."""
        body = result.model_dump_json() if hasattr(result, "model_dump_json") else str(result)
        return await self.send(url, body)
