"""Shared async HTTP transport for the Flowise API.

Every Flowise call used to be a blocking ``requests`` call issued from inside
``async def`` code, so a single slow store lookup stalled the whole event loop —
the same failure class as the webhook bug in :mod:`utils.webhook_utils`.

One :class:`httpx.AsyncClient` is now created per transport instance and reused
across every call. The pipeline owns one transport for its lifetime, which also
removes the per-message connection-pool leak: a client built per request never
releases its sockets, and file-descriptor exhaustion shows up as a worker that
slowly stops accepting new work.
"""

from __future__ import annotations

from typing import Any

import httpx
import structlog

log = structlog.get_logger("flowise_transport")


class FlowiseAPIError(RuntimeError):
    """Raised when the Flowise API answers with a non-success status."""

    def __init__(self, status_code: int, body: str, url: str) -> None:
        super().__init__(f"Flowise API error {status_code} for {url}: {body}")
        self.status_code = status_code
        self.body = body
        self.url = url


class FlowiseTransport:
    """Async request helper owning a connection pool."""

    def __init__(
        self,
        base_url: str,
        api_key: str = "",
        *,
        timeout: float = 60.0,
        upsert_timeout: float = 600.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or ""
        self.timeout = timeout
        self.upsert_timeout = upsert_timeout
        # An injected client is owned by the caller: we must not close it.
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(timeout=timeout)

    def headers(self, *, multipart: bool = False) -> dict[str, str]:
        headers: dict[str, str] = {}
        # With files= httpx builds a multipart/form-data body and has to emit
        # the boundary itself. Forcing application/json here produced a request
        # Flowise's multipart parser rejected with "Boundary not found".
        if not multipart:
            headers["Content-Type"] = "application/json"
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    async def request(
        self,
        method: str,
        path: str,
        *,
        timeout: float | None = None,
        expect_json: bool = True,
        **kwargs: Any,
    ) -> Any:
        """Issue one request and return the decoded body.

        ``expect_json=False`` covers endpoints that answer 204 with no body, in
        which case an empty dict is returned instead of a decode error.
        """
        url = f"{self.base_url}{path}"
        is_multipart = bool(kwargs.get("files"))
        kwargs.setdefault("headers", self.headers(multipart=is_multipart))

        try:
            response = await self._client.request(
                method, url, timeout=timeout or self.timeout, **kwargs
            )
        except httpx.HTTPError as exc:
            log.warning("flowise_request_failed", method=method, url=url, error=str(exc))
            raise FlowiseAPIError(0, str(exc), url) from exc

        if response.status_code >= 400:
            log.warning(
                "flowise_request_rejected",
                method=method,
                url=url,
                status=response.status_code,
            )
            raise FlowiseAPIError(response.status_code, response.text, url)

        if response.status_code == 204 or not expect_json:
            return {}
        if not response.content:
            return {}
        return response.json()

    async def stream_lines(self, path: str, payload: dict[str, Any]):
        """Yield SSE ``data:`` payloads from a streaming endpoint."""
        async with self._client.stream(
            "POST",
            f"{self.base_url}{path}",
            json=payload,
            headers=self.headers(),
            timeout=self.timeout,
        ) as response:
            if response.status_code >= 400:
                body = await response.aread()
                raise FlowiseAPIError(response.status_code, body.decode(errors="replace"), path)
            async for line in response.aiter_lines():
                if line.startswith("data:"):
                    yield line[len("data:") :].strip()

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()
