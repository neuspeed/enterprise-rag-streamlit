"""Fetching source documents for ingestion.

Handles two shapes of source:

* a plain ``http(s)`` URL, optionally behind an authenticated WebDAV mount;
* any URL when a WebDAV base is configured, in which case the URL's path is
  remapped onto that base so private Nextcloud/ownCloud links work.

The client is also the SSRF boundary. Sources arrive from user-supplied
payloads, so an unguarded fetch turns the service into a proxy for whatever the
container can reach: the cloud metadata endpoint, an internal admin panel, a
Redis with no auth. Redirects are therefore followed manually and every hop is
re-validated.
"""

from __future__ import annotations

import asyncio
import ipaddress
import socket
from dataclasses import dataclass
from pathlib import Path
from typing import Final
from urllib.parse import unquote, urlparse, urlunparse

import aiofiles
import aiohttp
import structlog
from yarl import URL

from know_everything_ai.settings import Settings

log = structlog.get_logger("source_client")

CHUNK_SIZE: Final = 64 * 1024

REDIRECT_STATUSES: Final = frozenset({301, 302, 303, 307, 308})

DEFAULT_USER_AGENT: Final = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/125.0 Safari/537.36 know-everything-ai"
)


class SourceFetchError(Exception):
    """A source could not be retrieved."""


class UnsafeURLError(SourceFetchError):
    """A source URL resolves to a non-public address."""


@dataclass(frozen=True, slots=True)
class FetchResult:
    path: Path
    source_url: str
    content_type: str | None
    size: int


class SourceClient:
    """Streams a remote source to a local file."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        # An empty base must stay empty: "" .rstrip("/") + "/" would yield "/",
        # which is truthy and would remap every source URL onto the root path.
        self.base_url = (
            f"{settings.WEBDAV_BASE_URL.rstrip('/')}/"
            if settings.WEBDAV_BASE_URL
            else ""
        )
        self.auth = (
            aiohttp.BasicAuth(settings.WEBDAV_USER, settings.WEBDAV_PASSWORD)
            if settings.WEBDAV_BASE_URL and settings.WEBDAV_USER
            else None
        )
        self.max_bytes = settings.MAX_UPLOAD_MB * 1024 * 1024
        self.timeout = aiohttp.ClientTimeout(total=settings.SOURCE_HTTP_TIMEOUT)

    @staticmethod
    def is_url(value: str) -> bool:
        cleaned = str(value).replace("\\/", "/").strip()
        return cleaned.lower().startswith(("http://", "https://"))

    def resolve(self, url: str) -> str:
        """Map a source URL onto the configured WebDAV base, when there is one."""
        cleaned = str(url).replace("\\/", "/").strip()
        if not self.base_url:
            return cleaned

        parsed = urlparse(cleaned)
        base = urlparse(self.base_url)
        base_path = base.path.rstrip("/")

        # The URL is already inside the mount when it shares the host and its
        # path starts with the base path. Comparing whole paths here would
        # double the prefix, because the base path always carries a trailing
        # slash while a document URL carries the file name after it.
        already_mounted = parsed.netloc == base.netloc and (
            parsed.path == base_path or parsed.path.startswith(f"{base_path}/")
        )
        relative = (
            parsed.path[len(base_path):].lstrip("/")
            if already_mounted
            else parsed.path.lstrip("/")
        )

        return urlunparse(
            (base.scheme, base.netloc, f"{base_path}/{relative}", "", parsed.query, "")
        )

    async def fetch_to_file(
        self,
        url: str,
        dest_path: Path,
        *,
        max_bytes: int | None = None,
    ) -> FetchResult:
        """Download ``url`` into ``dest_path``, streaming to disk."""
        limit = max_bytes if max_bytes is not None else self.max_bytes
        url = self.resolve(url)

        await self._assert_public(url)

        dest_path.parent.mkdir(parents=True, exist_ok=True)

        timeout = self.timeout
        headers = {"User-Agent": DEFAULT_USER_AGENT, "Accept": "*/*"}

        async with aiohttp.ClientSession(
            timeout=timeout,
            headers=headers,
        ) as session:
            current = url
            for hop in range(self.settings.SOURCE_MAX_REDIRECTS + 1):
                async with session.get(
                    current,
                    allow_redirects=False,
                    auth=self._auth_for(current),
                ) as response:
                    if response.status in REDIRECT_STATUSES:
                        location = response.headers.get("Location")
                        if not location:
                            raise SourceFetchError(
                                f"Redirect without Location header: {current}"
                            )
                        current = str(response.url.join(URL(location)))
                        await self._assert_public(current)
                        log.debug("source_redirect", url=url, hop=hop + 1, target=current)
                        continue

                    if response.status >= 400:
                        raise SourceFetchError(
                            f"Source returned HTTP {response.status} for {current}"
                        )

                    declared = response.content_length
                    if declared is not None and declared > limit:
                        raise SourceFetchError(
                            f"Source is {declared} bytes, limit is {limit}"
                        )

                    written = 0
                    async with aiofiles.open(dest_path, "wb") as handle:
                        async for chunk in response.content.iter_chunked(CHUNK_SIZE):
                            written += len(chunk)
                            if written > limit:
                                raise SourceFetchError(
                                    f"Source exceeded the {limit} byte limit"
                                )
                            await handle.write(chunk)

                    log.info(
                        "source_fetched",
                        url=url,
                        dest=str(dest_path),
                        size=written,
                    )
                    return FetchResult(
                        path=dest_path,
                        source_url=url,
                        content_type=response.headers.get("Content-Type"),
                        size=written,
                    )

        raise SourceFetchError(
            f"Exceeded {self.settings.SOURCE_MAX_REDIRECTS} redirects for {url}"
        )

    def _auth_for(self, url: str) -> aiohttp.BasicAuth | None:
        """Only authenticate against the configured WebDAV host.

        Credentials are attached per request rather than session-wide so that a
        redirect to an unrelated host cannot leak them.
        """
        if self.auth is None:
            return None
        target = urlparse(url)
        base = urlparse(self.base_url)
        if target.netloc == base.netloc and target.path.startswith(base.path):
            return self.auth
        return None

    async def _assert_public(self, url: str) -> None:
        """Reject URLs that resolve to loopback, private or reserved addresses."""
        if self.settings.ALLOW_PRIVATE_URLS:
            return

        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            raise UnsafeURLError(f"Unsupported scheme: {parsed.scheme!r}")

        host = parsed.hostname
        if not host:
            raise UnsafeURLError(f"URL has no host: {url!r}")

        port = parsed.port or (443 if parsed.scheme == "https" else 80)

        try:
            addresses = [ipaddress.ip_address(host)]
        except ValueError:
            try:
                infos = await asyncio.get_running_loop().getaddrinfo(
                    host, port, type=socket.SOCK_STREAM
                )
            except socket.gaierror as exc:
                raise UnsafeURLError(f"Cannot resolve host {host!r}: {exc}") from exc
            addresses = []
            for info in infos:
                try:
                    addresses.append(ipaddress.ip_address(info[4][0]))
                except ValueError:
                    continue

        if not addresses:
            raise UnsafeURLError(f"Host {host!r} resolved to no usable address")

        for address in addresses:
            if (
                address.is_private
                or address.is_loopback
                or address.is_link_local
                or address.is_multicast
                or address.is_reserved
                or address.is_unspecified
            ):
                raise UnsafeURLError(
                    f"Host {host!r} resolves to non-public address {address}"
                )


def guess_extension(url: str, default: str = "") -> str:
    """Best-effort file extension for a source URL.

    The extension drives loader selection, so it is lowercased here rather than
    at each comparison site: a source served as ``report.PDF`` must resolve to
    the same loader as ``report.pdf``.
    """
    path = urlparse(str(url).replace("\\/", "/")).path
    suffix = Path(unquote(path)).suffix
    if not suffix or len(suffix) > 10:
        return default
    return suffix.lower()
