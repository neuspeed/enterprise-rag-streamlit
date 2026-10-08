"""Source fetching tests.

Every loader funnels through :class:`SourceClient`, so the guards tested here —
the size cap and the SSRF check — are the only thing standing between a
knowledge base upload and a fetch of ``file://`` or the instance metadata
endpoint.
"""

from __future__ import annotations

import pytest

from know_everything_ai.settings import Settings
from know_everything_ai.utils.source_client import SourceClient, guess_extension


@pytest.fixture
def client() -> SourceClient:
    return SourceClient(Settings(_env_file=None, MAX_UPLOAD_MB=1))


def test_size_cap_comes_from_max_upload_mb(client):
    """MAX_UPLOAD_MB is the documented knob and must be the one that applies.

    It used to be shadowed by a separate MAX_SOURCE_BYTES, so setting
    MAX_UPLOAD_MB=1 changed nothing and the only symptom was an OOM under load.
    """
    assert client.max_bytes == 1024 * 1024


def test_max_upload_mb_scales_the_cap():
    assert (
        SourceClient(Settings(_env_file=None, MAX_UPLOAD_MB=250)).max_bytes
        == 250 * 1024 * 1024
    )


def test_empty_webdav_base_stays_empty():
    """`"" .rstrip("/") + "/"` is "/", which is truthy and would remap every URL."""
    client = SourceClient(Settings(_env_file=None, WEBDAV_BASE_URL=""))
    assert client.base_url == ""
    assert client.auth is None


def test_webdav_base_keeps_a_single_trailing_slash():
    client = SourceClient(
        Settings(_env_file=None, WEBDAV_BASE_URL="https://dav.example.com/files")
    )
    assert client.base_url == "https://dav.example.com/files/"


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://example.com/a/b.pdf", ".pdf"),
        ("https://example.com/a/b.PDF", ".pdf"),
        ("https://example.com/a/b.docx?token=1", ".docx"),
        ("https://example.com/a/b", ""),
        ("https://example.com/", ""),
    ],
)
def test_extension_is_guessed_from_the_path_not_the_query(url, expected):
    assert guess_extension(url) == expected


def test_extension_falls_back_when_absent():
    assert guess_extension("https://example.com/a/b", ".txt") == ".txt"


def test_is_url_distinguishes_urls_from_paths():
    assert SourceClient.is_url("https://example.com/a.pdf") is True
    assert SourceClient.is_url("http://example.com/a.pdf") is True
    assert SourceClient.is_url("/srv/files/a.pdf") is False
    assert SourceClient.is_url("a.pdf") is False
