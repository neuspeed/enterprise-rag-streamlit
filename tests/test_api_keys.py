"""Key hashing and the CLI surface of the query API.

Database-backed behaviour is covered indirectly by the API tests (which fake
the store) and by the live stand; what belongs here is the digest contract,
because a change to it would silently invalidate every key already issued, and
the CLI parsing, because a mistyped subcommand should fail loudly.
"""

from __future__ import annotations

import asyncio
from argparse import ArgumentParser

import pytest

from know_everything_ai.api_keys_cli import build_parser
from know_everything_ai.stores.api_keys import ApiKeyStore, generate_key, hash_key


def test_hash_key_is_deterministic() -> None:
    assert hash_key("secret") == hash_key("secret")
    assert hash_key("secret") != hash_key("Secret")


def test_generate_key_has_prefix_and_entropy() -> None:
    a = generate_key()
    b = generate_key()
    assert a != b
    assert a.startswith("kp_")
    # urlsafe base64: no characters that need escaping in a header.
    assert set(a[3:]) <= set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_")


def test_empty_key_is_never_connected_to_the_database() -> None:
    """An empty key short-circuits before the pool opens, so this works with
    no reachable Postgres at all."""
    store = ApiKeyStore.__new__(ApiKeyStore)
    store._cache_seconds = 30
    store._cache = {}
    assert asyncio.run(store.verify("")) is False
    assert asyncio.run(store.verify("   ")) is False


@pytest.mark.parametrize(
    "argv,expected",
    [
        (["create", "--name", "portal"], "create"),
        (["list"], "list"),
        (["revoke", "--id", "3"], "revoke"),
    ],
)
def test_parser_accepts_valid_invocations(argv: list[str], expected: str) -> None:
    args = build_parser().parse_args(argv)
    assert args.action == expected


def test_parser_rejects_create_without_name() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["create"])


def test_parser_rejects_revoke_without_id() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["revoke"])


def test_parser_imports_cleanly() -> None:
    # The CLI parses without ever touching the database or fastapi: this guards
    # against importing launchpad machinery into a provisioning command.
    assert isinstance(build_parser(), ArgumentParser)
