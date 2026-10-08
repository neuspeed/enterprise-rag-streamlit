"""``python -m know_everything_ai apikey ...``

Key issuance lives here rather than in the API service because the two have
opposite requirements: issuing needs no HTTP server at all and must work in a
container with no web framework installed, while serving needs both.

The plaintext key exists for exactly one line of output. It is never logged and
never stored, so losing it means issuing a new one.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from know_everything_ai.settings import Settings
from know_everything_ai.stores.api_keys import ApiKeyStore


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="know-everything-ai apikey",
        description="Issue and retire bearer keys for the query API.",
    )
    sub = parser.add_subparsers(dest="action", required=True)

    create = sub.add_parser("create", help="Mint a new key and print it once.")
    create.add_argument(
        "--name",
        required=True,
        help="Who or what this key belongs to, e.g. 'portal'.",
    )

    sub.add_parser("list", help="List known keys. Never prints the secret.")

    revoke = sub.add_parser("revoke", help="Retire a key by id.")
    revoke.add_argument("--id", type=int, required=True, help="Key id from `list`.")

    return parser


async def run(settings: Settings, argv: Sequence[str]) -> int:
    args = build_parser().parse_args(argv)
    store = ApiKeyStore(settings)
    try:
        if args.action == "create":
            issued = await store.create(args.name)
            # stdout only, so it can be captured without the surrounding
            # structured log. Printed exactly once: there is no way to read it
            # back, by design.
            print(f"id={issued.id}")
            print(f"name={issued.name}")
            print(f"key={issued.plain}")
            print(
                "The key above is shown once and cannot be recovered. "
                "Store it now; revoke it with `apikey revoke --id`."
            )
            return 0

        if args.action == "list":
            rows = await store.list()
            if not rows:
                print("No keys. Create one with `apikey create --name <name>`.")
                return 0
            print(f"{'id':>4}  {'name':<24}  {'created':<20}  state")
            for row in rows:
                created = row.get("created_at") or ""
                state = "revoked" if row.get("revoked_at") else "active"
                print(
                    f"{row['id']:>4}  {str(row['name']):<24}  "
                    f"{str(created):<20}  {state}"
                )
            return 0

        # revoke
        if not await store.revoke(args.id):
            # Not an error to log with a stack trace: the caller simply named a
            # key that does not exist, or one already retired.
            print(f"No active key with id={args.id}.", file=sys.stderr)
            return 1
        print(f"Revoked key id={args.id}. In-flight requests using it are "
              f"rejected once the in-memory cache expires.")
        return 0
    finally:
        await store.aclose()
