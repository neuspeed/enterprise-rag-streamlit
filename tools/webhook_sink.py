"""Webhook sink for the local stand.

The worker always finishes by posting a ``ReturnPayload`` to the buyer's
webhook. For a local run there is no buyer, so this receives it, prints a
summary and writes the full body next to itself so a failed enrichment can be
inspected after the fact.

Runs on port 80 inside the compose network because the worker validates
``external_url`` against a hostname-and-optional-port pattern; ``:8099`` works
too, but keeping the URL port-less makes the demo payload easier to read.

    python tools/webhook_sink.py [--port 80] [--out ./logs/webhook]
"""

from __future__ import annotations

import argparse
import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path

from aiohttp import web

ROUTER = web.RouteTableDef()


@ROUTER.post("/hook")
async def receive(request: web.Request) -> web.Response:
    try:
        payload = await request.json()
    except (json.JSONDecodeError, ValueError):
        body = await request.text()
        print(f"[webhook] non-JSON body ({len(body)} bytes): {body[:200]!r}", flush=True)
        return web.json_response({"ok": False}, status=400)

    text_out = payload.get("text_out") or ""
    out_dir: Path = request.app["out_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    job = payload.get("job_id", "unknown")
    target = out_dir / f"return_{job}_{stamp}.json"
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[webhook] job={job} kb_type={payload.get('kb_type')} "
          f"status={payload.get('status')} reason={payload.get('reason')}", flush=True)
    print(f"[webhook] text_out: {len(text_out)} chars -> {target}", flush=True)
    print(f"[webhook] cost: {json.dumps(payload.get('cost'), ensure_ascii=False)}", flush=True)

    if text_out:
        preview = text_out[:600]
        print(f"[webhook] preview:\n{preview}\n", flush=True)

    return web.json_response({"ok": True})


@ROUTER.get("/health")
async def health(_: web.Request) -> web.Response:
    return web.json_response({"ok": True})


async def _serve(port: int, out_dir: Path) -> None:
    app = web.Application()
    app["out_dir"] = out_dir
    app.add_routes(ROUTER)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", port)
    await site.start()
    print(f"[webhook] listening on 0.0.0.0:{port}, writing to {out_dir}", flush=True)

    stop = asyncio.Event()
    try:
        await stop.wait()
    finally:
        await runner.cleanup()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=80)
    parser.add_argument("--out", type=Path, default=Path("./logs/webhook"))
    args = parser.parse_args()

    try:
        asyncio.run(_serve(args.port, args.out))
    except KeyboardInterrupt:
        print("[webhook] stopped", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
