"""Publish one job to the stand's RabbitMQ queue for a manual end-to-end check.

Run it from the host against the published RabbitMQ port:

    .venv/Scripts/python.exe tools/publish_job.py --id 1005

The defaults point at the host-published ports, so no flags are needed on the
stand. Inside the compose network, pass --mq-url with the rabbitmq hostname
instead. The result lands in logs/webhook/next_return_<job_id>_*.json, written
by the sink container.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os

import aio_pika


async def publish(queue: str, payload: dict) -> None:
    """Publish one job; aio_pika because that is what the worker ships with."""
    url = os.getenv("MQ_URL") or (
        f"amqp://{os.getenv('MQ_USER', 'guest')}:{os.getenv('MQ_PASS', 'guest')}"
        f"@{os.getenv('MQ_HOST', '127.0.0.1')}:{os.getenv('MQ_PORT', '5672')}/%2F"
    )
    params = aio_pika.connect_robust(url)
    connection = await params
    async with connection:
        channel = await connection.channel()
        await channel.default_exchange.publish(
            aio_pika.Message(
                body=json.dumps(payload).encode(),
                content_type="application/json",
            ),
            routing_key=queue,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", type=int, required=True)
    parser.add_argument("--job-id", type=int, default=None)
    parser.add_argument("--kb-type", default="auto")
    parser.add_argument("--kb-external-id", default="acme_manual_check")
    parser.add_argument("--data", default="http://files/sample_ru.pdf")
    parser.add_argument("--queue", default="dev_knowledge_base_tasks.acme")
    parser.add_argument(
        "--mq-url",
        default=None,
        help="AMQP URL. Defaults to the host-published RabbitMQ port; "
        "inside the compose network use amqp://guest:guest@rabbitmq:5672/%%2F.",
    )
    parser.add_argument(
        "--webhook",
        default="http://webhook_sink/hook",
        help="Where the worker posts the result. This URL is resolved by the "
        "worker, not by you, so the default names the compose service. The "
        "sink publishes no host port, so there is nothing to point at 127.0.0.1.",
    )
    args = parser.parse_args()

    payload = {
        "id": args.id,
        "job_id": args.job_id or args.id + 4000,
        "kb_type": args.kb_type,
        "kb_external_id": args.kb_external_id,
        "data": args.data,
        "external_url": args.webhook,
    }
    if args.mq_url:
        os.environ["MQ_URL"] = args.mq_url
    asyncio.run(publish(args.queue, payload))
    print(f"published id={payload['id']} to {args.queue}")


if __name__ == "__main__":
    main()
