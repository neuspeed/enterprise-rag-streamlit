import asyncio
import signal
import sys
import traceback
from datetime import datetime, timezone
from typing import Dict, Optional

import aio_pika
import structlog
from aiormq.exceptions import ChannelInvalidStateError
from pydantic import ValidationError

from pipeline import KnowledgePipeline
from settings import Settings
from utils.schemas import Payload, ReturnPayload
from utils.webhook_utils import WebhookSender


log = structlog.get_logger("consumer_dispatcher")


class Dispatcher:
    def __init__(self):
        self.settings = Settings()
        self.connection: Optional[aio_pika.RobustConnection] = None
        self.channel: Optional[aio_pika.RobustChannel] = None
        self.queues: Dict[str, aio_pika.RobustQueue] = {}
        self.retry_queues: Dict[tuple[str, int], aio_pika.RobustQueue] = {}
        self.dlq_queue: Optional[aio_pika.RobustQueue] = None
        self.pending_tasks: set[asyncio.Task] = set()
        self.consumer_tags: Dict[str, str] = {}

        self.mq_exchange = self.settings.MQ_EXCHANGE.format(
            ENV=self.settings.ENV
        )
        self.dlx_exchange = self.settings.MQ_DLX_EXCHANGE.format(
            ENV=self.settings.ENV
        )
        self.dlq_queue_name = self.settings.MQ_DLQ_QUEUE.format(
            ENV=self.settings.ENV
        )

        self.global_semaphore = asyncio.Semaphore(
            self.settings.GLOBAL_CONCURRENCY
        )
        self.client_semaphores: Dict[str, asyncio.Semaphore] = {
            name: asyncio.Semaphore(limit)
            for name, limit in self.settings.CLIENT_WEIGHTS.items()
        }
        self.webhook_sender = WebhookSender(self.settings)
        self._stop_event = asyncio.Event()

    async def connect(self) -> None:
        self.connection = await aio_pika.connect_robust(
            host=self.settings.MQ_HOST,
            port=self.settings.MQ_PORT,
            login=self.settings.MQ_USER,
            password=self.settings.MQ_PASS,
            virtualhost=self.settings.MQ_VIRTUAL_HOST,
        )

        self.channel = await self.connection.channel(
            publisher_confirms=True
        )

        await self.channel.set_qos(
            prefetch_count=self.settings.GLOBAL_CONCURRENCY
        )

        await self.channel.declare_exchange(
            self.mq_exchange,
            aio_pika.ExchangeType.DIRECT,
            durable=True,
        )

        await self.channel.declare_exchange(
            self.dlx_exchange,
            aio_pika.ExchangeType.DIRECT,
            durable=True,
        )

        self.dlq_queue = await self.channel.declare_queue(
            name=self.dlq_queue_name,
            durable=True,
            arguments={
                "x-queue-type": "quorum",
                "x-message-ttl": 604800000,
            },
        )
        await self.dlq_queue.bind(
            self.dlx_exchange,
            routing_key=self.dlq_queue_name,
        )

        for client_name in self.settings.CLIENT_WEIGHTS:
            queue_name = (
                f"{self.settings.ENV}_knowledge_base_tasks."
                f"{client_name}"
            )

            queue = await self.channel.declare_queue(
                name=queue_name,
                durable=True,
                arguments={
                    "x-max-priority": 10,
                    "x-message-ttl": 3600000,
                    # "x-dead-letter-exchange": self.dlx_exchange,
                    # "x-dead-letter-routing-key": self.dlq_queue_name,
                },
            )
            await queue.bind(
                self.mq_exchange,
                routing_key=client_name,
            )
            self.queues[client_name] = queue

        for client_name in self.settings.CLIENT_WEIGHTS:
            for retry_count in range(1, self.settings.MAX_RETRIES + 1):
                delay = self._get_retry_delay(retry_count)
                queue_name = self._get_retry_queue_name(
                    client_name,
                    retry_count,
                )

                retry_queue = await self.channel.declare_queue(
                    name=queue_name,
                    durable=True,
                    arguments={
                        "x-message-ttl": delay * 1000,
                        "x-dead-letter-exchange": self.mq_exchange,
                        "x-dead-letter-routing-key": client_name,
                    },
                )
                self.retry_queues[
                    (client_name, retry_count)
                ] = retry_queue

    def _get_retry_delay(self, retry_count: int) -> int:
        return min(
            self.settings.RETRY_DELAY_SECONDS
            * (2 ** (retry_count - 1)),
            3600,
        )

    def _get_retry_queue_name(
        self,
        client_name: str,
        retry_count: int,
    ) -> str:
        return (
            f"{self.settings.ENV}_knowledge_base_tasks."
            f"{client_name}.retry.{retry_count}"
        )

    def _get_retry_count(self, msg: aio_pika.IncomingMessage) -> int:
        value = (msg.headers or {}).get("x-retry-count", 0)

        try:
            return int(value)
        except (TypeError, ValueError):
            log.warning(
                "invalid_retry_count",
                message_id=msg.message_id,
                value=value,
            )
            return 0

    def _channel_ok(self, msg: aio_pika.IncomingMessage) -> bool:
        if self.connection is None or self.connection.is_closed:
            return False
        if self.channel is None or self.channel.is_closed:
            return False
        try:
            if msg.channel.is_closed:
                return False
        except ChannelInvalidStateError:
            return False
        return True

    async def _safe_ack(self, msg: aio_pika.IncomingMessage) -> None:
        if msg.processed:
            return
        if not self._channel_ok(msg):
            log.debug(
                "ack_skipped_channel_unavailable",
                message_id=msg.message_id,
            )
            return
        try:
            await msg.ack()
        except ChannelInvalidStateError:
            log.debug(
                "ack_skipped_channel_closed",
                message_id=msg.message_id,
            )

    async def _safe_nack(
        self,
        msg: aio_pika.IncomingMessage,
        requeue: bool = True,
    ) -> None:
        if msg.processed:
            return
        if not self._channel_ok(msg):
            log.debug(
                "nack_skipped_channel_unavailable",
                message_id=msg.message_id,
            )
            return
        try:
            await msg.nack(requeue=requeue)
        except ChannelInvalidStateError:
            log.debug(
                "nack_skipped_channel_closed",
                message_id=msg.message_id,
            )

    def _get_client_routing_key(
        self,
        msg: aio_pika.IncomingMessage,
    ) -> str:
        headers = msg.headers or {}

        client_routing_key = headers.get("x-original-routing-key")
        if client_routing_key:
            if client_routing_key not in self.settings.CLIENT_WEIGHTS:
                raise ValueError(
                    f"Unknown client routing key: {client_routing_key!r}"
                )
            return client_routing_key

        routing_key = msg.routing_key or ""

        # Сообщения, опубликованные через default exchange,
        # получают в routing_key имя очереди.
        queue_prefix = f"{self.settings.ENV}_knowledge_base_tasks."

        if routing_key.startswith(queue_prefix):
            client_routing_key = routing_key[len(queue_prefix):]

            # Для retry queue:
            # dev_knowledge_base_tasks.test.retry.1
            retry_marker = ".retry."
            if retry_marker in client_routing_key:
                client_routing_key = client_routing_key.split(
                    retry_marker,
                    1,
                )[0]

            if client_routing_key in self.settings.CLIENT_WEIGHTS:
                return client_routing_key

        if routing_key in self.settings.CLIENT_WEIGHTS:
            return routing_key

        raise ValueError(
            f"Unable to determine client routing key: "
            f"routing_key={routing_key!r}, "
            f"headers={headers!r}"
        )

    async def _retry_with_delay(
        self,
        msg: aio_pika.IncomingMessage,
        retry_count: int,
    ) -> None:
        if self.channel is None or self.channel.is_closed:
            log.warning(
                "retry_publish_skipped_channel_closed",
                message_id=msg.message_id,
                retry_count=retry_count,
            )
            return

        client_routing_key = self._get_client_routing_key(msg)
        retry_queue_key = (client_routing_key, retry_count)

        if retry_queue_key not in self.retry_queues:
            raise ValueError(
                f"Retry queue is not configured: "
                f"client={client_routing_key!r}, "
                f"retry_count={retry_count}"
            )

        delay = self._get_retry_delay(retry_count)

        headers = dict(msg.headers or {})
        headers.update(
            {
                "x-retry-count": retry_count,
                "x-last-retry": datetime.now(
                    timezone.utc
                ).isoformat(),
                "x-original-routing-key": client_routing_key,
            }
        )

        retry_message = aio_pika.Message(
            body=msg.body,
            headers=headers,
            message_id=msg.message_id,
            correlation_id=msg.correlation_id,
            content_type=msg.content_type,
            delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
        )

        try:
            await self.channel.default_exchange.publish(
                retry_message,
                routing_key=self._get_retry_queue_name(
                    client_routing_key,
                    retry_count,
                ),
                mandatory=True,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log.exception(
                "retry_publish_failed",
                message_id=msg.message_id,
                retry_count=retry_count,
                error=str(exc),
            )
            return

        log.info(
            "message_retry_scheduled",
            message_id=msg.message_id,
            retry_count=retry_count,
            delay=delay,
            routing_key=client_routing_key,
        )

    async def _send_to_dlq(
        self,
        msg: aio_pika.IncomingMessage,
        error: str,
        error_type: str = "permanent",
    ) -> None:
        if self.channel is None or self.channel.is_closed:
            log.warning(
                "dlq_publish_skipped_channel_closed",
                message_id=msg.message_id,
                error=error,
            )
            return

        headers = dict(msg.headers or {})
        headers.update(
            {
                "x-dlq-reason": error,
                "x-dlq-timestamp": datetime.now(
                    timezone.utc
                ).isoformat(),
                "x-dlq-original-routing-key": self._get_client_routing_key(msg),
                "x-dlq-retry-count": self._get_retry_count(msg),
                "x-dlq-error-type": error_type,
            }
        )

        dlq_message = aio_pika.Message(
            body=msg.body,
            headers=headers,
            message_id=msg.message_id,
            correlation_id=msg.correlation_id,
            content_type=msg.content_type,
            delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
        )

        try:
            await self.channel.default_exchange.publish(
                dlq_message,
                routing_key=self.dlq_queue_name,
                mandatory=True,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log.exception(
                "dlq_publish_failed",
                message_id=msg.message_id,
                error=error,
                error_type=error_type,
                publish_error=str(exc),
            )
            return

        log.warning(
            "message_sent_to_dlq",
            message_id=msg.message_id,
            error=error,
            error_type=error_type,
            routing_key=msg.routing_key,
        )

    async def _process_one_message(
        self,
        msg: aio_pika.IncomingMessage,
    ) -> None:
        payload: Optional[Payload] = None
        retry_count = self._get_retry_count(msg)

        try:
            payload = Payload.model_validate_json(
                msg.body.decode()
            )

            pipeline = KnowledgePipeline(
                settings=self.settings
            )

            await asyncio.wait_for(
                pipeline.run(payload=payload),
                timeout=self.settings.MESSAGE_PROCESS_TIMEOUT_SECONDS,
            )

            await self._safe_ack(msg)

            log.info(
                "message_processed",
                message_id=msg.message_id,
                job_id=payload.job_id,
                retry_count=retry_count,
            )

        except ValidationError as exc:
            error_msg = (
                f"Validation_error: Неверный формат данных: {exc}"
            )

            await self._send_error_response(
                payload,
                error_msg,
            )

            await self._send_to_dlq(
                msg,
                error_msg,
                error_type="permanent",
            )

            await self._safe_ack(msg)

        except (FileNotFoundError, ValueError) as exc:
            error_msg = f"invalid_data: {exc}"

            await self._send_error_response(
                payload,
                error_msg,
            )

            await self._send_to_dlq(
                msg,
                error_msg,
                error_type="permanent",
            )

            await self._safe_ack(msg)

        except (AttributeError, TypeError, KeyError) as exc:
            error_msg = (
                f"invalid_data: Некорректная структура данных: {exc}"
            )

            log.error(
                "process_message_invalid_data",
                error=error_msg,
                message_id=msg.message_id,
                retry_count=retry_count,
            )

            await self._send_error_response(
                payload,
                error_msg,
            )

            await self._send_to_dlq(
                msg,
                error_msg,
                error_type="permanent",
            )

            await self._safe_ack(msg)

        except (ConnectionError, TimeoutError, asyncio.TimeoutError) as exc:
            if retry_count >= self.settings.MAX_RETRIES:
                error_msg = f"Превышено число попыток: {exc}"

                await self._send_error_response(
                    payload,
                    error_msg,
                )

                await self._send_to_dlq(
                    msg,
                    error_msg,
                    error_type="temporary",
                )

                await self._safe_ack(msg)
                return

            next_retry_count = retry_count + 1

            log.warning(
                "temporary_error_retry",
                error=str(exc),
                retry_count=retry_count,
                next_retry_count=next_retry_count,
                message_id=msg.message_id,
            )

            await self._retry_with_delay(
                msg,
                next_retry_count,
            )

            await self._safe_ack(msg)

        except Exception as exc:
            error_msg = (
                "Проблема обработки данных. "
                "Обратитесь в техническую поддержку"
            )

            log.exception(
                "process_message_unexpected_error",
                message_id=msg.message_id,
                retry_count=retry_count,
            )

            if retry_count >= self.settings.MAX_RETRIES:
                await self._send_error_response(
                    payload,
                    error_msg,
                )

                await self._send_to_dlq(
                    msg,
                    error_msg,
                    error_type="internal",
                )

                await self._safe_ack(msg)
                return

            next_retry_count = retry_count + 1

            await self._retry_with_delay(
                msg,
                next_retry_count,
            )

            await self._safe_ack(msg)


    async def _send_error_response(
        self,
        payload: Optional[Payload],
        error_message: str,
    ) -> None:
        if payload is None:
            log.warning(
                "error_response_skipped",
                reason="payload_is_none",
            )
            return

        return_payload = ReturnPayload(
            id=payload.id,
            job_id=payload.job_id,
            text_out="",
            kb_type=(
                payload.kb_type
                if payload.kb_type in ["vector", "context"]
                else "context"
            ),
            cost={},
            status="error",
            reason=error_message,
        )

        result = self.webhook_sender.send_to_webhook(
            payload.external_url,
            data=return_payload.model_dump_json(),
        )

        if asyncio.iscoroutine(result):
            await result

    async def _worker(self, client_name: str) -> None:
        queue = self.queues[client_name]

        async def _callback(
            msg: aio_pika.IncomingMessage,
        ) -> None:
            task = asyncio.current_task()
            if task is not None:
                self.pending_tasks.add(task)
            try:
                async with self.global_semaphore:
                    async with self.client_semaphores[client_name]:
                        try:
                            await self._process_one_message(msg)
                        except asyncio.CancelledError:
                            await self._safe_nack(
                                msg,
                                requeue=True,
                            )
                            raise
                        except Exception:
                            log.exception(
                                "message_processing_failed",
                                message_id=msg.message_id,
                            )
                            await self._safe_nack(
                                msg,
                                requeue=True,
                            )
            finally:
                if task is not None:
                    self.pending_tasks.discard(task)

        consumer_tag = await queue.consume(
            _callback,
            no_ack=False,
        )
        self.consumer_tags[client_name] = consumer_tag

        await self._stop_event.wait()

    async def start(self) -> None:
        workers = [
            asyncio.create_task(
                self._worker(name)
            )
            for name in self.settings.CLIENT_WEIGHTS
        ]

        try:
            await asyncio.gather(*workers)
        finally:
            for worker in workers:
                if not worker.done():
                    worker.cancel()

            await asyncio.gather(
                *workers,
                return_exceptions=True,
            )

    async def close(self) -> None:
        self._stop_event.set()

        for client_name, queue in self.queues.items():
            consumer_tag = self.consumer_tags.get(client_name)
            if consumer_tag is None:
                continue
            try:
                await queue.cancel(consumer_tag)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                log.warning(
                    "consumer_cancel_failed",
                    consumer_tag=consumer_tag,
                    error=str(exc),
                )

        if self.pending_tasks:
            done, pending = await asyncio.wait(
                set(self.pending_tasks),
                timeout=self.settings.SHUTDOWN_DRAIN_TIMEOUT_SECONDS,
            )
            if pending:
                for task in pending:
                    task.cancel()
                await asyncio.gather(
                    *pending,
                    return_exceptions=True,
                )
            log.info(
                "in_flight_messages_drained",
                done=len(done),
                cancelled=len(pending),
            )

        if self.channel and not self.channel.is_closed:
            await self.channel.close()

        if self.connection and not self.connection.is_closed:
            await self.connection.close()

        log.info(
            "rmq_connection_closed"
        )


async def main() -> None:
    dispatcher = Dispatcher()
    await dispatcher.connect()

    loop = asyncio.get_running_loop()
    stop = asyncio.Event()

    if sys.platform != "win32":
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(
                    sig,
                    stop.set,
                )
            except NotImplementedError:
                pass

    run_task = asyncio.create_task(
        dispatcher.start()
    )

    try:
        await stop.wait()
    except KeyboardInterrupt:
        log.info("shutdown_requested")
    finally:
        await dispatcher.close()

        if not run_task.done():
            run_task.cancel()

        await asyncio.gather(
            run_task,
            return_exceptions=True,
        )


if __name__ == "__main__":
    asyncio.run(main())