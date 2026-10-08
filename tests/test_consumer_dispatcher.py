import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

from aiormq.exceptions import ChannelInvalidStateError

from know_everything_ai.transports.rabbitmq import Dispatcher


class FakeChannel:
    def __init__(self, is_closed=False):
        self.is_closed = is_closed

    async def close(self):
        self.is_closed = True


class FakeConnection:
    def __init__(self, is_closed=False):
        self.is_closed = is_closed

    async def close(self):
        self.is_closed = True


class FakeIncomingMessage:
    def __init__(self, channel, processed=False):
        self._channel = channel
        self.processed = processed
        self.message_id = "msg-1"

        def _mark_processed(**kwargs):
            self.processed = True

        self.ack = AsyncMock(side_effect=_mark_processed)
        self.nack = AsyncMock(side_effect=_mark_processed)

    @property
    def channel(self):
        if self._channel is None or self._channel.is_closed:
            raise ChannelInvalidStateError()
        return self._channel


class FakePipeline:
    def __init__(self):
        self.closed = False

    async def aclose(self):
        self.closed = True


def make_dispatcher(channel, connection):
    """Build a Dispatcher without going through __init__.

    __init__ opens a real httpx pool and talks to no broker but still reads the
    environment; these tests are about ack/nack and drain behaviour, so the
    broker half is stubbed out and only the attributes under test are set.
    """
    dispatcher = Dispatcher.__new__(Dispatcher)
    dispatcher.settings = SimpleNamespace(
        SHUTDOWN_DRAIN_TIMEOUT_SECONDS=1,
        MESSAGE_PROCESS_TIMEOUT_SECONDS=10,
        MAX_RETRIES=3,
    )
    dispatcher.channel = channel
    dispatcher.connection = connection
    dispatcher.queues = {}
    dispatcher.retry_queues = {}
    dispatcher.dlq_queue = None
    dispatcher.dlq_queue_name = "test_dlq"
    dispatcher.consumer_tags = {}
    dispatcher.pending_tasks = set()
    dispatcher._stop_event = asyncio.Event()
    dispatcher.webhook_sender = None
    dispatcher.pipeline = FakePipeline()
    return dispatcher


def test_channel_ok_open():
    dispatcher = make_dispatcher(FakeChannel(), FakeConnection())
    msg = FakeIncomingMessage(FakeChannel())
    assert dispatcher._channel_ok(msg) is True


def test_channel_ok_closed_channel():
    dispatcher = make_dispatcher(FakeChannel(), FakeConnection())
    msg = FakeIncomingMessage(FakeChannel(is_closed=True))
    assert dispatcher._channel_ok(msg) is False


def test_channel_ok_closed_connection():
    dispatcher = make_dispatcher(FakeChannel(), FakeConnection(is_closed=True))
    msg = FakeIncomingMessage(FakeChannel())
    assert dispatcher._channel_ok(msg) is False


def test_safe_ack_happy_path():
    dispatcher = make_dispatcher(FakeChannel(), FakeConnection())
    msg = FakeIncomingMessage(FakeChannel())
    asyncio.run(dispatcher._safe_ack(msg))
    msg.ack.assert_awaited_once()
    assert msg.processed is True


def test_safe_ack_already_processed():
    dispatcher = make_dispatcher(FakeChannel(), FakeConnection())
    msg = FakeIncomingMessage(FakeChannel(), processed=True)
    asyncio.run(dispatcher._safe_ack(msg))
    msg.ack.assert_not_awaited()


def test_safe_ack_closed_channel_no_raise():
    dispatcher = make_dispatcher(FakeChannel(), FakeConnection())
    msg = FakeIncomingMessage(FakeChannel(is_closed=True))
    asyncio.run(dispatcher._safe_ack(msg))
    msg.ack.assert_not_awaited()


def test_safe_nack_happy_path():
    dispatcher = make_dispatcher(FakeChannel(), FakeConnection())
    msg = FakeIncomingMessage(FakeChannel())
    asyncio.run(dispatcher._safe_nack(msg, requeue=True))
    msg.nack.assert_awaited_once_with(requeue=True)
    assert msg.processed is True


def test_safe_nack_closed_channel_no_raise():
    """Regression: nack must not raise ChannelInvalidStateError when
    the channel is already closed (see aio_pika _task_done trace)."""
    dispatcher = make_dispatcher(FakeChannel(), FakeConnection())
    msg = FakeIncomingMessage(FakeChannel(is_closed=True))
    asyncio.run(dispatcher._safe_nack(msg, requeue=True))
    msg.nack.assert_not_awaited()


def test_safe_nack_already_processed():
    dispatcher = make_dispatcher(FakeChannel(), FakeConnection())
    msg = FakeIncomingMessage(FakeChannel(), processed=True)
    asyncio.run(dispatcher._safe_nack(msg, requeue=True))
    msg.nack.assert_not_awaited()


def test_retry_with_delay_skips_closed_channel():
    dispatcher = make_dispatcher(FakeChannel(is_closed=True), FakeConnection())
    msg = FakeIncomingMessage(FakeChannel(is_closed=True))
    asyncio.run(dispatcher._retry_with_delay(msg, 1))


def test_send_to_dlq_skips_closed_channel():
    dispatcher = make_dispatcher(FakeChannel(is_closed=True), FakeConnection())
    msg = FakeIncomingMessage(FakeChannel(is_closed=True))
    asyncio.run(dispatcher._send_to_dlq(msg, "boom"))


def test_close_drains_pending_tasks_and_closes_connection():
    dispatcher = make_dispatcher(FakeChannel(), FakeConnection())

    async def run():
        task = asyncio.create_task(asyncio.sleep(0.01))
        dispatcher.pending_tasks.add(task)
        await asyncio.sleep(0.02)
        await dispatcher.close()
        return task

    task = asyncio.run(run())
    assert task.done()
    assert dispatcher.channel.is_closed is True
    assert dispatcher.connection.is_closed is True


def test_close_releases_the_pipeline():
    """The shared pipeline owns an httpx pool; without this it leaks per worker.

    It used to be rebuilt per message and never closed at all, so a busy
    deployment accumulated a socket pool per delivery until the box ran out of
    file descriptors.
    """
    dispatcher = make_dispatcher(FakeChannel(), FakeConnection())
    asyncio.run(dispatcher.close())
    assert dispatcher.pipeline.closed is True


def test_error_webhook_uses_the_async_sender():
    """`_send_error_response` called a sync helper and awaited the result.

    The old call returned a coroutine that was conditionally awaited, so a
    failed webhook was silently dropped on the exact path where the buyer most
    needs to hear about the failure.
    """
    from know_everything_ai.schemas import Payload, ReturnPayload

    dispatcher = make_dispatcher(FakeChannel(), FakeConnection())
    sender = SimpleNamespace(send_payload=AsyncMock(return_value=False))
    dispatcher.webhook_sender = sender

    payload = Payload(
        id=1,
        job_id=2,
        kb_external_id="acme",
        kb_type="vector",
        data={"text": "x"},
        external_url="https://example.com/hook",
    )

    asyncio.run(dispatcher._send_error_response(payload, "boom"))

    sender.send_payload.assert_awaited_once()
    sent_url, sent_body = sender.send_payload.await_args[0]
    assert str(sent_url) == "https://example.com/hook"
    assert isinstance(sent_body, ReturnPayload)
    assert sent_body.status == "error"
    assert sent_body.reason == "boom"


def test_retryable_errors_cover_provider_outages():
    """A rate limit or a 502 is not a broken document.

    httpx transport errors do not subclass the builtin ConnectionError and the
    OpenAI status errors do not either, so a model provider answering 429 used
    to be dead-lettered as an internal fault with no retry.
    """
    import httpx
    import openai

    from know_everything_ai.transports.rabbitmq import RETRYABLE_ERRORS

    for exc in (
        ConnectionError(),
        TimeoutError(),
        asyncio.TimeoutError(),
        httpx.ConnectError("refused"),
        httpx.ReadTimeout("slow"),
        openai.APIConnectionError(request=httpx.Request("POST", "http://x")),
        openai.RateLimitError("429", response=httpx.Response(429, request=httpx.Request("POST", "http://x")), body=None),
        openai.InternalServerError("500", response=httpx.Response(500, request=httpx.Request("POST", "http://x")), body=None),
    ):
        assert isinstance(exc, RETRYABLE_ERRORS), type(exc)

    # Bad input must stay permanent: retrying a malformed payload forever helps
    # nobody.
    for exc in (ValueError(), FileNotFoundError()):
        assert not isinstance(exc, RETRYABLE_ERRORS)


def make_queue_dispatcher(message_process_timeout_seconds: int) -> Dispatcher:
    """Bare dispatcher that only carries what the queue arguments need."""
    dispatcher = Dispatcher.__new__(Dispatcher)
    dispatcher.settings = SimpleNamespace(
        MESSAGE_PROCESS_TIMEOUT_SECONDS=message_process_timeout_seconds,
    )
    dispatcher.mq_exchange = "dev_knowledge_base_tasks"
    dispatcher.dlx_exchange = "dlx.knowledge.dev"
    dispatcher.dlq_queue_name = "dev_knowledge_base_tasks.dlq"
    return dispatcher


def test_consumer_timeout_outlives_the_app_timeout():
    """The broker must never be the one that ends a job.

    Both used to sit at 30 minutes, but the broker's clock starts at delivery
    and the app's at dispatch, so the broker won: it closed the channel and
    requeued, and a second copy of the job started beside the first.
    """
    dispatcher = make_queue_dispatcher(message_process_timeout_seconds=1800)

    app_ceiling_ms = 1800 * 1000
    assert dispatcher._consumer_timeout_ms() > app_ceiling_ms


def test_task_queue_declares_a_consumer_timeout():
    dispatcher = make_queue_dispatcher(message_process_timeout_seconds=1800)
    arguments = dispatcher._task_queue_arguments()

    assert arguments["x-consumer-timeout"] == dispatcher._consumer_timeout_ms()
    # The rest of the contract must survive the change.
    assert arguments["x-max-priority"] == 10
    assert arguments["x-message-ttl"] == 3600000


def test_retry_queue_declares_a_consumer_timeout():
    """A delivery out of a retry queue runs the same job, so it needs the room."""
    dispatcher = make_queue_dispatcher(message_process_timeout_seconds=1800)
    arguments = dispatcher._retry_queue_arguments("acme", 60)

    assert arguments["x-consumer-timeout"] == dispatcher._consumer_timeout_ms()
    assert arguments["x-message-ttl"] == 60000
    assert arguments["x-dead-letter-routing-key"] == "acme"
