import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

from aiormq.exceptions import ChannelInvalidStateError

from consumer_dispatcher import Dispatcher


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


def make_dispatcher(channel, connection):
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