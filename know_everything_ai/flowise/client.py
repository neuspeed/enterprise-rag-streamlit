from collections.abc import AsyncGenerator

import httpx

from know_everything_ai.flowise.resources.document_store import DocumentStoreResource
from know_everything_ai.flowise.transport import FlowiseTransport


class IFileUpload:
    def __init__(self, data: str | None, type: str, name: str, mime: str):
        self.data = data
        self.type = type
        self.name = name
        self.mime = mime


class IMessage:
    def __init__(self, message: str, type: str, role: str | None = None, content: str | None = None):
        self.message = message
        self.type = type
        self.role = role
        self.content = content


class PredictionData:
    def __init__(
        self,
        chatflowId: str,
        question: str,
        overrideConfig: dict | None = None,
        chatId: str | None = None,
        streaming: bool | None = False,
        history: list[IMessage] | None = None,
        uploads: list[IFileUpload] | None = None,
    ):
        self.chatflowId = chatflowId
        self.question = question
        self.overrideConfig = overrideConfig
        self.chatId = chatId
        self.streaming = streaming
        self.history = history
        self.uploads = uploads

    def _payload(self) -> dict:
        return {
            "chatflowId": self.chatflowId,
            "question": self.question,
            "overrideConfig": self.overrideConfig,
            "chatId": self.chatId,
            "history": [msg.__dict__ for msg in (self.history or [])],
            "uploads": [upload.__dict__ for upload in (self.uploads or [])],
        }


class Flowise:
    """Entry point for the Flowise API.

    The transport is created once and reused. Pass ``transport`` to share a pool
    with the rest of the worker, or ``client`` to reuse an existing httpx client.
    """

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        *,
        transport: FlowiseTransport | None = None,
        client: httpx.AsyncClient | None = None,
        timeout: float = 60.0,
    ) -> None:
        self.base_url = base_url or "http://localhost:3000/api/v1"
        self.api_key = api_key or ""
        self.transport = transport or FlowiseTransport(
            self.base_url,
            self.api_key,
            timeout=timeout,
            client=client,
        )
        self.document_store = DocumentStoreResource(self.transport)

    async def create_prediction(
        self, data: PredictionData
    ) -> AsyncGenerator[str | dict, None]:
        """Run a chatflow prediction, streaming SSE events when available."""
        probe = await self.transport.request(
            "GET", f"/chatflows-streaming/{data.chatflowId}"
        )
        if probe.get("isStreaming", False) and data.streaming:
            payload = data._payload() | {"streaming": True}
            async for event in self.transport.stream_lines(
                f"/prediction/{data.chatflowId}", payload
            ):
                yield event
            return

        result = await self.transport.request(
            "POST", f"/prediction/{data.chatflowId}", json=data._payload()
        )
        yield result

    async def aclose(self) -> None:
        await self.transport.aclose()
