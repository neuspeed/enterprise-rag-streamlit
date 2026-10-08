"""Which model failures are worth trying again somewhere else.

Used in two places that must agree: the RabbitMQ dispatcher, which decides
whether a job is temporary or permanent, and :mod:`know_everything_ai.utils.llm_utils`,
which decides whether a second provider is worth trying. Keeping one list means
a 429 is never "temporary" to the queue and "give up" to the failover.
"""

from __future__ import annotations

import asyncio

import httpx
import openai

#: Failures worth retrying rather than reporting to the buyer as a broken
#: document. httpx errors do not inherit from the builtin ConnectionError, and
#: the OpenAI SDK's status errors do not either, so without naming them a rate
#: limit or a 502 lands in the generic handler: logged as an internal bug and
#: dead-lettered as "internal" instead of "temporary".
RETRYABLE_ERRORS: tuple[type[BaseException], ...] = (
    ConnectionError,
    TimeoutError,
    asyncio.TimeoutError,
    httpx.TransportError,
    openai.APIConnectionError,
    openai.APITimeoutError,
    openai.RateLimitError,
    openai.InternalServerError,
)
