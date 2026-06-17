"""Resilience: retries and timeouts at the right layer.

The naive instinct is to wrap `Runner.run(...)` in a retry loop. **Don't** — an agent run
has side effects (it writes files, runs code, loads tables). Re-running it after a partial
failure can duplicate work or corrupt state.

The correct place to retry is the **individual HTTP request** to the model API. Transient
failures there (429 rate limits, timeouts, 5xx, dropped connections) should be retried with
exponential backoff *without* re-running the agent loop. The OpenAI client does exactly this
for us — we just configure it and register it as the SDK's default client.

So "resilience" here is one well-placed call, plus the understanding of *why* it goes there.
"""

from __future__ import annotations

import logging
import os

from agents import set_default_openai_client
from openai import AsyncOpenAI

logger = logging.getLogger("data_agent")


def configure_resilience() -> None:
    """Install a model client that retries transient API failures with backoff.

    `max_retries` retries 408/409/429/5xx and connection errors with exponential backoff,
    transparently — the agent loop never sees the blip. `timeout` bounds a single request so
    a hung call can't stall a turn forever. Both are overridable via env.
    """
    max_retries = int(os.getenv("OPENAI_MAX_RETRIES", "5"))
    timeout = float(os.getenv("OPENAI_TIMEOUT", "60"))
    client = AsyncOpenAI(max_retries=max_retries, timeout=timeout)
    set_default_openai_client(client)
    logger.info("resilience configured max_retries=%s timeout=%s", max_retries, timeout)
