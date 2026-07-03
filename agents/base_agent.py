from __future__ import annotations
import asyncio
from abc import ABC, abstractmethod
from datetime import datetime, timezone
import redis.asyncio as redis
import structlog
from pydantic import ValidationError
from shared.agent_score_schema import AgentScore
from shared.transaction_schema import TransactionPayload

logger = structlog.get_logger()
class BaseSwarmAgent(ABC):
    def __init__(
        self,
        agent_id: str,
        redis_url: str,
        input_channel: str,
        output_channel: str,
        threshold: float,
        feature_used: str,
        max_retries: int = 5,
        max_concurrency: int = 32,
    ) -> None:
        self.agent_id = agent_id
        self.redis_url = redis_url
        self.input_channel = input_channel
        self.output_channel = output_channel
        self.threshold = threshold
        self.feature_used = feature_used
        self.max_retries = max_retries
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._shutdown_event = asyncio.Event()

    @abstractmethod
    async def compute_score(self, tx: TransactionPayload) -> float: ...

    async def _connect(self) -> redis.Redis:
        attempt = 0
        delay_seconds = 1.0
        while attempt < self.max_retries:
            try:
                client = redis.from_url(self.redis_url, decode_responses=True)
                await client.ping()
                logger.info("redis_connected", agent_id=self.agent_id, attempt=attempt)
                return client
            except (redis.ConnectionError, redis.TimeoutError) as exc:
                attempt += 1
                logger.warning(
                    "redis_connection_failed",
                    agent_id=self.agent_id,
                    attempt=attempt,
                    error=str(exc),
                )
                if attempt >= self.max_retries:
                    raise
                await asyncio.sleep(delay_seconds)
                delay_seconds *= 2
        raise RuntimeError(f"redis connection exhausted for agent {self.agent_id}")

    async def _publish_score(self, client: redis.Redis, score: AgentScore) -> None:
        await client.publish(self.output_channel, score.model_dump_json())

    async def _handle_message(self, client: redis.Redis, raw_payload: str) -> None:
        async with self._semaphore:
            try:
                tx = TransactionPayload.model_validate_json(raw_payload)
            except ValidationError as exc:
                logger.error(
                    "payload_validation_error", agent_id=self.agent_id, error=str(exc)
                )
                return

            try:
                score_value = await self.compute_score(tx)
            except Exception as exc:
                logger.error(
                    "compute_score_error",
                    agent_id=self.agent_id,
                    transaction_id=tx.transaction_id,
                    error=str(exc),
                )
                return

            agent_score = AgentScore(
                agent_id=self.agent_id,
                transaction_id=tx.transaction_id,
                score=score_value,
                flag=score_value >= self.threshold,
                threshold=self.threshold,
                feature_used=self.feature_used,
                computed_at=datetime.now(timezone.utc),
            )
            await self._publish_score(client, agent_score)
            logger.info(
                "score_computed",
                agent_id=self.agent_id,
                transaction_id=tx.transaction_id,
                score=score_value,
                flag=agent_score.flag,
            )

    def shutdown(self) -> None:
        self._shutdown_event.set()

    async def run(self) -> None:
        client = await self._connect()
        pubsub = client.pubsub()
        await pubsub.subscribe(self.input_channel)
        logger.info("agent_started", agent_id=self.agent_id, channel=self.input_channel)

        in_flight: set[asyncio.Task[None]] = set()

        try:
            while not self._shutdown_event.is_set():
                try:
                    message = await pubsub.get_message(
                        ignore_subscribe_messages=True, timeout=5.0
                    )
                except (redis.ConnectionError, redis.TimeoutError) as exc:
                    logger.warning(
                        "redis_message_loop_error", agent_id=self.agent_id, error=str(exc)
                    )
                    await pubsub.aclose()
                    client = await self._connect()
                    pubsub = client.pubsub()
                    await pubsub.subscribe(self.input_channel)
                    continue

                if message is None:
                    continue

                task = asyncio.create_task(self._handle_message(client, message["data"]))
                in_flight.add(task)
                task.add_done_callback(in_flight.discard)
        finally:
            if in_flight:
                await asyncio.gather(*in_flight, return_exceptions=True)
            await pubsub.aclose()
            await client.aclose()
            logger.info("agent_stopped", agent_id=self.agent_id)