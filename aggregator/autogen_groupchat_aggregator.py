from __future__ import annotations
import asyncio
import json
import os
import time
from collections import OrderedDict
import redis.asyncio as redis
import structlog
from openai import AsyncOpenAI
from prometheus_client import start_http_server
from pydantic import BaseModel, Field, ValidationError
from shared.agent_score_schema import AgentScore

logger = structlog.get_logger()

FRAUD_THRESHOLD = 0.70

AGENT_WEIGHTS: dict[str, float] = {
    "agent_amount": 0.20,
    "agent_frequency": 0.25,
    "agent_geolocation": 0.30,
    "agent_device": 0.25,
}

EXPECTED_AGENT_IDS: frozenset[str] = frozenset(AGENT_WEIGHTS.keys())


def compute_weighted_score(scores: list[AgentScore]) -> float:
    weight_sum = sum(AGENT_WEIGHTS[score.agent_id] for score in scores)
    if weight_sum == 0.0:
        return 0.0
    weighted_sum = sum(AGENT_WEIGHTS[score.agent_id] * score.score for score in scores)
    return weighted_sum / weight_sum


class JustificationOutput(BaseModel):
    justification: str = Field(min_length=1, max_length=500)


class OllamaJustificationAgent:
    def __init__(self, model: str | None = None, base_url: str | None = None) -> None:
        self._model = model or os.environ.get("OLLAMA_MODEL", "llama3")
        self._base_url = base_url or os.environ.get("OLLAMA_BASE_URL", "http://host.docker.internal:11434/v1")
        self._client = AsyncOpenAI(
            base_url=self._base_url,
            api_key="ollama"
        )

    async def justify(self, scores: list[AgentScore], s_final: float, decision: str) -> str:
        score_lines = "\n".join(
            f"{score.agent_id}: score={score.score:.4f} flag={score.flag}" for score in scores
        )
        system_prompt = "You are a fraud analyst producing a two sentence natural language justification for a transaction decision given weighted agent scores."
        user_prompt = (
            f"Transaction decision: {decision}\n"
            f"Weighted fraud score: {s_final:.4f}\n"
            f"Individual agent scores:\n{score_lines}\n\n"
            "Please provide your justification as a JSON object with a single 'justification' field."
        )
        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                response_format={"type": "json_object"}
            )
            content = response.choices[0].message.content
            if content:
                parsed = json.loads(content)
                justification = parsed.get("justification", "LLM unavailable")
                if len(justification) <= 500:
                    return justification
            return "LLM unavailable"
        except Exception as exc:
            logger.warning("llm_justification_failed", error=str(exc))
            return "LLM unavailable"

    async def close(self) -> None:
        await self._client.close()


class TransactionCollector:
    def __init__(self, collection_timeout_seconds: float, max_finalized: int = 50_000) -> None:
        self._collection_timeout_seconds = collection_timeout_seconds
        self._max_finalized = max_finalized
        self._buffers: dict[str, dict[str, AgentScore]] = {}
        self._finalized: OrderedDict[str, None] = OrderedDict()
        self._on_finalize = None

    def set_finalize_callback(self, callback) -> None:
        self._on_finalize = callback

    def _mark_finalized(self, transaction_id: str) -> None:
        self._finalized[transaction_id] = None
        self._finalized.move_to_end(transaction_id)
        if len(self._finalized) > self._max_finalized:
            self._finalized.popitem(last=False)

    async def ingest(self, score: AgentScore) -> None:
        transaction_id = score.transaction_id
        if transaction_id in self._finalized:
            return
        if transaction_id not in self._buffers:
            self._buffers[transaction_id] = {}
            asyncio.create_task(self._finalize_after_timeout(transaction_id))
        self._buffers[transaction_id][score.agent_id] = score
        if EXPECTED_AGENT_IDS.issubset(self._buffers[transaction_id].keys()):
            await self._finalize(transaction_id)

    async def _finalize_after_timeout(self, transaction_id: str) -> None:
        await asyncio.sleep(self._collection_timeout_seconds)
        if transaction_id not in self._finalized:
            await self._finalize(transaction_id)

    async def _finalize(self, transaction_id: str) -> None:
        if transaction_id in self._finalized:
            return
        self._mark_finalized(transaction_id)
        collected_scores = list(self._buffers.pop(transaction_id, {}).values())
        if collected_scores and self._on_finalize is not None:
            asyncio.create_task(self._on_finalize(transaction_id, collected_scores))


class AggregatorService:
    def __init__(
        self,
        redis_url: str,
        agent_scores_channel: str,
        decisions_channel: str,
        collection_timeout_seconds: float = 2.0,
        max_retries: int = 5,
    ) -> None:
        self.redis_url = redis_url
        self.agent_scores_channel = agent_scores_channel
        self.decisions_channel = decisions_channel
        self.max_retries = max_retries
        self._collector = TransactionCollector(collection_timeout_seconds)
        self._collector.set_finalize_callback(self._finalize_decision)
        self._redis_client: redis.Redis | None = None
        self._justification_agent = OllamaJustificationAgent()

    async def _connect(self) -> redis.Redis:
        attempt = 0
        delay_seconds = 1.0
        while attempt < self.max_retries:
            try:
                client = redis.from_url(self.redis_url, decode_responses=True)
                await client.ping()
                logger.info("redis_connected", attempt=attempt)
                return client
            except (redis.ConnectionError, redis.TimeoutError) as exc:
                attempt += 1
                logger.warning("redis_connection_failed", attempt=attempt, error=str(exc))
                if attempt >= self.max_retries:
                    raise
                await asyncio.sleep(delay_seconds)
                delay_seconds *= 2
        raise RuntimeError("redis connection exhausted for aggregator")

    async def _finalize_decision(self, transaction_id: str, scores: list[AgentScore]) -> None:
        start_time = time.perf_counter()
        s_final = compute_weighted_score(scores)
        decision = "FRAUD" if s_final >= FRAUD_THRESHOLD else "LEGITIMATE"
        active_flags = [score.agent_id for score in scores if score.flag]

        justification = await self._justification_agent.justify(scores, s_final, decision)

        latency_ms = (time.perf_counter() - start_time) * 1000.0
        payload = {
            "transaction_id": transaction_id,
            "S_final": s_final,
            "decision": decision,
            "active_flags": active_flags,
            "agents_reporting": len(scores),
            "justification": justification,
            "latency_ms": latency_ms,
        }

        if self._redis_client is not None:
            try:
                await self._redis_client.publish(self.decisions_channel, json.dumps(payload, default=str))
            except (redis.ConnectionError, redis.TimeoutError) as exc:
                logger.error("decision_publish_failed", transaction_id=transaction_id, error=str(exc))
                return

        logger.info(
            "decision_published",
            transaction_id=transaction_id,
            decision=decision,
            s_final=s_final,
            agents_reporting=len(scores),
        )

    async def run(self) -> None:
        client = await self._connect()
        self._redis_client = client
        pubsub = client.pubsub()
        await pubsub.subscribe(self.agent_scores_channel)
        logger.info("aggregator_started", channel=self.agent_scores_channel)

        try:
            while True:
                try:
                    message = await pubsub.get_message(ignore_subscribe_messages=True, timeout=5.0)
                except (redis.ConnectionError, redis.TimeoutError) as exc:
                    logger.warning("redis_message_loop_error", error=str(exc))
                    await pubsub.aclose()
                    client = await self._connect()
                    self._redis_client = client
                    pubsub = client.pubsub()
                    await pubsub.subscribe(self.agent_scores_channel)
                    continue

                if message is None:
                    continue

                try:
                    score = AgentScore.model_validate_json(message["data"])
                except ValidationError as exc:
                    logger.error("payload_validation_error", error=str(exc))
                    continue

                await self._collector.ingest(score)
        finally:
            await pubsub.aclose()
            await client.aclose()
            await self._justification_agent.close()


def build_service_from_env() -> AggregatorService:
    return AggregatorService(
        redis_url=os.environ.get("REDIS_URL", "redis://redis:6379/0"),
        agent_scores_channel=os.environ.get("AGENT_SCORES_CHANNEL", "agent_scores"),
        decisions_channel=os.environ.get("DECISIONS_CHANNEL", "decisions"),
        collection_timeout_seconds=float(os.environ.get("COLLECTION_TIMEOUT_SECONDS", "2.0")),
    )


async def main() -> None:
    start_http_server(int(os.environ.get("METRICS_PORT", "8005")))
    service = build_service_from_env()
    await service.run()


if __name__ == "__main__":
    asyncio.run(main())