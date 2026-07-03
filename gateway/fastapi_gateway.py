from __future__ import annotations

import asyncio
import json
import os
import time
from contextlib import asynccontextmanager

import httpx
import redis.asyncio as redis
import structlog
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

from shared.transaction_schema import TransactionPayload

logger = structlog.get_logger()

REQUEST_COUNT = Counter("gateway_request_count", "Total analyzed transactions", ["route"])
LATENCY_HISTOGRAM = Histogram("gateway_latency_ms", "End to end decision latency in ms", ["route"])
FRAUD_RATE_GAUGE = Gauge("gateway_fraud_rate", "Rolling fraud rate percentage")

_fraud_window: list[bool] = []
_fraud_window_maxlen = 200


class GatewayState:
    def __init__(self) -> None:
        self.redis_client: redis.Redis | None = None
        self.redis_url = os.environ.get("REDIS_URL", "redis://redis:6379/0")
        self.tx_events_channel = os.environ.get("TX_EVENTS_CHANNEL", "tx_events")
        self.decisions_channel = os.environ.get("DECISIONS_CHANNEL", "decisions")
        self.ollama_base_url = os.environ.get(
            "OLLAMA_BASE_URL", "http://host.docker.internal:11434/v1"
        )
        self.decision_timeout_seconds = float(os.environ.get("DECISION_TIMEOUT_SECONDS", "3.0"))


state = GatewayState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    state.redis_client = redis.from_url(state.redis_url, decode_responses=True)
    yield
    if state.redis_client is not None:
        await state.redis_client.aclose()


app = FastAPI(title="Swarm Fraud Detection Gateway", lifespan=lifespan)


async def wait_for_decision(transaction_id: str, timeout_seconds: float) -> dict[str, object] | None:
    if state.redis_client is None:
        return None
    pubsub = state.redis_client.pubsub()
    await pubsub.subscribe(state.decisions_channel)
    deadline = time.monotonic() + timeout_seconds
    try:
        while time.monotonic() < deadline:
            remaining = deadline - time.monotonic()
            message = await pubsub.get_message(
                ignore_subscribe_messages=True, timeout=max(remaining, 0.01)
            )
            if message is None:
                continue
            payload = json.loads(message["data"])
            if payload.get("transaction_id") == transaction_id:
                return payload
        return None
    finally:
        await pubsub.unsubscribe(state.decisions_channel)
        await pubsub.aclose()


def _record_fraud_rate(decision: str) -> None:
    _fraud_window.append(decision == "FRAUD")
    if len(_fraud_window) > _fraud_window_maxlen:
        _fraud_window.pop(0)
    fraud_count = sum(_fraud_window)
    FRAUD_RATE_GAUGE.set((fraud_count / len(_fraud_window)) * 100.0)


@app.post("/analyze")
async def analyze_transaction(payload: TransactionPayload) -> dict[str, object]:
    if state.redis_client is None:
        raise HTTPException(status_code=503, detail="redis unavailable")
    REQUEST_COUNT.labels(route="analyze").inc()
    start_time = time.perf_counter()
    decision_task = asyncio.create_task(
        wait_for_decision(payload.transaction_id, state.decision_timeout_seconds)
    )
    await state.redis_client.publish(state.tx_events_channel, payload.model_dump_json())
    decision = await decision_task
    latency_ms = (time.perf_counter() - start_time) * 1000.0
    LATENCY_HISTOGRAM.labels(route="analyze").observe(latency_ms)
    if decision is None:
        raise HTTPException(status_code=504, detail="decision timeout")
    _record_fraud_rate(str(decision.get("decision")))
    return decision


@app.websocket("/stream")
async def stream_transactions(websocket: WebSocket) -> None:
    await websocket.accept()
    if state.redis_client is None:
        await websocket.close(code=1011)
        return
    try:
        while True:
            raw_payload = await websocket.receive_text()
            payload = TransactionPayload.model_validate_json(raw_payload)
            REQUEST_COUNT.labels(route="stream").inc()
            start_time = time.perf_counter()
            decision_task = asyncio.create_task(
                wait_for_decision(payload.transaction_id, state.decision_timeout_seconds)
            )
            await state.redis_client.publish(state.tx_events_channel, payload.model_dump_json())
            decision = await decision_task
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            LATENCY_HISTOGRAM.labels(route="stream").observe(latency_ms)
            if decision is None:
                await websocket.send_json({"transaction_id": payload.transaction_id, "error": "decision timeout"})
                continue
            _record_fraud_rate(str(decision.get("decision")))
            await websocket.send_json(decision)
    except WebSocketDisconnect:
        logger.info("websocket_disconnected")


@app.get("/health")
async def health_check() -> dict[str, object]:
    redis_healthy = False
    ollama_healthy = False
    if state.redis_client is not None:
        try:
            redis_healthy = await state.redis_client.ping()
        except redis.RedisError:
            redis_healthy = False
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            response = await client.get(f"{state.ollama_base_url}/models")
            ollama_healthy = response.status_code == 200
    except httpx.HTTPError:
        ollama_healthy = False
    return {"status": "ok", "redis": redis_healthy, "ollama": ollama_healthy}


@app.get("/metrics")
async def metrics() -> Response:
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


# Serve static files from frontend/build
app.mount("/static", StaticFiles(directory="/app/frontend/build/static"), name="static")


# Serve index.html for all other routes (SPA fallback)
@app.get("/{path:path}")
async def serve_index(path: str) -> FileResponse:
    index_path = "/app/frontend/build/index.html"
    if os.path.exists(index_path):
        return FileResponse(index_path)
    raise HTTPException(status_code=404, detail="Not found")