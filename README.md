# Swarm Fraud Detection

A decentralized multi-agent system for real-time financial fraud detection on mobile transactions. Four specialized agents analyze transactions in parallel, coordinate via Redis Pub/Sub, and converge on a weighted consensus decision powered by a local LLM (Ollama) for regulatory-grade explainability.

---

## Architecture Overview

```
Transaction (JSON)
      ↓
FastAPI Gateway (:8000)
      ↓
Redis Pub/Sub — channel: tx_events
      ↓ (broadcast)
┌─────────────────────────────────────┐
│  Agent A      Agent B               │
│  Amount       Frequency             │
│  IsolForest   Z-score rolling       │
│                                     │
│  Agent C      Agent D               │
│  Geolocation  Device Profile        │
│  Haversine    OneClassSVM           │
└─────────────────────────────────────┘
      ↓ (scores → channel: agent_scores)
AutoGen GroupChat — weighted vote
S_final = 0.20·A + 0.25·B + 0.30·C + 0.25·D
      ↓
Ollama (llama3 local) — justification
      ↓
FRAUD (S_final ≥ 0.70) | LEGITIMATE
      ↓
FastAPI Response (JSON)
```

---

## Agents

| Agent | Feature | Model | Threshold |
|---|---|---|---|
| `agent_amount` | Transaction amount | IsolationForest | 0.65 |
| `agent_frequency` | Transactions per hour per user | Z-score rolling window | 0.70 |
| `agent_geolocation` | Displacement velocity (km/h) | Haversine + rule | 0.80 |
| `agent_device` | Device ID novelty | OneClassSVM | 0.75 |

Each agent runs in an isolated Docker container, subscribes to `tx_events`, publishes an `AgentScore` to `agent_scores`, and has no knowledge of other agents' outputs.

---

## Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11 |
| Data validation | Pydantic v2 |
| Agent coordination | pyautogen 0.2.x (GroupChat) |
| ML models | scikit-learn 1.4+ |
| Message bus | Redis 5.x (asyncio) |
| Gateway | FastAPI + Uvicorn |
| LLM | Ollama (llama3 / mistral — local) |
| Containerization | Docker + Docker Compose v2 |
| Testing | pytest, pytest-asyncio, pytest-cov |
| Monitoring | Prometheus + Grafana |
| Dataset | PaySim (mobile transaction simulation) |

---

## Project Structure

```
swarm_fraud_detection/
├── docker-compose.yml
├── .env.example
├── requirements.txt
│
├── shared/
│   ├── transaction_schema.py       # Pydantic v2: TransactionPayload
│   └── agent_score_schema.py       # Pydantic v2: AgentScore
│
├── agents/
│   ├── base_agent.py               # Abstract BaseSwarmAgent
│   ├── Dockerfile.agent
│   ├── agent_amount/
│   │   └── amount_anomaly_agent.py
│   ├── agent_frequency/
│   │   └── frequency_zscore_agent.py
│   ├── agent_geolocation/
│   │   └── geolocation_velocity_agent.py
│   └── agent_device/
│       └── device_profile_agent.py
│
├── aggregator/
│   ├── Dockerfile
│   └── autogen_groupchat_aggregator.py
│
├── gateway/
│   ├── Dockerfile
│   └── fastapi_gateway.py
│
├── monitoring/
│   ├── prometheus.yml
│   └── grafana_dashboard.json
│
└── tests/
    ├── conftest.py
    ├── unit/
    │   ├── test_amount_agent.py
    │   ├── test_frequency_agent.py
    │   ├── test_geolocation_agent.py
    │   ├── test_device_agent.py
    │   └── test_aggregator_consensus.py
    ├── integration/
    │   ├── test_redis_pubsub_pipeline.py
    │   └── test_swarm_end_to_end.py
    └── performance/
        └── test_latency_p99.py
```

---

## Prerequisites

- Docker Desktop (or Docker Engine + Compose v2)
- Ollama installed locally with at least one model pulled

```bash
ollama pull llama3
```

- PaySim dataset (`paysim.csv`) placed at the path defined by `PAYSIM_CSV_PATH`

---

## Environment Variables

Copy `.env.example` to `.env` and set values before starting:

```bash
cp .env.example .env
```

| Variable | Default | Description |
|---|---|---|
| `PAYSIM_CSV_PATH` | `/data/paysim.csv` | Path to PaySim dataset inside containers |
| `OLLAMA_MODEL` | `llama3` | Ollama model name |
| `REDIS_URL` | `redis://redis:6379` | Redis connection string |
| `GATEWAY_HOST` | `0.0.0.0` | FastAPI bind host |
| `GATEWAY_PORT` | `8000` | FastAPI bind port |
| `THRESHOLD_AMOUNT` | `0.65` | Agent A fraud threshold |
| `THRESHOLD_FREQUENCY` | `0.70` | Agent B fraud threshold |
| `THRESHOLD_GEOLOCATION` | `0.80` | Agent C fraud threshold |
| `THRESHOLD_DEVICE` | `0.75` | Agent D fraud threshold |
| `FRAUD_DECISION_THRESHOLD` | `0.70` | Final S_final cutoff |

---

## Running the System

```bash
docker compose up --build
```

All services start in dependency order: Redis → Agents → Aggregator → Gateway → Prometheus → Grafana.

### Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/analyze` | Submit a transaction, receive fraud decision |
| `WS` | `/stream` | WebSocket stream for real-time decisions |
| `GET` | `/health` | Redis + Ollama connectivity check |
| `GET` | `/metrics` | Prometheus exposition |

### Example request

```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "transaction_id": "550e8400-e29b-41d4-a716-446655440000",
    "user_id": "user_42",
    "amount": 9999.99,
    "merchant_id": "merch_01",
    "device_id": "dev_xyz",
    "latitude": 48.8566,
    "longitude": 2.3522,
    "timestamp": "2025-06-01T14:30:00Z"
  }'
```

### Example response

```json
{
  "transaction_id": "550e8400-e29b-41d4-a716-446655440000",
  "S_final": 0.83,
  "decision": "FRAUD",
  "active_flags": ["agent_amount", "agent_geolocation"],
  "justification": "High transaction amount combined with impossible geographic displacement triggered two agents above threshold.",
  "latency_ms": 212.4
}
```

---

## Running Tests

```bash
pip install -r requirements.txt

pytest tests/unit/ -v --cov=agents --cov=aggregator --cov-report=term-missing
pytest tests/integration/ -v
pytest tests/performance/ -v
```

Coverage target: **80% minimum** across `agents/` and `aggregator/`.

### Test categories

| Suite | Scope | Redis required |
|---|---|---|
| `tests/unit/` | Agent logic, consensus math, schema validation | No |
| `tests/integration/` | Pub/Sub pipeline, end-to-end flow (Ollama mocked) | Yes |
| `tests/performance/` | p99 latency < 500 ms, throughput ≥ 50 req/s | Yes |

---

## Monitoring

| Service | URL | Credentials |
|---|---|---|
| Grafana | http://localhost:3000 | admin / admin |
| Prometheus | http://localhost:9090 | — |

Grafana ships with a pre-built dashboard (`monitoring/grafana_dashboard.json`) exposing:

- Request rate (req/s)
- Latency p50 / p95 / p99 (ms)
- Fraud rate (%)
- Per-agent score distribution
- Redis pub/sub message lag
- Active agent count

---

## Performance Targets

| Metric | Target |
|---|---|
| Latency p99 | < 500 ms |
| Throughput | ≥ 50 req/s on local Docker |
| Fault tolerance | Decision produced with 3/4 agents active |
| Coverage | ≥ 80% on agents/ and aggregator/ |

---

## Decision Logic

```
weights = {
    agent_amount:      0.20,
    agent_frequency:   0.25,
    agent_geolocation: 0.30,
    agent_device:      0.25
}

S_final = Σ weights[i] · score[i]

if S_final >= 0.70 → FRAUD
else               → LEGITIMATE
```

If an agent times out (> 2 s), it is excluded from the vote and the decision is flagged as partial. If Ollama is unreachable, `justification` returns `"LLM unavailable"` and the pipeline continues uninterrupted.

---

## Dataset

The system trains all ML models at container startup using `https://www.kaggle.com/datasets/ealaxi/paysim1` , a synthetic mobile money transaction dataset with 6.3 million rows and ground-truth fraud labels. No data is persisted between container restarts. Set `PAYSIM_CSV_PATH` to the mounted CSV path.

---

## Regulatory Notes

Ollama-generated justifications are designed to satisfy post-hoc explainability requirements under RGPD (right to explanation) and MiFID II audit trails. Each response includes the contributing agent IDs and their individual scores, enabling full traceability of the collective decision.
