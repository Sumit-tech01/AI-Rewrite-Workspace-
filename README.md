# ML Reliability Platform for Text Intelligence

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-3.x-000000?logo=flask&logoColor=white)
![Postgres](https://img.shields.io/badge/Postgres-16+-4169E1?logo=postgresql&logoColor=white)
![Redis](https://img.shields.io/badge/Redis-7+-DC382D?logo=redis&logoColor=white)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)

Production-oriented NLP inference platform focused on reliability under real-world failure conditions.  
This project combines local ML inference with external provider orchestration, deterministic fallback behavior, quota governance, and request-level observability.

Elevator pitch: this platform is designed for graceful degradation, not just happy-path demos.

Who this is for:
- ML platform engineers building reliable inference APIs
- Backend engineers shipping production NLP services
- Product teams needing measurable reliability, quotas, and observability

## 1) Project Identity

This is not a tutorial sentiment app. It is a reliability-focused text intelligence backend built to demonstrate:
- bounded retries and controlled fallback escalation
- deterministic local fallback guarantees
- API key governance, rate controls, and abuse guardrails
- structured telemetry and drift-aware operations

## 2) Core Features (Reliability-Focused)

- Sentiment inference using local TF-IDF + Logistic Regression
- Emotion detection and token-level explainability output
- Rewrite orchestration: Google AI primary + deterministic local fallback
- Model agreement tracking (ML vs lexicon baseline)
- Drift monitoring design using PSI and Jensen-Shannon divergence
- Tier-based quota governance by API key
- Abuse signal capture and request fingerprinting design
- Structured logging, request IDs, and failure taxonomy
- Health checks for platform and dependencies
- Live analytics dashboard and usage insights

## 3) Architecture Overview

### System topology

- Web Service: Flask + Gunicorn API/UI layer
- Worker Service: RQ worker for async aggregation and drift jobs
- Postgres: event persistence and aggregate tables
- Redis: rate limiting, cache, queue backend, hot counters, circuit-breaker state
- Local model: in-process low-latency sentiment inference
- External provider: Google AI rewrite with local deterministic fallback

### ASCII architecture diagram

```text
                    +----------------------+
                    |   Web / API Clients  |
                    +----------+-----------+
                               |
                               v
                  +------------+-------------+
                  | Flask API (Gunicorn)     |
                  | request_id + auth + SLO  |
                  +---+-----------+----------+
                      |           |
          in-process  |           | external
            model     |           v
     +----------------+--+   +----------------------+
     | Local ML Inference |   | Google Rewrite API   |
     | Sentiment + Explain|   | Primary Provider     |
     +--------------------+   +----------+-----------+
                                          |
                                          | fail/timeout
                                          v
                                +----------------------+
                                | Deterministic Local  |
                                | Rewrite Fallback     |
                                +----------------------+

                      +-------------------+
                      |      Redis        |
                      | limiter/cache/queue|
                      +----+----------+---+
                           |          |
                           v          v
                    +------+----+  +--+----------------+
                    | RQ Worker |  | Circuit Breaker   |
                    | jobs      |  | + hot counters    |
                    +------+----+  +-------------------+
                           |
                           v
                     +-----+------+
                     |  Postgres  |
                     | events +   |
                     | aggregates |
                     +------------+
```

### Request lifecycle

1. Request enters Flask route with generated `request_id`.
2. API key auth, tier policy, and rate checks execute.
3. Service layer runs sentiment or rewrite orchestration.
4. Event is persisted to Postgres.
5. Non-critical analytics tasks are queued to worker.
6. Response returns with backward-compatible payload and observability metadata.

### Fallback lifecycle

1. Call external rewrite provider with strict timeout.
2. Retry only retryable failures with bounded exponential backoff.
3. Escalate to fallback provider model if configured.
4. If still unavailable, execute deterministic local rewrite.
5. Return success with `provider` and `fallback_used` metadata.

### Drift monitoring loop

1. Worker computes rolling sentiment/confidence distributions.
2. Compare current windows to baseline via PSI and JS divergence.
3. Persist drift snapshots and trend metrics.
4. Surface alerts on dashboard and operational logs.

### Quota enforcement flow

1. Resolve API key plan tier.
2. Increment Redis counters for key + endpoint + window.
3. Reject over-limit requests with quota headers.
4. Reconcile Redis counters into Postgres aggregates via worker.

## 4) Reliability Design

### Retry strategy

- Retry only transient categories (`timeout`, `503`, provider-unavailable)
- Bounded retries with jittered exponential backoff
- Hard timeout budget per provider call

### Circuit breaker states

- `closed`: normal traffic
- `open`: provider bypassed after repeated failures
- `half_open`: sampled probes to restore provider

### Deterministic fallback

- Local rewrite path does not depend on external services
- Predictable output behavior for same input + tone + style
- Guarantees service continuity under provider outage

### Latency budgeting

- Endpoint-level latency budget split by stages:
- `auth_ms`, `quota_ms`, `provider_ms`, `model_ms`, `db_ms`, `total_ms`

### Error taxonomy

- `validation_error`
- `auth_error`
- `quota_error`
- `provider_error`
- `fallback_error`
- `storage_error`
- `internal_error`

### SLO thinking

- Availability target: 99.9%
- p95 latency target per endpoint family
- Fallback recovery rate target for provider outages
- Error budget tracked monthly

## 5) Security Model

- API key hashing with per-key salt and server-side pepper
- Tier-based quotas (`free`, `pro`, `internal`)
- Redis-backed rate limiting by API key + request fingerprint
- Abuse detection signals: burst rate, invalid-key ratio, endpoint mix anomalies
- Admin role model with audited admin operations
- Secret management via Render environment variables only

## 6) Observability

### Structured logging schema

```json
{
  "timestamp": "2026-03-03T12:00:00Z",
  "level": "INFO",
  "request_id": "req_01JXYZ...",
  "endpoint": "/api/rewrite",
  "status_code": 200,
  "api_key_id": "key_abc123",
  "provider": "google_ai",
  "fallback_used": false,
  "latency_ms": 312,
  "error_category": null
}
```

### Metrics tracked

- p50 and p95 latency by endpoint
- rewrite provider success rate
- fallback activation rate
- drift score trend (PSI / JS)
- error rate by category
- cache hit ratio

### Health endpoints

- `/health`: fast app health + model readiness
- `/health/deep`: app + Postgres + Redis + provider probe

### Request tracing

- Every request carries a `request_id`
- `request_id` appears in logs, events, and response metadata

## 7) API Overview

Current endpoints remain backward-compatible.

### Example request

```bash
curl -X POST http://localhost:5000/api/rewrite \
  -H "Content-Type: application/json" \
  -H "X-API-Key: sk-..." \
  -d '{
    "text": "this app is bad and slow",
    "style": "friendly",
    "target_tone": "positive"
  }'
```

### Example response

```json
{
  "success": true,
  "result": {
    "original_text": "this app is bad and slow",
    "style": "friendly",
    "target_tone": "positive",
    "corrected_text": "This app is bad and slow.",
    "rewritten_text": "Good news, this app is improving and steadily improving in speed.",
    "notes": "...",
    "sentiment": "positive",
    "confidence": 0.88,
    "provider": "Local Fallback Rewriter",
    "model": "local-rules-v2"
  },
  "meta": {
    "request_id": "req_01J...",
    "fallback_used": true,
    "error_category": null
  }
}
```

### Quota headers example

```text
X-Quota-Plan: free
X-Quota-Remaining: 482
X-Quota-Reset: 2026-03-04T00:00:00Z
X-RateLimit-Remaining: 19
```

### Error payload example

```json
{
  "success": false,
  "error": "Too Many Requests",
  "message": "20 per 1 minute",
  "error_category": "quota_error",
  "request_id": "req_01J..."
}
```

### Versioning strategy

- Keep existing routes stable
- Additive fields only for backwards compatibility
- Introduce explicit `/v1` namespace for future major revisions

## 8) Local Development

### Quickstart (3 steps)

1. Install dependencies

```bash
python3 -m pip install -r requirements.txt
```

2. Configure environment and train local model

```bash
cp .env.example .env
python3 train_model.py
```

3. Run API and worker

```bash
# web
python3 server.py

# worker (requires Redis + rq package)
rq worker -u redis://localhost:6379 text-intel
```

### Required environment variables

```bash
SECRET_KEY=...
ADMIN_TOKEN=...
GOOGLE_AI_STUDIO_API_KEY=...
GOOGLE_AI_MODEL=gemini-flash-latest
GOOGLE_AI_FALLBACK_MODEL=gemini-2.5-flash
RATE_LIMIT_DEFAULT=20 per minute
RATE_LIMIT_REWRITE=30 per minute
RATE_LIMIT_STORAGE_URI=redis://localhost:6379/0
DATABASE_URL=postgresql://...
REDIS_URL=redis://...
API_KEY_PEPPER=...
```

## 9) Deployment (Render)

### Required services

- Web Service: Flask + Gunicorn
- Worker Service: RQ worker
- Postgres add-on
- Redis add-on

### Scaling model

- Horizontal scale web instances behind Render load balancer
- Separate worker scaling based on queue depth
- Redis + Postgres remain shared state services

### Rollback strategy

- Feature flags for DB-read switch, quotas, and drift jobs
- Dual-write phase before read cutover
- Rollback by toggling flags without API contract changes

## 10) Metrics and Performance

Example reliability scorecard format (SLO-driven):

- p95 `/api/sentiment`: `< 200ms` (local inference path)
- p95 `/api/rewrite`: `< 1500ms` healthy provider path
- Fallback recovery rate: `> 99%` during provider outage windows
- Provider success rate: tracked hourly and daily
- Drift monitoring frequency: hourly aggregates + daily snapshot
- Error budget model: monthly budget tied to 99.9% availability target

## 11) Roadmap

- Full Postgres migration with dual-write and read-cutover flags
- Redis-backed quota engine with tiered plans
- Drift alert thresholds and anomaly notification hooks
- Billing integration (Stripe-ready entitlement model)
- Multi-region readiness and queue partitioning

## 12) Interview Positioning

This project demonstrates depth in production ML systems beyond model accuracy:

- Reliable inference orchestration under dependency failures
- Deterministic fallback design and graceful degradation
- Data-driven reliability via metrics, logs, and drift signals
- Security and quota governance required for SaaS monetization
- Deployment realism on Render with web/worker/cache/database topology

If you are reviewing this as an ML infrastructure project, the core value is operational rigor: predictable behavior, controlled failure modes, and measurable service quality at runtime.

## License

MIT
