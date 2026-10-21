> Historical planning snapshot from June 2026, preserved from branch `client-recommendation` during repository cleanup on 2026-10-21. This is not the current deployment or implementation guide; current service code is maintained in `gba-ai-services`.

# GBA Recommendation & Procurement Services — Master Plan

> Status: **living plan**. Two ground-up Python services, orchestrated by gba-server (development branch).
> Frame: June 2026, best-available techniques. Build fresh; copy/fix/tune functionality from the
> 7-month-old `bi-server-concord` prototype where it's sound. Update this doc before code.

---

## 1. Goal & scope

Build **two** production Python microservices, consumed by **gba-server (.NET, development branch)**:

1. **Client Recommendation Service** — for each B2B client, what products to (re)buy.
   *Evolves the existing `bi-server-concord` V3.2 hybrid recommender.*
2. **Procurement / Replenishment Service** — for each producer/supplier, what to order and when.
   *New build: demand forecasting + inventory replenishment. The prototype has NO procurement side.*

Out of scope for now: the AI-BI text-to-SQL engine (`ai-bi-server`) — separate track, parked.

```
                   ┌────────────────────────────────────────────┐
   gba-server  ───▶│  Service 1: Client Recommendations (FastAPI)│──▶ ConcordDb_V5 (read)
   (.NET, dev)     │  Service 2: Procurement Forecast (FastAPI)  │──▶ ConcordDb_V5 (read)
   orchestrator    └────────────────────────────────────────────┘
        │                         │ both: Redis cache + background workers
        └── existing .NET DTOs (ClientPredictionApiModels) + RecommendationAPI config already stubbed
```

---

## 2. What we inherit from the prototype (asset audit)

**Reusable (copy + fix + tune):**
- V3.2 hybrid algorithm: segment-aware repurchase (frequency×recency) + collaborative discovery (Jaccard) + diversity. Validated 75.4% precision@50 (Heavy 89% / Regular 88% / Light 54%).
- Customer segmentation logic (HEAVY/REGULAR-CONSISTENT/REGULAR-EXPLORATORY/LIGHT by order count + repurchase rate).
- Redis caching + weekly pre-compute worker pattern.
- Docker setup (FastAPI + Redis + worker, MS ODBC driver baked).
- The data access path: `ClientAgreement → Order → OrderItem` is the correct sales spine.

**Must discard / rebuild (prototype debt — see CODE_AUDIT below):**
- Historical prototype contained committed database credentials. Credential values are intentionally omitted from this archived document.
- Dead LightFM+DuckDB+Dagster layer (`api/routes/recommendations.py`, `api/models/recommendation_schemas.py`, imports a non-existent `ml.models`).
- **Two divergent workers** (`workers/` uses V32 + key `recommendations:customer:*`; `scripts/` uses V31 + key `weekly_recs:*`) and **three inconsistent Redis key schemes** — API reads keys the workers don't all write. Pick ONE.
- SQL built by f-string interpolation everywhere — move to parameterized queries.
- Contract drift: README/main.py say `score/rank/source/precision_estimate`; the Pydantic models say `product_name/reason/metadata`. The .NET DTO matches main.py. Lock ONE contract.

---

## 3. Data foundation (verified against live ConcordDb_V5, 2026-06)

**Sales spine (Service 1):** `ClientAgreement → [Order] → OrderItem(ProductID, Qty)`; product groups via `ProductProductGroup`.

**Procurement spine (Service 2) — verified tables + row counts:**
- `SupplyOrder` (4,131) — orders to producers. KEY columns: `OrganizationID` (producer), `Created` (order date), `OrderShippedDate`, `ShipArrived`/`PlaneArrived`/`VechicalArrived`, `OrderArrivedDate`, `CompleteDate`, `TransportationType`. → **real per-producer lead-time is computable** (Created → OrderArrivedDate).
- `SupplyOrderItem` (30,356) — `ProductID, Qty, UnitPrice`.
- `SupplyOrganization` (352) — producers/suppliers; `SupplyOrganizationAgreement` (313).
- `ProductIncome` (4,545) / `ProductIncomeItem` (30,999) — actual goods receipts.
- `ProductAvailability` (17,286) — current stock (`Amount` per `ProductID`+`StorageID`).
- `ProductReservation` (1,258) — reserved stock (open demand).
- `Consignment`/`ConsignmentItem` (31,200) + `ConsignmentItemMovement` (32,085) — stock movements.

This is enough for demand forecasting + lead-time-aware replenishment. No external data needed for v1.

---

## 4. Service 1 — Client Recommendations (evolve)

**Mission:** per client, top-N products to buy (repurchase + discovery), with reasons.

**v1 (parity + hardening):** port V3.2 clean — parameterized SQL, env secrets, one Redis key scheme, one worker, one contract, tests. Lock the contract to the .NET DTO (`customer_id, product_id, score, rank, segment, source, precision_estimate, latency_ms, cached`).

**v2 (2026 techniques — candidates, each A/B'd on a precision benchmark before adoption):**
- **Implicit-feedback matrix factorization** (ALS / BPR via `implicit`, or LightFM with item features) to replace hand-weighted scores — learns latent client↔product affinity; the prototype already gestured at LightFM.
- **Item-2-vec / co-purchase embeddings** for "bought together" discovery (sequential baskets → embeddings), stronger than raw Jaccard.
- **Time-aware repurchase intervals**: model each client×product reorder cycle (e.g. expected days-to-next-buy) → recommend what's *due now*, not just frequent. B2B parts have natural replenishment cycles — high-value signal.
- **Two-stage retrieve→rank**: cheap candidate gen (co-purchase + repurchase) → gradient-boosted ranker (LightGBM) on features (recency, frequency, monetary, seasonality, price, stock).
- **Cold-start (LIGHT, 54% today)**: content features (product group, analogues, producer) + popularity priors.
- **Offline eval harness**: time-split precision@K / recall@K / MAP, per segment — the benchmark that gates every change.

**Key insight (carried from the AI-BI track):** measure every change against a held-out benchmark; intuition misleads. Build the eval harness FIRST.

---

## 5. Service 2 — Procurement / Replenishment (new build)

**Mission:** per producer (and per product), recommend WHAT to order, HOW MUCH, and WHEN — so stock covers forecast demand over the producer's lead time without over-stocking.

**Pipeline:**
1. **Demand forecasting** per product (and per storage): from `Order/OrderItem` history → forecast units/week.
   - 2026 stack candidates (A/B'd): statistical baselines (Croston/SBA for intermittent B2B demand — most parts sell sporadically, classic forecasters fail here), exponential smoothing (statsforecast), and ML (LightGBM global model with lags/seasonality/price). Intermittent-demand-aware is essential for spare parts.
2. **Lead-time model** per producer: empirical distribution of `OrderArrivedDate − Created` from `SupplyOrder` history (mean + variance → safety stock).
3. **Inventory position**: on-hand (`ProductAvailability.Amount`) − reserved (`ProductReservation`) + on-order (open `SupplyOrder` not yet arrived).
4. **Replenishment policy**: reorder point = demand-over-leadtime + safety stock (service-level z×σ); order-up-to / EOQ for quantity. Group by producer into a single suggested purchase order.
5. **Output**: suggested supply orders — per producer: list of (product, suggested_qty, reason, urgency, days-of-cover-left, forecast).

**Why this is genuinely new:** the prototype has zero procurement/forecasting. But the *data* (lead times, receipts, stock, reservations) is all present and rich.

---

## 6. Shared architecture (both services, 2026 baseline)

- **Lang/stack:** Python 3.12, FastAPI, Pydantic v2, `uv` for deps (fast, lockfile), `ruff` lint.
- **DB access:** read-only SQL login on ConcordDb_V5 (NEVER `sa`), SQLAlchemy core + parameterized queries, connection pool.
- **Cache:** Redis; one documented key scheme per service.
- **Workers:** background pre-compute (APScheduler or cron) → Redis; idempotent, resumable, observable.
- **Contracts:** versioned OpenAPI; lock JSON to match the existing .NET DTOs so gba-server integration is drop-in.
- **Eval:** offline benchmark harness per service (time-split), gating every algorithm change. Non-negotiable.
- **Ops:** Docker compose per service; structured logging; `/health` `/metrics`; secrets via env only; `.env.example` with NO real values.
- **Repo layout:** likely a fresh repo (or monorepo with `services/recommendations` + `services/procurement` + shared `libs/`), NOT continuing the messy prototype tree.

---

## 7. Open decisions (need answers before building)

1. **Repo strategy:** fresh monorepo for both services, or two separate repos? (Prototype repo is too messy to continue in-place — recommend fresh, copy code over.)
2. **DB credentials:** can we get a dedicated **read-only** SQL login on ConcordDb_V5 (dev)? (The prototype's `sa`/leaked creds must die.)
3. **gba-server integration shape:** .NET already has `ClientPredictionApiModels` + `RecommendationAPI` config (URL/Recommendations/Batch/SalesForecast endpoints). Service 2 likely maps to `SalesForecastEndpoint`. Confirm the .NET side is the integration target and we honor those DTOs.
4. **Order semantics:** is `ClientAgreement.ClientID` the right client grain, or should we roll up `MainClientID/RootClientID` (client hierarchy)? Affects both services.
5. **Build order:** Service 1 first (we understand it, faster win) then Service 2? Or scaffold both, shared libs first?

---

## 8. Proposed phasing

- **Phase 0 — Foundations:** fresh repo + shared libs (db, cache, config, eval harness skeleton), read-only DB login, secrets hygiene. Lock contracts to .NET DTOs.
- **Phase 1 — Service 1 v1 (parity, clean):** port V3.2 properly; eval harness; reproduce ~75% precision on a time-split benchmark; deploy behind gba-server.
- **Phase 2 — Service 2 v1 (replenishment baseline):** lead-time model + Croston/SBA demand + reorder-point policy; per-producer suggested POs; eval (forecast accuracy, stockout/overstock backtests).
- **Phase 3 — v2 quality:** Service 1 → MF/ranker + reorder-cycle; Service 2 → ML global forecaster; both gated by benchmarks.
- **Phase 4 — Productionize:** monitoring, A/B in gba-server, feedback loops (did the client buy the rec? did the PO prevent stockout?).
