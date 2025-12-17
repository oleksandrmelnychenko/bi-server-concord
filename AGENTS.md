# Repository Guidelines

## Project Structure & Module Organization
- `api/`: FastAPI surface (`api/main.py`), request/response schemas in `api/models/`, DB pool in `api/db_pool.py`.
- `scripts/`: Core logic. Recommendations in `scripts/improved_hybrid_recommender_v33.py` (ANN + hybrid heuristics); forecasting engine in `scripts/forecasting/core/` plus model selector. Diagnostics live beside them.
- `workers/`: Production entrypoints (e.g., `workers/weekly_recommendation_worker.py`, `scripts/forecasting/forecast_worker.py`) used by schedulers/cron.
- `docs/`: Algorithm notes, worker guides, and API reference (`docs/api/API_ENDPOINTS.md`). `Deployment/` and Dockerfiles cover packaging; `sample_forecasts/` holds reference outputs.

## Build, Test, and Development Commands
- Install deps (recommended venv):  
  ```bash
  python3 -m venv .venv && source .venv/bin/activate
  pip install -r requirements.txt
  cp .env.example .env  # fill MSSQL/Redis creds
  ```
- Run API locally: `uvicorn api.main:app --reload --host 0.0.0.0 --port 8000`.
- Run workers by hand:  
  - Recommendations: `python scripts/weekly_recommendation_worker.py`  
  - Forecasts: `python scripts/forecasting/forecast_worker.py`
- Build ANN neighbors for discovery if data changes:  
  `python scripts/build_ann_neighbors.py` (configure MSSQL_* env vars and ANN_OUTPUT_PATH).

## Testing Guidelines
- Forecast end-to-end smoke: `python test_forecast.py 25432060` (requires DB + Redis).  
- Forecast unit probes: `python scripts/forecasting/test_worker.py` or `python scripts/forecasting/test_rfm_accuracy.py`.  
- Recommendation precision check: `python scripts/validate_precision.py` (document IDs/date used).  
- Prefer reproducible inputs; never run against production data without explicit approval.

## Coding Style & Naming Conventions
- Python 3.11+, PEP 8, 4-space indents, type hints. Pydantic models for all API payloads.  
- SQL: favor parameterized queries (`%s` placeholders with pymssql) over f-strings; keep filters for `Deleted=0` and `IsForSale=1`.  
- Naming: snake_case for vars/functions/files, PascalCase for classes/schemas. Logging via `logging` (INFO for flow, WARNING/ERROR for issues). Avoid hardcoded secrets—use env vars and `.env`.

## Commit & Pull Request Guidelines
- Commits: imperative, scoped, and reviewable (e.g., `Harden ANN candidate filters`, `Add ETS fallback to forecast`). One logical change per commit when possible.
- PRs: state problem/approach, list endpoints or workers touched, include test commands + results, note data migrations/backfills, and attach sample payload diffs if schema fields shift.

## Security & Configuration Tips
- Never commit credentials or ANN output files derived from customer data.  
- Verify DB/Redis connectivity before long jobs; cache TTL is set via `CACHE_TTL`.  
- When sharing logs, redact customer/product IDs and connection strings.***
