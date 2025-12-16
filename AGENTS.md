# Repository Guidelines

## Project Structure & Modules
- `api/`: FastAPI app (`api/main.py`) with routes in `api/routes`, schemas in `api/models`, and DB pool helpers in `api/db_pool.py`.
- `scripts/`: Core recommendation and forecasting logic plus diagnostics (see `scripts/forecasting/core/` for engines and analyzers). Use these for batch jobs and experiments.
- `workers/`: Production worker entrypoints (e.g., `workers/weekly_recommendation_worker.py`) mirroring the scripts used by schedulers.
- `docs/`: Algorithm, worker, and API references; skim `docs/api/API_ENDPOINTS.md` before changing request/response shapes.
- `docker-compose.yml`, `Dockerfile.api`, `Dockerfile.worker`: Container entrypoints; `sample_forecasts/` holds reference outputs for spot checks.

## Setup, Build & Run
- Create a venv and install deps: `python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt`.
- Copy config: `cp .env.example .env`, then fill DB/Redis settings.
- Run API locally: `uvicorn api.main:app --reload --host 0.0.0.0 --port 8000`.
- Start with Docker (API + Redis + workers): `docker-compose up -d` (see `DOCKER_DEPLOYMENT.md` for tuning).
- Regenerate data pipelines manually when needed:
  - Recommendations: `python scripts/weekly_recommendation_worker.py`
  - Forecasts: `python scripts/forecasting/forecast_worker.py`

## Test & QA
- Integration smoke for forecasting: `python test_forecast.py 25367399` (requires live MSSQL + Redis).
- Targeted diagnostics: `python scripts/forecasting/test_rfm_accuracy.py` or `python scripts/validate_precision.py` for model checks.
- Prefer deterministic inputs; document product/customer IDs used for reproducibility. Ensure `.env` points to non-production data before running destructive tests.

## Coding Style & Naming
- Python 3.11+, PEP 8, 4-space indents. Use type hints and Pydantic models for request/response validation.
- Naming: snake_case for functions/vars/files; PascalCase for classes and Pydantic schemas.
- Logging: prefer `logging` (INFO for flow, WARNING/ERROR for issues); avoid `print`. Preserve structured payload fields returned by the API and workers.
- Configuration comes from environment variables; do not hardcode secrets or hostnames.

## Commit & PR Guidelines
- Commits are short, imperative, and scoped (e.g., `Fix forecast API params error`, `Add worker management scripts`). Keep one logical change per commit when possible.
- PRs: include what/why, affected endpoints or workers, test evidence (commands + results), and any data/backfill steps. Link issues or tickets. Add before/after samples for payload changes (see `docs/api/API_ENDPOINTS.md` for format).

## Security & Configuration Tips
- Never commit `.env` or credentials; rely on environment variables and local secrets stores.
- Validate external connections (MSSQL/Redis) before running batch jobs; cache TTL is controlled via `CACHE_TTL`.
- When sharing logs, redact customer/product identifiers and connection strings.***
