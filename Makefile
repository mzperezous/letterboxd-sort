# Common dev loop. `make help` lists targets.
.DEFAULT_GOAL := help
export DAGSTER_HOME ?= $(CURDIR)/.dagster

help:  ## Show targets
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-14s %s\n", $$1, $$2}'

install:  ## Install Python deps (incl. dev + dagster)
	uv sync --all-extras

up:  ## Start Postgres (add PROFILE=s3 for MinIO)
	docker compose -f infra/docker-compose.yml $(if $(PROFILE),--profile $(PROFILE)) up -d --wait

down:  ## Stop services
	docker compose -f infra/docker-compose.yml --profile s3 down

ingest:  ## Bootstrap: TMDB export + MovieLens + seed catalog
	uv run reel tmdb export
	-uv run reel movielens
	uv run reel silver ml_movie
	uv run reel tmdb seed

build:  ## Silver tables, models, clusters, publish to Postgres
	uv run reel silver
	uv run reel ml train
	uv run reel ml cluster
	uv run reel db publish

refresh:  ## Daily CDC: new ids + TMDB changes, then rebuild
	uv run reel tmdb export
	uv run reel tmdb changes
	$(MAKE) build

api:  ## Run the API on :8000
	uv run uvicorn reel.api.app:app --reload

dagster:  ## Dagster UI on :3000
	mkdir -p $(DAGSTER_HOME)
	uv run dagster dev -m reel.pipelines.definitions

test:  ## Fast tests (no network; postgres tests run if REEL_TEST_PG_DSN is set)
	uv run pytest -q

lint:  ## Ruff lint + format check
	uv run ruff check src tests
	uv run ruff format --check src tests

fmt:  ## Auto-format
	uv run ruff check --fix src tests
	uv run ruff format src tests
