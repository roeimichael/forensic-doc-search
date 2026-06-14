# Thin wrappers around the documented commands. Recipes use TABS (Make requirement).
# On Windows without `make`, the README lists the equivalent raw commands.
.PHONY: install up down gen ingest serve ui eval test run

install:
	pip install -r requirements.txt && pip install -e .

up:
	docker compose up -d

down:
	docker compose down

gen:
	rag generate

ingest:
	python scripts/wait_for_qdrant.py && rag ingest

serve:
	uvicorn ragforce.api.app:app --host 0.0.0.0 --port 8000

ui:
	streamlit run ui/streamlit_app.py

eval:
	rag eval

test:
	pytest -q

# One-shot (command #3 of the ≤3-command flow): wait for Qdrant, generate the
# synthetic corpus, ingest it idempotently, then serve the search API.
run:
	python scripts/wait_for_qdrant.py
	rag generate
	rag ingest
	uvicorn ragforce.api.app:app --host 0.0.0.0 --port 8000
